"""PhaseNet: a deep-neural-network-based seismic arrival-time picking method.

Zhu & Beroza, Geophysical Journal International 216(1):261-273 (2019), doi:10.1093/gji/ggy423
(arXiv 1803.03211).

Port of the official TensorFlow code, AI4EPS/PhaseNet at commit
``62005c6195638f88d077f8870ea895971062715a``: ``phasenet/model.py`` (``UNet``, ``crop_and_concat``,
``ModelConfig``), ``phasenet/data_reader.py`` (``normalize``) and ``phasenet/util.py``
(``detect_peaks_thread``). MIT License, Copyright (c) 2021 Weiqiang Zhu.

The network is a 1-D U-Net over three-component waveforms ``(batch, 3, samples)`` that returns
per-sample probabilities of noise, P and S (softmax over the three classes). With the released
configuration (``model/190703-214543/config.log``: depths 5, filters_root 8, kernel 7, pool 4) the
channel widths are 8, 16, 32, 64, 128 and the model has 268,443 trainable parameters. The paper's
Figure 5 shows widths 8, 11, 16, 22, 32 and no batch normalisation; the port follows the code and the
released checkpoint, which use 8-128 with batch normalisation.

Implementation notes:

- Module and parameter names mirror the TensorFlow variable scopes (``Input/input_conv``,
  ``DownConv_<d>/down_conv1_<d+1>``, ``UpConv_<d>/up_conv0_<d+1>``, ``Output/output_conv``, ...), so a
  TensorFlow variable ``A/b/kernel`` becomes the PyTorch parameter ``A.b.weight`` (see
  :func:`state_dict_from_tf_checkpoint`).
- TensorFlow ``"same"`` padding is reproduced exactly, including its asymmetric split for the stride-4
  convolutions (more padding on the right) and the ``"same"`` transposed convolutions (output length
  ``stride * input``); skip connections crop the up-sampled branch at offset ``(n_up - n_skip) // 2``
  like ``crop_and_concat``. Any input length works, as in the official code.
- Batch normalisation uses TensorFlow's defaults (epsilon 1e-3, moving-average momentum 0.99, i.e.
  PyTorch momentum 0.01). Weights are initialised like the reference (Glorot-uniform kernels
  ``VarianceScaling(1.0, "fan_avg", "uniform")``, zero biases); TensorFlow's random draws cannot be
  reproduced in PyTorch.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._pretrained import cached_download
from ._seismic import TracePicks, WaveformPicker, as_sequence, check_waveforms, picks_from_peaks, std_normalize
from ._tf_checkpoint import read_tf_checkpoint

# Official checkpoint model/190703-214543 (config.log above), raw files pinned to the reference commit.
_PHASENET_RAW = "https://raw.githubusercontent.com/AI4EPS/PhaseNet/62005c6195638f88d077f8870ea895971062715a/model/190703-214543/"
PHASENET_WEIGHTS: Dict[str, Dict[str, object]] = {
    "original": {
        "checkpoint": "model_95.ckpt",
        "files": {
            "model_95.ckpt.index": "f96b553b76be4ebae9a455eaf8d83cfa8c0e110f06cfba958de2568e5b6b2780",
            "model_95.ckpt.data-00000-of-00001": "9ee2c15dd78fb15de45a55ad64a446f1a0ced152ba4ac5c506d82b9194da85b4",
        },
        "license": "MIT (AI4EPS/PhaseNet)",
    },
}


def _tf_same_padding(length: int, kernel_size: int, stride: int) -> tuple:
    out = math.ceil(length / stride)
    total = max((out - 1) * stride + kernel_size - length, 0)
    return total // 2, total - total // 2


class _TFConv1d(nn.Conv1d):
    """``tf.compat.v1.layers.conv2d`` with ``padding="same"`` on ``(batch, channels, samples)``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = _tf_same_padding(x.shape[-1], self.kernel_size[0], self.stride[0])
        return F.conv1d(F.pad(x, (left, right)), self.weight, self.bias, self.stride)


class _TFConvTranspose1d(nn.ConvTranspose1d):
    """``tf.compat.v1.layers.conv2d_transpose`` with ``padding="same"`` (output ``stride * length``)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1] * self.stride[0]
        left, _ = _tf_same_padding(length, self.kernel_size[0], self.stride[0])
        full = F.conv_transpose1d(x, self.weight, self.bias, self.stride)
        return full[..., left : left + length]


def _tf_batch_norm(channels: int) -> nn.BatchNorm1d:
    return nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)


def _glorot_uniform_(weight: torch.Tensor, fan_in: int, fan_out: int) -> None:
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    with torch.no_grad():
        weight.uniform_(-limit, limit)


class PhaseNet(nn.Module, WaveformPicker):
    """PhaseNet U-Net: ``(batch, in_channels, samples)`` waveforms -> ``(batch, classes, samples)``.

    ``forward`` returns softmax probabilities of (noise, P, S); ``forward(x, logits=True)`` returns
    the pre-softmax output that the official cross-entropy loss uses. :meth:`annotate` adds the
    official per-window normalisation, :meth:`extract_picks` the official peak picking.
    """

    sampling_rate = 100.0
    component_order = "ENZ"
    output_names = ("noise", "P", "S")
    phase_channels = {"P": 1, "S": 2}

    def __init__(
        self,
        in_channels: int = 3,
        classes: int = 3,
        depths: int = 5,
        filters_root: int = 8,
        kernel_size: int = 7,
        pool_size: int = 4,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        for label, value in (("in_channels", in_channels), ("classes", classes), ("depths", depths),
                             ("filters_root", filters_root), ("kernel_size", kernel_size), ("pool_size", pool_size)):
            if int(value) < 1:
                raise ValueError(f"PhaseNet {label} must be positive, got {value}.")
        if not 0.0 <= drop_rate < 1.0:
            raise ValueError(f"PhaseNet drop_rate must be in [0, 1), got {drop_rate}.")
        self.in_channels = int(in_channels)
        self.classes = int(classes)
        self.depths = int(depths)
        self.filters_root = int(filters_root)
        self.kernel_size = int(kernel_size)
        self.pool_size = int(pool_size)
        self.drop_rate = float(drop_rate)
        k, s = self.kernel_size, self.pool_size

        # Modules are created in the order of the TensorFlow graph (UNet.add_prediction_op).
        inputs = nn.Module()
        inputs.add_module("input_conv", _TFConv1d(in_channels, filters_root, k, bias=True))
        inputs.add_module("input_bn", _tf_batch_norm(filters_root))
        self.add_module("Input", inputs)
        for depth in range(self.depths):
            filters = int(2**depth * filters_root)
            previous = filters_root if depth == 0 else int(2 ** (depth - 1) * filters_root)
            block = nn.Module()
            block.add_module(f"down_conv1_{depth + 1}", _TFConv1d(previous, filters, k, bias=False))
            block.add_module(f"down_bn1_{depth + 1}", _tf_batch_norm(filters))
            if depth < self.depths - 1:
                block.add_module(f"down_conv3_{depth + 1}", _TFConv1d(filters, filters, k, stride=s, bias=False))
                block.add_module(f"down_bn3_{depth + 1}", _tf_batch_norm(filters))
            self.add_module(f"DownConv_{depth}", block)
        for depth in range(self.depths - 2, -1, -1):
            filters = int(2**depth * filters_root)
            block = nn.Module()
            block.add_module(f"up_conv0_{depth + 1}", _TFConvTranspose1d(2 * filters, filters, k, stride=s, bias=False))
            block.add_module(f"up_bn0_{depth + 1}", _tf_batch_norm(filters))
            block.add_module(f"up_conv1_{depth + 1}", _TFConv1d(2 * filters, filters, k, bias=False))
            block.add_module(f"up_bn1_{depth + 1}", _tf_batch_norm(filters))
            self.add_module(f"UpConv_{depth}", block)
        outputs = nn.Module()
        outputs.add_module("output_conv", _TFConv1d(filters_root, classes, 1, bias=True))
        self.add_module("Output", outputs)
        self.dropout = nn.Dropout(self.drop_rate)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Glorot-uniform kernels (fans of the TensorFlow kernel shape), zero biases, BN 1/0."""
        for module in self.modules():
            if isinstance(module, (_TFConv1d, _TFConvTranspose1d)):
                kernel = module.weight.shape[-1]
                if isinstance(module, _TFConvTranspose1d):  # TF kernel (k, 1, out, in)
                    fan_in, fan_out = kernel * module.out_channels, kernel * module.in_channels
                else:  # TF kernel (k, 1, in, out)
                    fan_in, fan_out = kernel * module.in_channels, kernel * module.out_channels
                _glorot_uniform_(module.weight, fan_in, fan_out)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                module.reset_parameters()

    def _block(self, name: str) -> nn.Module:
        return getattr(self, name)

    def _conv_bn_relu(self, x: torch.Tensor, block: nn.Module, conv: str, bn: str) -> torch.Tensor:
        x = getattr(block, bn)(getattr(block, conv)(x))
        return self.dropout(F.relu(x))

    def forward(self, x: torch.Tensor, logits: bool = False) -> torch.Tensor:
        check_waveforms(x, self.in_channels, "PhaseNet")
        if x.shape[-1] < 1:
            raise ValueError("PhaseNet needs at least one sample.")
        inputs = self._block("Input")
        net = self._conv_bn_relu(x, inputs, "input_conv", "input_bn")
        skips: List[torch.Tensor] = []
        for depth in range(self.depths):
            block = self._block(f"DownConv_{depth}")
            net = self._conv_bn_relu(net, block, f"down_conv1_{depth + 1}", f"down_bn1_{depth + 1}")
            skips.append(net)
            if depth < self.depths - 1:
                net = self._conv_bn_relu(net, block, f"down_conv3_{depth + 1}", f"down_bn3_{depth + 1}")
        for depth in range(self.depths - 2, -1, -1):
            block = self._block(f"UpConv_{depth}")
            net = self._conv_bn_relu(net, block, f"up_conv0_{depth + 1}", f"up_bn0_{depth + 1}")
            skip = skips[depth]
            offset = (net.shape[-1] - skip.shape[-1]) // 2
            net = torch.cat([skip, net[..., offset : offset + skip.shape[-1]]], dim=1)
            net = self._conv_bn_relu(net, block, f"up_conv1_{depth + 1}", f"up_bn1_{depth + 1}")
        out = self._block("Output").output_conv(net)
        return out if logits else torch.softmax(out, dim=1)

    def annotate(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Per-sample (noise, P, S) probabilities of raw windows, normalised like the official reader."""
        return self(std_normalize(waveforms))

    def extract_picks(
        self,
        annotations: torch.Tensor,
        p_threshold: float = 0.5,
        s_threshold: float = 0.5,
        min_distance: Optional[int] = None,
        **kwargs,
    ) -> List[TracePicks]:
        """Official peak picking: ``detect_peaks(prob, mph=threshold, mpd=0.5 s)`` for P and S.

        Thresholds 0.5 are the paper's ("peak probabilities above 0.5 are counted as positive picks");
        ``phasenet/util.py`` uses the same 0.5 and ``mpd = 0.5 / dt`` (50 samples at 100 Hz).
        """
        if kwargs:
            raise TypeError(f"Unexpected PhaseNet pick parameters: {sorted(kwargs)}.")
        distance = int(round(0.5 * self.sampling_rate)) if min_distance is None else int(min_distance)
        thresholds = as_sequence({"P": p_threshold, "S": s_threshold}, ("P", "S"))
        return picks_from_peaks(annotations, self.phase_channels, thresholds, distance)


def state_dict_from_tf_checkpoint(variables: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """PhaseNet state dict from TensorFlow checkpoint variables (optimizer slots are ignored)."""
    renames = {"kernel": "weight", "bias": "bias", "gamma": "weight", "beta": "bias",
               "moving_mean": "running_mean", "moving_variance": "running_var"}
    state: Dict[str, torch.Tensor] = {}
    for name, value in variables.items():
        parts = name.split("/")
        if len(parts) != 3 or parts[0] not in ("Input", "Output") and not parts[0].startswith(("DownConv_", "UpConv_")):
            continue  # global_step, beta1_power, ...
        if parts[2] not in renames:
            continue  # Adam slots: <var>/Adam, <var>/Adam_1 have four components
        tensor = torch.from_numpy(np.ascontiguousarray(value))
        if parts[2] == "kernel":  # (k, 1, in, out) -> (out, in, k); transposed convs: (k, 1, out, in) -> (in, out, k)
            tensor = tensor[:, 0].permute(2, 1, 0).contiguous()
        key = f"{parts[0]}.{parts[1]}.{renames[parts[2]]}"
        state[key] = tensor
        if key.endswith(".running_var"):  # PyTorch's BatchNorm counter, absent in TensorFlow
            state[key[: -len("running_var")] + "num_batches_tracked"] = torch.tensor(0, dtype=torch.long)
    return state


def phasenet_checkpoint_prefix(name: str = "original") -> Path:
    """Local prefix of an official TensorFlow checkpoint, downloaded on first use."""
    if name not in PHASENET_WEIGHTS:
        raise ValueError(f"Unknown PhaseNet weights {name!r}; expected one of {sorted(PHASENET_WEIGHTS)}.")
    spec = PHASENET_WEIGHTS[name]
    path = None
    for filename, sha256 in spec["files"].items():  # type: ignore[union-attr]
        path = cached_download("phasenet", filename, [_PHASENET_RAW + filename], sha256)
    assert path is not None
    return path.parent / str(spec["checkpoint"])


def load_phasenet_tf_checkpoint(model: PhaseNet, prefix: Union[str, Path]) -> PhaseNet:
    """Load a PhaseNet TensorFlow checkpoint (``<prefix>.index`` + data file) with ``strict=True``."""
    model.load_state_dict(state_dict_from_tf_checkpoint(read_tf_checkpoint(prefix)), strict=True)
    return model


def phasenet_builder(
    task: str,
    in_channels: int = 3,
    classes: int = 3,
    depths: int = 5,
    filters_root: int = 8,
    kernel_size: int = 7,
    pool_size: int = 4,
    drop_rate: float = 0.0,
    pretrained: Optional[Union[bool, str, Path]] = None,
    **kwargs,
) -> PhaseNet:
    """PhaseNet for ``task="picking"``.

    ``pretrained``: ``True`` or ``"original"`` loads the official checkpoint ``190703-214543`` (MIT,
    downloaded from the pinned repository commit and checked by sha256); a path is read as a
    TensorFlow checkpoint prefix (``.../model_95.ckpt``).
    """
    kwargs.pop("name", None)
    if kwargs:
        raise TypeError(f"Unexpected PhaseNet arguments: {sorted(kwargs)}.")
    if task.lower() != "picking":
        raise ValueError(f"PhaseNet supports task='picking', got {task!r}.")
    model = PhaseNet(in_channels, classes, depths, filters_root, kernel_size, pool_size, drop_rate)
    if pretrained:
        name = "original" if pretrained is True else str(pretrained)
        if name in PHASENET_WEIGHTS:
            if (in_channels, classes, depths, filters_root, kernel_size, pool_size) != (3, 3, 5, 8, 7, 4):
                raise ValueError("The official PhaseNet checkpoint needs the default configuration.")
            prefix = phasenet_checkpoint_prefix(name)
        else:
            prefix = Path(name)
        load_phasenet_tf_checkpoint(model, prefix)
    return model


__all__ = [
    "PHASENET_WEIGHTS",
    "PhaseNet",
    "load_phasenet_tf_checkpoint",
    "phasenet_builder",
    "phasenet_checkpoint_prefix",
    "state_dict_from_tf_checkpoint",
]
