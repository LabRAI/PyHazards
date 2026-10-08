"""Generalized Phase Detection (GPD): a P / S / noise classifier for 4-s three-component windows.

Ross, Meier, Hauksson & Heaton, "Generalized Seismic Phase Detection with Deep Learning",
Bull. Seismol. Soc. Am. 108(5A):2894-2901, 2018 (https://doi.org/10.1785/0120180080, arXiv:1805.01075).

Port of the official release interseismic/generalized-phase-detection at commit
``ea81ef17d204797de6a99d277fd2b9407fc77df7`` (MIT License, Copyright (c) 2018 Zachary E. Ross):
the network is the Keras 2.2.2 ``Sequential`` stored in ``model_pol.json`` (whose weights are
``model_pol_best.hdf5``), and :meth:`GPD.annotate` / :meth:`GPD.extract_picks` follow the sliding-window
picker of ``gpd_predict.py``.

Network (input: 400 samples = 4 s at 100 Hz, channels N, E, Z): four blocks of ``Conv1D`` ('same'
padding; 32 / 64 / 128 / 256 filters, kernels 21 / 15 / 11 / 9) + ``BatchNormalization`` + ReLU +
``MaxPooling1D(2)``, ``Flatten`` (25 x 256), two ``Dense(200)`` + ``BatchNormalization`` + ReLU, and
``Dense(3)`` + softmax over the classes (P, S, noise). 1,741,003 trainable parameters.

Choices that differ from a literal transcription, without changing the network's numbers:

- Inputs are ``(batch, channels, samples)``; Keras uses ``(batch, samples, channels)``. Keras'
  ``Flatten`` is time-major on ``(25, 256)``, so the feature map is transposed before flattening and
  ``dense_1`` keeps the Keras row order (its kernel loads with a plain transpose).
- BatchNorm (:class:`KerasBatchNorm1d`) reproduces Keras 2.2.2 with the TensorFlow backend, the
  version that trained the released weights: ``eps=1e-3``; in training, outputs use the batch mean and
  biased variance (as PyTorch), but the moving statistics are updated with the variance scaled by
  ``n / (n - 1 - eps)`` and with TensorFlow's zero-debiased moving average (decay 0.99), so the first
  update replaces the stored statistics by the batch's. PyTorch's ``BatchNorm1d`` would instead use
  ``n / (n - 1)`` and a plain exponential average.
- Module names are the Keras layer names of ``model_pol.json`` (``conv1d_1``, ``batch_normalization_1``,
  ..., ``dense_3``), so every Keras weight has an obvious counterpart (:func:`load_gpd_keras_weights`).
- Initialisation follows Keras: Glorot-uniform kernels (``VarianceScaling(1, fan_avg, uniform)`` in the
  JSON), zero biases, BatchNorm ``gamma=1, beta=0``. Random draws cannot match TensorFlow's.
- The release wraps the network in Keras' multi-GPU ``Lambda``/``Concatenate`` layers; they only split
  the batch across GPUs and are not ported.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._pretrained import cached_download
from ._seismic import KerasBatchNorm1d, TracePicks, WaveformPicker

_REPO = "interseismic/generalized-phase-detection"
_COMMIT = "ea81ef17d204797de6a99d277fd2b9407fc77df7"

# Official weights (MIT, in the repository). Downloaded on first use, never bundled.
GPD_WEIGHTS: Dict[str, Dict[str, object]] = {
    "original": {
        "filename": "model_pol_best.hdf5",
        "urls": (f"https://raw.githubusercontent.com/{_REPO}/{_COMMIT}/model_pol_best.hdf5",),
        "sha256": "b6f1980f64c4e816d43759af7a54c93c1ec9dd3ad56793ce6a09aad66adf3868",
        "license": "MIT",
    },
}

_CONV_SPECS = ((32, 21), (64, 15), (128, 11), (256, 9))
_DENSE_UNITS = 200


class GPD(nn.Module, WaveformPicker):
    """GPD window classifier: ``(batch, 3, 400)`` waveforms -> ``(batch, 3)`` P / S / noise probabilities.

    Inputs are 4-s windows at 100 Hz in the channel order N, E, Z, each scaled by its maximum absolute
    amplitude (as :meth:`annotate` does). ``forward(x, logits=True)`` returns the values before the
    softmax.
    """

    sampling_rate = 100.0
    component_order = "NEZ"
    output_names = ("P", "S", "noise")
    phase_channels = {"P": 0, "S": 1}
    window_samples = 400
    pick_offset = 200  # gpd_predict.py: a window's pick time is its centre (half_dur = 2 s)

    def __init__(self, in_channels: int = 3, classes: int = 3):
        super().__init__()
        if in_channels <= 0 or classes <= 0:
            raise ValueError(f"in_channels and classes must be positive, got {in_channels} and {classes}.")
        self.in_channels = int(in_channels)
        self.classes = int(classes)
        channels = self.in_channels
        for index, (filters, kernel) in enumerate(_CONV_SPECS, start=1):
            setattr(self, f"conv1d_{index}", nn.Conv1d(channels, filters, kernel, padding=kernel // 2))
            setattr(self, f"batch_normalization_{index}", KerasBatchNorm1d(filters, zero_debias=True))
            channels = filters
        flat = (self.window_samples // 2 ** len(_CONV_SPECS)) * channels  # 25 x 256 = 6400
        self.dense_1 = nn.Linear(flat, _DENSE_UNITS)
        self.batch_normalization_5 = KerasBatchNorm1d(_DENSE_UNITS, zero_debias=True)
        self.dense_2 = nn.Linear(_DENSE_UNITS, _DENSE_UNITS)
        self.batch_normalization_6 = KerasBatchNorm1d(_DENSE_UNITS, zero_debias=True)
        self.dense_3 = nn.Linear(_DENSE_UNITS, self.classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Keras initialisation: Glorot-uniform kernels, zero biases, BatchNorm ones / zeros."""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                module.reset_parameters()

    def forward(self, x: torch.Tensor, logits: bool = False) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.in_channels or x.shape[2] != self.window_samples:
            raise ValueError(
                f"GPD expects windows shaped (batch, {self.in_channels}, {self.window_samples}) "
                f"(channels N, E, Z at 100 Hz), got shape {tuple(x.shape)}."
            )
        for index in range(1, len(_CONV_SPECS) + 1):
            conv = getattr(self, f"conv1d_{index}")
            norm = getattr(self, f"batch_normalization_{index}")
            x = F.max_pool1d(torch.relu(norm(conv(x))), 2)
        x = x.transpose(1, 2).flatten(1)  # Keras Flatten on (time, channels)
        x = torch.relu(self.batch_normalization_5(self.dense_1(x)))
        x = torch.relu(self.batch_normalization_6(self.dense_2(x)))
        x = self.dense_3(x)
        return x if logits else torch.softmax(x, dim=-1)

    def sliding_windows(self, waveforms: torch.Tensor, stride: int = 10) -> torch.Tensor:
        """``(batch, n_windows, channels, 400)`` windows, each divided by its max absolute amplitude.

        ``gpd_predict.py``: ``sliding_window(data, 400, stepsize=n_shift)`` per channel and
        ``tr_win / max(|tr_win|)`` over channels and samples. An all-zero window (where the official
        code divides by zero and returns NaN) is left at zero.
        """
        if waveforms.ndim != 3 or waveforms.shape[1] != self.in_channels:
            raise ValueError(
                f"GPD.annotate expects waveforms shaped (batch, {self.in_channels}, samples), "
                f"got shape {tuple(waveforms.shape)}."
            )
        if waveforms.shape[-1] < self.window_samples:
            raise ValueError(
                f"GPD.annotate needs at least {self.window_samples} samples, got shape {tuple(waveforms.shape)}."
            )
        if int(stride) < 1:
            raise ValueError(f"stride must be a positive number of samples, got {stride}.")
        windows = waveforms.unfold(-1, self.window_samples, int(stride)).transpose(1, 2)
        scale = windows.abs().amax(dim=(-2, -1), keepdim=True)
        return windows / torch.where(scale == 0, torch.ones_like(scale), scale)

    def annotate(self, waveforms: torch.Tensor, stride: int = 10, batch_size: int = 3000) -> torch.Tensor:
        """Sliding-window class probabilities ``(batch, 3, n_windows)`` of ``(batch, 3, samples)`` traces.

        Window ``i`` covers samples ``[i * stride, i * stride + 400)`` and is assigned to its centre,
        sample ``i * stride + 200``. Windows are normalised in the input's dtype and then cast to the
        model's dtype (``gpd_predict.py`` normalises in float64 and Keras casts to float32). Filtering
        (3-20 Hz band-pass in ``gpd_predict.py``) is left to the caller.
        """
        windows = self.sliding_windows(waveforms, stride=stride)
        batch, n_windows = windows.shape[:2]
        flat = windows.reshape(batch * n_windows, self.in_channels, self.window_samples)
        flat = flat.to(self.dense_3.weight.dtype)
        outputs = [self(flat[start : start + batch_size]) for start in range(0, flat.shape[0], int(batch_size))]
        probabilities = torch.cat(outputs) if outputs else flat.new_zeros((0, self.classes))
        return probabilities.reshape(batch, n_windows, self.classes).transpose(1, 2)

    def extract_picks(
        self,
        annotations: Union[torch.Tensor, np.ndarray],
        min_proba: float = 0.95,
        trigger_off: float = 0.1,
        stride: int = 10,
    ) -> List[TracePicks]:
        """Picks from :meth:`annotate` output with the ``gpd_predict.py`` rule.

        Per phase: ``trigger_onset(prob, min_proba, 0.1)`` over the window sequence; triggers with
        ``on == off`` are skipped; the pick is the window with the highest probability in
        ``prob[on:off]`` (``off`` excluded, as in the official slice), placed at
        ``window * stride + 200``. ``min_proba`` defaults to the repository's 0.95 (the paper reports
        detections at 0.98). Thresholds are compared in the dtype of the probabilities, as NumPy does
        for the official float32 Keras outputs.
        """
        from ..metrics.picking import trigger_onset

        values = annotations.detach().cpu().numpy() if isinstance(annotations, torch.Tensor) else np.asarray(annotations)
        if values.ndim != 3 or values.shape[1] < max(self.phase_channels.values()) + 1:
            raise ValueError(
                f"GPD.extract_picks expects annotations shaped (batch, {self.classes}, n_windows), "
                f"got shape {values.shape}."
            )
        picks: List[TracePicks] = []
        for trace in values:
            trace_picks: TracePicks = {}
            for phase, channel in self.phase_channels.items():
                prob = trace[channel]
                on_threshold = float(np.asarray(min_proba, dtype=prob.dtype))
                off_threshold = float(np.asarray(trigger_off, dtype=prob.dtype))
                phase_picks = []
                for on, off in trigger_onset(prob, on_threshold, off_threshold):
                    if off == on:
                        continue
                    window = int(np.argmax(prob[on:off])) + int(on)
                    phase_picks.append((float(window * int(stride) + self.pick_offset), float(prob[window])))
                trace_picks[phase] = phase_picks
            picks.append(trace_picks)
        return picks


_KERAS_NAME = re.compile(
    r"^(?P<layer>(conv1d|batch_normalization|dense)_\d+)(_\d+)?/(?P<weight>kernel|bias|gamma|beta|moving_mean|moving_variance):0$"
)
_WEIGHT_TARGETS = {
    "kernel": "weight",
    "bias": "bias",
    "gamma": "weight",
    "beta": "bias",
    "moving_mean": "running_mean",
    "moving_variance": "running_var",
}


def _keras_weight_arrays(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    import h5py

    arrays: Dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        root = handle["model_weights"] if "model_weights" in handle else handle

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                arrays[name] = np.asarray(obj)

        root.visititems(visit)
    return arrays


def load_gpd_keras_weights(model: GPD, path: Union[str, Path]) -> GPD:
    """Load a Keras GPD weight file (``model_pol_best.hdf5``) into ``model``; every tensor must match.

    Keras names (``sequential_1/conv1d_1_1/kernel:0``, ``.../batch_normalization_1_1/moving_mean:0``,
    ...) map to the module of the same Keras layer name; ``Conv1D`` kernels ``(k, in, out)`` and
    ``Dense`` kernels ``(in, out)`` are transposed to PyTorch layouts.
    """
    state = model.state_dict()
    expected = {key for key in state if not key.endswith("num_batches_tracked")}
    loaded: Dict[str, torch.Tensor] = {}
    for name, array in _keras_weight_arrays(path).items():
        match = _KERAS_NAME.match("/".join(name.split("/")[-2:]))
        if match is None:
            raise ValueError(f"Unexpected tensor {name!r} in {path}.")
        key = f"{match.group('layer')}.{_WEIGHT_TARGETS[match.group('weight')]}"
        if key not in expected:
            raise ValueError(f"Keras tensor {name!r} has no counterpart ({key!r}) in GPD.")
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        if match.group("weight") == "kernel":
            tensor = tensor.permute(2, 1, 0) if tensor.ndim == 3 else tensor.t()
        if tuple(tensor.shape) != tuple(state[key].shape):
            raise ValueError(f"{name!r} has shape {tuple(array.shape)}, which does not fit {key!r} {tuple(state[key].shape)}.")
        if key in loaded:
            raise ValueError(f"{path} holds more than one tensor for {key!r}.")
        loaded[key] = tensor.to(state[key].dtype).contiguous()
    missing = sorted(expected - set(loaded))
    if missing:
        raise ValueError(f"{path} lacks tensors for {missing}.")
    model.load_state_dict({**{k: v for k, v in state.items() if k.endswith("num_batches_tracked")}, **loaded}, strict=True)
    return model


def gpd_weights_path(name: str = "original") -> Path:
    """Local path of the official weights (downloaded and checked against the pinned sha256)."""
    if name not in GPD_WEIGHTS:
        raise ValueError(f"Unknown GPD weights {name!r}; expected one of {sorted(GPD_WEIGHTS)}.")
    spec = GPD_WEIGHTS[name]
    return cached_download("gpd", str(spec["filename"]), tuple(spec["urls"]), str(spec["sha256"]))


def gpd_builder(
    task: str,
    in_channels: int = 3,
    pretrained: Optional[Union[bool, str, Path]] = None,
    **kwargs,
) -> nn.Module:
    """GPD (Ross et al. 2018). ``task``: ``"classification"`` or ``"picking"`` (the same model).

    ``forward`` classifies ``(batch, 3, 400)`` windows; :meth:`GPD.annotate` and
    :meth:`GPD.extract_picks` pick longer traces with the official sliding window. ``pretrained``:
    ``True`` / ``"original"`` for the released weights (``model_pol_best.hdf5``, MIT) or a path to a
    Keras weight file of the same network.
    """
    kwargs.pop("name", None)
    if kwargs:
        raise TypeError(f"gpd_builder got unexpected arguments {sorted(kwargs)}.")
    if task.lower() not in ("classification", "picking"):
        raise ValueError(f"gpd supports task='classification' or 'picking', got {task!r}.")
    model = GPD(in_channels=in_channels)
    if pretrained is not None and pretrained is not False:
        if int(in_channels) != 3:
            raise ValueError(f"The released GPD weights need in_channels=3, got {in_channels}.")
        if pretrained is True or str(pretrained) in GPD_WEIGHTS:
            path = gpd_weights_path("original" if pretrained is True else str(pretrained))
        else:
            path = Path(pretrained)
        load_gpd_keras_weights(model, path)
    return model


__all__ = ["GPD", "GPD_WEIGHTS", "KerasBatchNorm1d", "gpd_builder", "gpd_weights_path", "load_gpd_keras_weights"]
