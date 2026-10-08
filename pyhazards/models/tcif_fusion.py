"""TCIF-fusion: model-knowledge-guided 24-hour tropical cyclone intensity forecasting.

Paper: C. Wang, X. Li and G. Zheng, "Tropical cyclone intensity forecasting using model knowledge
guided deep learning model", Environmental Research Letters 19, 024006 (2024),
doi:10.1088/1748-9326/ad1bde (open access, CC BY 4.0).

The official repository (github.com/wangchong96/TCIF-fusion) holds two Keras 2 / TensorFlow 1.x
notebooks and no LICENSE file, so this module is written from the paper (Section 2.3, Figure 1) and the
layer list of the notebook's ``TCIF_fusion()`` graph; the notebook is used only as a test oracle
(tests/oracle/test_tcif_fusion_oracle.py rebuilds its graph in Keras 3) and is never copied.

TCIF-fusion forecasts the maximum sustained wind (2-minute, m/s) 24 hours ahead from five inputs, all
in the channels-last layout of the official notebook (the paper's "x-y-t-z" arrangement):

* ``u``, ``v``, ``w``: ERA5 wind components ``(batch, 25, 25, 5, 4)`` - 25 x 25 one-degree grid around
  the storm, 5 six-hourly times, 4 pressure levels (200, 500, 850, 1000 hPa) as channels;
* ``sst``: ERA5 sea surface temperature ``(batch, 25, 25, 5, 1)``;
* ``all``: the four ERA5 fields concatenated and reshaped to ``(batch, 25, 25, 65)`` (Figure 1, "ALL";
  built from ``u``, ``v``, ``w``, ``sst`` with :func:`era5_all` when not given);
* ``his``: 30 historical storm features (Figure 1: historical wind speed and pressure) ``(batch, 30)``;
* ``ir``: GridSat-B1 11 um infrared images ``(batch, 224, 224, 5)``.

Network (``ReLU`` everywhere except the linear output): each ERA5 field has a 3-D CNN branch of three
blocks (three 3 x 3 x 5 convolutions with 128 / 256 / 512 filters and a 2 x 2 x 2 max pooling with
stride (2, 2, 1), Keras "same" padding); a fusion branch adds the four branches' outputs after every
block and runs blocks 2 and 3 on the sums; the ALL input runs through three 2-D blocks (three 3 x 3
convolutions, 3 x 3 max pooling with stride 2). The flattened maps of the four field branches, the
fusion branch and the ALL branch (212,992 values) go through Dense(256) and Dense(15); the history
through Dense(32) and Dense(64); the IR images through a VGG-19 (filters 64-128-256-512-512, Dense
1024-1024-15). The 94 concatenated features pass two residual blocks of three dense layers (256, 128)
and a linear output. 299,588,223 parameters with the 65-channel ALL input.

Model knowledge (MK, Section 2.4): a first TCIF-fusion model is trained; Grad-CAM heatmaps of its
inputs are computed; the inputs are multiplied element-wise by the absolute heatmaps and a second
TCIF-fusion is trained on them. :func:`model_knowledge_heatmaps` and :func:`apply_model_knowledge`
implement this step (see their docstrings for the parts the paper leaves open).

The ALL input of the notebook's graph has 85 channels; the paper's Figure 1 shows 25 x 25 x 65, which is
exactly U, V, W (3 x 5 times x 4 levels) and SST (5 times), and the notebook's model-knowledge example
allocates 65 channels. ``all_channels=65`` (default) follows the paper; ``all_channels=85`` rebuilds the
notebook graph (299,611,263 parameters, the number printed in the notebook).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#: Training settings of the notebook (paper Section 2.3): MAE loss, Adam 1e-4, batch 512, 100 epochs,
#: ReduceLROnPlateau(factor 0.5, patience 10, min_lr 1e-6) on the validation loss, best checkpoint kept.
TCIF_TRAINING = {
    "loss": "mae",
    "learning_rate": 1e-4,
    "batch_size": 512,
    "epochs": 100,
    "reduce_lr_on_plateau": {"factor": 0.5, "patience": 10, "min_lr": 1e-6},
}
ERA5_FIELDS = ("u", "v", "w", "sst")
INPUT_NAMES = ("u", "v", "w", "sst", "all", "his", "ir")


def _same_pads(sizes: Sequence[int], kernel: Sequence[int], stride: Sequence[int]) -> List[int]:
    """F.pad arguments (last dimension first) for TensorFlow "same" padding."""
    pads: List[int] = []
    for size, k, s in reversed(list(zip(sizes, kernel, stride))):
        out = -(-int(size) // int(s))
        total = max((out - 1) * int(s) + int(k) - int(size), 0)
        pads += [total // 2, total - total // 2]
    return pads


def _same_max_pool(x: torch.Tensor, kernel: Sequence[int], stride: Sequence[int]) -> torch.Tensor:
    """Max pooling with TensorFlow "same" padding (padded cells never win)."""
    pads = _same_pads(x.shape[2:], kernel, stride)
    if any(pads):
        x = F.pad(x, pads, value=float("-inf"))
    pool = F.max_pool3d if len(kernel) == 3 else F.max_pool2d
    return pool(x, tuple(kernel), tuple(stride))


def _keras_init(module: nn.Module) -> nn.Module:
    """Keras defaults: glorot-uniform kernels, zero biases."""
    nn.init.xavier_uniform_(module.weight)
    nn.init.zeros_(module.bias)
    return module


def _conv3d(c_in: int, c_out: int) -> nn.Conv3d:
    return _keras_init(nn.Conv3d(c_in, c_out, kernel_size=(3, 3, 5), padding=(1, 1, 2)))


def _conv2d(c_in: int, c_out: int) -> nn.Conv2d:
    return _keras_init(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1))


def _dense(n_in: int, n_out: int) -> nn.Linear:
    return _keras_init(nn.Linear(n_in, n_out))


class _Block3d(nn.Module):
    """Three Conv3D(3 x 3 x 5, "same", ReLU) and MaxPooling3D(2 x 2 x 2, stride (2, 2, 1), "same")."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv1 = _conv3d(c_in, c_out)
        self.conv2 = _conv3d(c_out, c_out)
        self.conv3 = _conv3d(c_out, c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x))))))
        return _same_max_pool(x, (2, 2, 2), (2, 2, 1))


class _Block2d(nn.Module):
    """Three Conv2D(3 x 3, "same", ReLU) and MaxPooling2D(3 x 3, stride 2, "same")."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv1 = _conv2d(c_in, c_out)
        self.conv2 = _conv2d(c_out, c_out)
        self.conv3 = _conv2d(c_out, c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x))))))
        return _same_max_pool(x, (3, 3), (2, 2))


_VGG_LAYOUT = (2, 2, 4, 4, 4)  # convolutions per stage


class _VGG19(nn.Module):
    """VGG-19 on the IR images: 16 Conv2D(3 x 3, ReLU) in five stages with 2 x 2 max pooling, then
    Dense(fc) - Dense(fc) - Dense(classes), all with ReLU (the notebook's ``VGG192d``)."""

    def __init__(self, in_channels: int, image_size: int, filters: Sequence[int], fc_dim: int, classes: int):
        super().__init__()
        self.names: List[str] = []
        width = in_channels
        for stage, (count, out) in enumerate(zip(_VGG_LAYOUT, filters), start=1):
            for index in range(1, count + 1):
                name = f"conv{stage}_{index}"
                self.add_module(name, _conv2d(width, out))
                self.names.append(name)
                width = out
        size = image_size
        for _ in _VGG_LAYOUT:
            size = -(-size // 2)
        self.flat_dim = size * size * width
        self.fc1 = _dense(self.flat_dim, fc_dim)
        self.fc2 = _dense(fc_dim, fc_dim)
        self.prediction = _dense(fc_dim, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage, count in enumerate(_VGG_LAYOUT, start=1):
            for index in range(1, count + 1):
                x = F.relu(getattr(self, f"conv{stage}_{index}")(x))
            x = _same_max_pool(x, (2, 2), (2, 2))
        x = x.permute(0, 2, 3, 1).flatten(1)  # Keras Flatten order (channels last)
        return F.relu(self.prediction(F.relu(self.fc2(F.relu(self.fc1(x))))))


def era5_all(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, sst: torch.Tensor) -> torch.Tensor:
    """The ALL input: U, V, W ``(batch, X, Y, T, Z)`` and SST ``(batch, X, Y, T, 1)`` flattened over
    (time, level) - time-major - and concatenated, ``(batch, X, Y, 3 T Z + T)`` (65 channels by default)."""
    parts = [field.reshape(*field.shape[:3], -1) for field in (u, v, w, sst)]
    return torch.cat(parts, dim=-1)


class TCIFFusion(nn.Module):
    """TCIF-fusion intensity regressor; see the module docstring for the inputs.

    ``forward(inputs)`` takes a mapping with ``u``, ``v``, ``w``, ``sst``, ``his``, ``ir`` and optionally
    ``all`` (built from the ERA5 fields when missing; required when ``all_channels`` is not
    ``3 T Z + T``), or the seven tensors positionally in the notebook's input order
    ``(u, v, w, sst, all, his, ir)``. Returns ``(batch, 1)``: the 24-hour intensity in the training
    target's unit (m/s in the paper).
    """

    def __init__(
        self,
        grid_size: int = 25,
        time_steps: int = 5,
        levels: int = 4,
        all_channels: Optional[int] = 65,
        his_dim: int = 30,
        ir_size: int = 224,
        ir_channels: int = 5,
        widths: Sequence[int] = (128, 256, 512),
        vgg_filters: Sequence[int] = (64, 128, 256, 512, 512),
        vgg_fc: int = 1024,
        branch_dim: int = 15,
        fusion_dim: int = 256,
        head_dims: Sequence[int] = (256, 128),
    ):
        super().__init__()
        widths = tuple(int(c) for c in widths)
        if len(widths) != 3 or len(tuple(vgg_filters)) != 5 or len(tuple(head_dims)) != 2:
            raise ValueError("TCIF-fusion has three ERA5 blocks, five VGG stages and two residual blocks")
        self.grid_size, self.time_steps, self.levels = int(grid_size), int(time_steps), int(levels)
        self.derived_all_channels = 3 * self.time_steps * self.levels + self.time_steps
        self.all_channels = self.derived_all_channels if all_channels is None else int(all_channels)
        self.his_dim, self.ir_size, self.ir_channels = int(his_dim), int(ir_size), int(ir_channels)

        # Creation order of the notebook graph: 3-D blocks, ALL blocks, ERA5 dense, history, VGG, head.
        self.u_block1 = _Block3d(self.levels, widths[0])
        self.v_block1 = _Block3d(self.levels, widths[0])
        self.w_block1 = _Block3d(self.levels, widths[0])
        self.sst_block1 = _Block3d(1, widths[0])
        for stage in (2, 3):
            c_in, c_out = widths[stage - 2], widths[stage - 1]
            for name in ("u", "v", "w", "sst", "fusion"):
                self.add_module(f"{name}_block{stage}", _Block3d(c_in, c_out))
        self.all_block1 = _Block2d(self.all_channels, widths[0])
        self.all_block2 = _Block2d(widths[0], widths[1])
        self.all_block3 = _Block2d(widths[1], widths[2])

        side = self.grid_size
        for _ in range(3):
            side = -(-side // 2)
        self.era5_flat_dim = 5 * side * side * self.time_steps * widths[2] + side * side * widths[2]
        self.era5_dense1 = _dense(self.era5_flat_dim, fusion_dim)
        self.era5_dense2 = _dense(fusion_dim, branch_dim)
        self.his_dense1 = _dense(self.his_dim, 32)
        self.his_dense2 = _dense(32, 64)
        self.ir = _VGG19(self.ir_channels, self.ir_size, tuple(vgg_filters), int(vgg_fc), branch_dim)
        head_in = branch_dim + 64 + branch_dim
        self.res1_dense1 = _dense(head_in, head_dims[0])
        self.res1_dense2 = _dense(head_dims[0], head_dims[0])
        self.res1_dense3 = _dense(head_dims[0], head_dims[0])
        self.res2_dense1 = _dense(head_dims[0], head_dims[1])
        self.res2_dense2 = _dense(head_dims[1], head_dims[1])
        self.res2_dense3 = _dense(head_dims[1], head_dims[1])
        self.output = _dense(head_dims[1], 1)

    # -- inputs --------------------------------------------------------------------------------------
    def _gather(self, args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        if len(args) == 1 and isinstance(args[0], Mapping):
            values = dict(args[0])
        else:
            values = dict(zip(INPUT_NAMES, args))
        values.update({k: v for k, v in kwargs.items() if v is not None})
        missing = [name for name in INPUT_NAMES if name != "all" and values.get(name) is None]
        if missing:
            raise ValueError(f"TCIF-fusion needs inputs {list(INPUT_NAMES)}; missing {missing}")
        g, t, z = self.grid_size, self.time_steps, self.levels
        expected = {
            "u": (g, g, t, z),
            "v": (g, g, t, z),
            "w": (g, g, t, z),
            "sst": (g, g, t, 1),
            "all": (g, g, self.all_channels),
            "his": (self.his_dim,),
            "ir": (self.ir_size, self.ir_size, self.ir_channels),
        }
        batch = None
        for name in (*ERA5_FIELDS, "his", "ir", "all"):
            if name == "all" and values.get("all") is None:
                if self.all_channels != self.derived_all_channels:
                    raise ValueError(f"all_channels={self.all_channels} differs from U, V, W, SST ({self.derived_all_channels}); pass inputs['all']")
                values["all"] = era5_all(values["u"], values["v"], values["w"], values["sst"])
            tensor = values[name]
            if tensor.ndim != len(expected[name]) + 1 or tuple(tensor.shape[1:]) != expected[name]:
                raise ValueError(f"TCIF-fusion input {name!r} must be shaped (batch, {', '.join(map(str, expected[name]))}), got {tuple(tensor.shape)}")
            if batch is None:
                batch = tensor.size(0)
            elif tensor.size(0) != batch:
                raise ValueError(f"TCIF-fusion inputs have different batch sizes ({name!r} shape {tuple(tensor.shape)})")
        return values

    # -- network -------------------------------------------------------------------------------------
    @staticmethod
    def _flat3d(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 4, 1).flatten(1)  # Keras Flatten of (X, Y, T, C)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        x = self._gather(args, kwargs)
        # channels-last (B, X, Y, T, Z) -> (B, Z, X, Y, T)
        fields = {name: x[name].permute(0, 4, 1, 2, 3) for name in ERA5_FIELDS}
        u, v, w, t = self.u_block1(fields["u"]), self.v_block1(fields["v"]), self.w_block1(fields["w"]), self.sst_block1(fields["sst"])
        fused = u + v + w + t
        for stage in (2, 3):
            u = getattr(self, f"u_block{stage}")(u)
            v = getattr(self, f"v_block{stage}")(v)
            w = getattr(self, f"w_block{stage}")(w)
            t = getattr(self, f"sst_block{stage}")(t)
            c = getattr(self, f"fusion_block{stage}")(fused)
            fused = u + v + w + t + c
        h = self.all_block3(self.all_block2(self.all_block1(x["all"].permute(0, 3, 1, 2))))
        flat = torch.cat([self._flat3d(u), self._flat3d(v), self._flat3d(w), self._flat3d(t), self._flat3d(fused), h.permute(0, 2, 3, 1).flatten(1)], dim=1)
        era5 = F.relu(self.era5_dense2(F.relu(self.era5_dense1(flat))))
        his = F.relu(self.his_dense2(F.relu(self.his_dense1(x["his"]))))
        ir = self.ir(x["ir"].permute(0, 3, 1, 2))
        joined = torch.cat([era5, his, ir], dim=1)
        first = F.relu(self.res1_dense1(joined))
        joined = F.relu(self.res1_dense3(F.relu(self.res1_dense2(first)))) + first
        first = F.relu(self.res2_dense1(joined))
        joined = F.relu(self.res2_dense3(F.relu(self.res2_dense2(first)))) + first
        return self.output(joined)

    # -- Keras weights -------------------------------------------------------------------------------
    def keras_layer_names(self) -> Dict[str, str]:
        """Keras layer name (the notebook's Keras 2 numbering, as in its printed summary) -> module name."""
        conv3d = [f"{field}_block{stage}.conv{i}" for stage in (1, 2, 3) for field in (("u", "v", "w", "sst") if stage == 1 else ("u", "v", "w", "sst", "fusion")) for i in (1, 2, 3)]
        conv2d = [f"all_block{stage}.conv{i}" for stage in (1, 2, 3) for i in (1, 2, 3)] + [f"ir.{name}" for name in self.ir.names]
        dense = ["era5_dense1", "era5_dense2", "his_dense1", "his_dense2", "ir.fc1", "ir.fc2", "ir.prediction", "res1_dense1", "res1_dense2", "res1_dense3", "res2_dense1", "res2_dense2", "res2_dense3", "output"]
        names = {}
        for prefix, modules in (("conv3d", conv3d), ("conv2d", conv2d), ("dense", dense)):
            for index, module in enumerate(modules, start=1):
                names[f"{prefix}_{index}"] = module
        return names


def load_keras_weights(model: TCIFFusion, weights: Mapping[str, Sequence[np.ndarray]]) -> TCIFFusion:
    """Copy Keras weights ``{layer name: [kernel, bias]}`` (names as in :meth:`TCIFFusion.keras_layer_names`)."""
    names = model.keras_layer_names()
    missing = sorted(set(names) - set(weights))
    if missing:
        raise ValueError(f"missing Keras layers {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    modules = dict(model.named_modules())
    with torch.no_grad():
        for keras_name, module_name in names.items():
            kernel, bias = (np.asarray(a) for a in weights[keras_name])
            layer = modules[module_name]
            kernel_t = torch.as_tensor(kernel, dtype=layer.weight.dtype)
            if kernel_t.ndim == 5:
                kernel_t = kernel_t.permute(4, 3, 0, 1, 2)
            elif kernel_t.ndim == 4:
                kernel_t = kernel_t.permute(3, 2, 0, 1)
            else:
                kernel_t = kernel_t.T
            if kernel_t.shape != layer.weight.shape or tuple(bias.shape) != tuple(layer.bias.shape):
                raise ValueError(f"{keras_name} -> {module_name}: kernel {tuple(kernel.shape)} does not fit weight {tuple(layer.weight.shape)}")
            layer.weight.copy_(kernel_t)
            layer.bias.copy_(torch.as_tensor(bias, dtype=layer.bias.dtype))
    return model


# ---------------------------------------------------------------------------------------------------
# Model knowledge (Section 2.4)

MK_INPUTS = ("u", "v", "w", "sst", "ir")


def model_knowledge_heatmaps(model: nn.Module, inputs: Mapping[str, torch.Tensor], names: Iterable[str] = MK_INPUTS) -> Dict[str, torch.Tensor]:
    """Grad-CAM heatmaps of the inputs of a trained TCIF-fusion model (the "model knowledge").

    For every sample and input ``A`` (channels last), the gradient of the forecast with respect to
    ``A`` is averaged over the two horizontal axes to give one weight per remaining position and
    channel, and the heatmap is the channel mean of ``A`` times these weights: ``(batch, X, Y, T)`` for
    the ERA5 fields, ``(batch, H, W)`` for IR. This follows the Grad-CAM recipe the official notebook
    starts (gradient of the output w.r.t. an input, pooled over the batch and the horizontal axes); the
    notebook stops before forming the heatmap, and the paper gives no normalisation, so none is applied.
    """
    names = tuple(names)
    leaves = {key: value.detach().clone().requires_grad_(key in names) for key, value in inputs.items()}
    was_training = model.training
    model.eval()
    try:
        output = model(leaves)
        grads = torch.autograd.grad(output.sum(), [leaves[name] for name in names])
    finally:
        model.train(was_training)
    heatmaps = {}
    for name, grad in zip(names, grads):
        weights = grad.mean(dim=(1, 2), keepdim=True)
        heatmaps[name] = (leaves[name].detach() * weights).mean(dim=-1)
    return heatmaps


def apply_model_knowledge(inputs: Mapping[str, torch.Tensor], heatmaps: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """MK-guided inputs: every input with a heatmap multiplied element-wise by the heatmap's absolute value.

    ALL is rebuilt from the guided ERA5 fields (it is their concatenation, Figure 1) when the inputs
    carry the 65-channel layout; HIS is unchanged.
    """
    guided = dict(inputs)
    for name, heatmap in heatmaps.items():
        guided[name] = inputs[name] * heatmap.abs().unsqueeze(-1)
    if any(name in heatmaps for name in ERA5_FIELDS) and all(name in guided for name in ERA5_FIELDS):
        derived = era5_all(*(guided[name] for name in ERA5_FIELDS))
        if inputs.get("all") is None or inputs["all"].shape == derived.shape:
            guided["all"] = derived
    return guided


def tcif_fusion_builder(
    task: str,
    all_channels: Optional[int] = 65,
    grid_size: int = 25,
    time_steps: int = 5,
    levels: int = 4,
    his_dim: int = 30,
    ir_size: int = 224,
    ir_channels: int = 5,
    widths: Sequence[int] = (128, 256, 512),
    vgg_filters: Sequence[int] = (64, 128, 256, 512, 512),
    vgg_fc: int = 1024,
    **kwargs: Any,
) -> nn.Module:
    """TCIF-fusion; the defaults are the paper's model (``all_channels=85`` gives the notebook graph).

    The size arguments only exist to build small copies for smoke tests; checked are the defaults.
    """
    _ = kwargs
    if task.lower() != "regression":
        raise ValueError("tcif_fusion is a regression model (24-hour intensity); use task='regression'.")
    return TCIFFusion(
        grid_size=grid_size,
        time_steps=time_steps,
        levels=levels,
        all_channels=all_channels,
        his_dim=his_dim,
        ir_size=ir_size,
        ir_channels=ir_channels,
        widths=widths,
        vgg_filters=vgg_filters,
        vgg_fc=vgg_fc,
    )


__all__ = [
    "INPUT_NAMES",
    "MK_INPUTS",
    "TCIF_TRAINING",
    "TCIFFusion",
    "apply_model_knowledge",
    "era5_all",
    "load_keras_weights",
    "model_knowledge_heatmaps",
    "tcif_fusion_builder",
]
