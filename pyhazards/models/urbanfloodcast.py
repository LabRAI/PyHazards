"""UrbanFloodCast deep neural operator (DNO; Xu et al., Journal of Hydrology 661:133705, 2025).

The DNO is a U-shaped Fourier neural operator in space and time (U-NO): a lifting MLP, three operator
blocks that change the space-time grid (to 3/4 of the spatial size, then twice the time steps, then back
to the input grid) with concatenated skip connections, and a projection MLP. It maps the current water
depth and unit discharges, the rainfall of every future step and the terrain of an urban area to depth
and discharges at all ``T`` (24) future 5-minute steps at once ("one-shot").

The official repository (https://github.com/HydroPML/UrbanFloodCast) has no licence, so this module is
written from the paper and from permissively licensed building blocks:

- the operator blocks follow U-NO (Rahman, Ross and Azizzadenesheli, arXiv:2204.11127), ``integral_operators.py``
  of https://github.com/ashiq24/UNO @ 19462d82729ef64ef7b9e97056ddcaaf3044ad47 (BSD-2-Clause,
  Copyright (c) 2022, Md Ashiqur Rahman): ``SpectralConv3d_Uno``, ``pointwise_op_3D``, ``OperatorBlock_3D``;
- the extra point-wise MLP branch of every block is the two-layer 1x1x1 convolution MLP of the Fourier
  neural operator (https://github.com/neuraloperator/neuraloperator, MIT, Copyright (c) 2023 NeuralOperator
  developers).

As in the DNO, the point-wise branch of a block is a 1x1x1 convolution followed by trilinear
resampling (U-NO's current code also low-pass filters it), and each block adds the MLP branch.
Parameter names follow the official model so its state dicts load with ``strict=True``; the oracle test
(``tests/oracle/test_urbanfloodcast_oracle.py``) runs the official code (fetched at test time, never
vendored) and checks parameters, seeded initialisation and outputs.
"""

from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_MIN_SPATIAL = 27  # int(3 * D / 4) must reach the 20 spatial modes of conv0 / conv8.
_MIN_TIME = 14  # rfft over time must hold the 8 temporal modes.


class SpectralConv3d_Uno(nn.Module):
    """3-D Fourier layer with a different output grid (U-NO).

    FFT over (x, y, t), multiply the lowest ``modes1 x modes2 x modes3`` frequencies (four corners of the
    two signed spatial axes) by complex weights, and inverse-FFT onto the ``(dim1, dim2, dim3)`` grid;
    ``norm="forward"`` makes the operator independent of the grid size.
    """

    def __init__(self, in_codim, out_codim, dim1, dim2, dim3, modes1=None, modes2=None, modes3=None):
        super().__init__()
        in_codim = int(in_codim)
        out_codim = int(out_codim)
        self.in_channels = in_codim
        self.out_channels = out_codim
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        if modes1 is not None:
            self.modes1 = modes1
            self.modes2 = modes2
            self.modes3 = modes3
        else:
            self.modes1 = dim1
            self.modes2 = dim2
            self.modes3 = dim3 // 2 + 1
        self.scale = (1 / (2 * in_codim)) ** (1.0 / 2.0)
        shape = (in_codim, out_codim, self.modes1, self.modes2, self.modes3)
        self.weights1 = nn.Parameter(self.scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.randn(*shape, dtype=torch.cfloat))

    @staticmethod
    def compl_mul3d(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixyz,ioxyz->boxyz", inputs, weights)

    def forward(self, x: torch.Tensor, dim1=None, dim2=None, dim3=None) -> torch.Tensor:
        if dim1 is not None:
            # As in U-NO, the last requested grid becomes the default of later calls.
            self.dim1, self.dim2, self.dim3 = dim1, dim2, dim3
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1], norm="forward")
        out_ft = torch.zeros(
            batchsize, self.out_channels, self.dim1, self.dim2, self.dim3 // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        m1, m2, m3 = self.modes1, self.modes2, self.modes3
        out_ft[:, :, :m1, :m2, :m3] = self.compl_mul3d(x_ft[:, :, :m1, :m2, :m3], self.weights1)
        out_ft[:, :, -m1:, :m2, :m3] = self.compl_mul3d(x_ft[:, :, -m1:, :m2, :m3], self.weights2)
        out_ft[:, :, :m1, -m2:, :m3] = self.compl_mul3d(x_ft[:, :, :m1, -m2:, :m3], self.weights3)
        out_ft[:, :, -m1:, -m2:, :m3] = self.compl_mul3d(x_ft[:, :, -m1:, -m2:, :m3], self.weights4)
        return torch.fft.irfftn(out_ft, s=(self.dim1, self.dim2, self.dim3), norm="forward")


class MLP3d(nn.Module):
    """Point-wise two-layer MLP (1x1x1 convolutions, GELU) resampled trilinearly to the output grid."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int):
        super().__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, dim1: int, dim2: int, dim3: int) -> torch.Tensor:
        x = self.mlp2(F.gelu(self.mlp1(x)))
        return F.interpolate(x, size=(dim1, dim2, dim3), mode="trilinear", align_corners=True)


class pointwise_op_3D(nn.Module):  # noqa: N801 (reference class name)
    """Point-wise linear map (1x1x1 convolution) resampled trilinearly to the output grid."""

    def __init__(self, in_codim, out_codim, dim1, dim2, dim3):
        super().__init__()
        self.conv = nn.Conv3d(int(in_codim), int(out_codim), 1)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)
        self.dim3 = int(dim3)

    def forward(self, x: torch.Tensor, dim1=None, dim2=None, dim3=None) -> torch.Tensor:
        if dim1 is None:
            dim1, dim2, dim3 = self.dim1, self.dim2, self.dim3
        return F.interpolate(self.conv(x), size=(dim1, dim2, dim3), mode="trilinear", align_corners=True)


class OperatorBlock_3D(nn.Module):  # noqa: N801 (reference class name)
    """``GELU(Norm(K(v) + MLP(v) + W(v)))`` on a new grid: spectral, MLP and point-wise branches."""

    def __init__(self, in_codim, out_codim, dim1, dim2, dim3, modes1, modes2, modes3, Normalize=False, Non_Lin=True):  # noqa: N803
        super().__init__()
        self.conv = SpectralConv3d_Uno(in_codim, out_codim, dim1, dim2, dim3, modes1, modes2, modes3)
        self.mlp = MLP3d(in_codim, out_codim, 2 * out_codim)
        self.w = pointwise_op_3D(in_codim, out_codim, dim1, dim2, dim3)
        self.normalize = Normalize
        self.non_lin = Non_Lin
        if Normalize:
            self.normalize_layer = nn.InstanceNorm3d(int(out_codim), affine=True)

    def forward(self, x: torch.Tensor, dim1=None, dim2=None, dim3=None) -> torch.Tensor:
        x_out = self.conv(x, dim1, dim2, dim3) + self.mlp(x, dim1, dim2, dim3) + self.w(x, dim1, dim2, dim3)
        if self.normalize:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out


class UrbanFloodCast(nn.Module):
    """The UrbanFloodCast deep neural operator (DNO-3, the official default).

    Input ``(batch, Sy, Sx, T, initial_step, num_channels)`` (or with the last two axes merged): for each
    cell and each of the ``T`` output steps, the ``num_channels=5`` inputs of the UrbanFloodCast data
    pipeline - current water depth, x and y unit discharge (repeated over the output steps), the
    log-transformed rainfall of that step and the normalised terrain (see
    :func:`pyhazards.datasets.flood.urbanfloodcast.prepare_urbanfloodcast_event`). A (y, x, t) grid in
    [0, 1] is appended, the 8 channels are lifted by ``fc`` (8 -> 16) and ``fc0`` (16 -> ``width``),
    passed through ``conv0`` (to 3/4 of the spatial grid), ``conv7`` (twice the time steps; concatenated with
    resampled ``conv0`` output) and ``conv8`` (back to the input grid; concatenated with the lifted input),
    and projected by ``fc1`` / ``fc2`` to 3 outputs. Output ``(batch, Sy, Sx, T, 3)``: water depth and the
    two unit discharges at every step.

    Defaults are the official configuration (``DNO(num_channels=5, width=10, initial_step=1, pad=False,
    factor=1)``, 4,470,437 parameters, complex weights counted once). The fixed Fourier modes need
    ``Sy, Sx >= 27`` and ``T >= 14``.
    """

    def __init__(
        self,
        num_channels: int = 5,
        width: int = 10,
        initial_step: int = 1,
        pad: int = 0,
        factor: int = 1,
        pad_both: bool = False,
    ):
        super().__init__()
        if pad:
            raise ValueError(
                "UrbanFloodCast supports pad=0 only: the official time padding (--time_pad) changes the output "
                "length and fails in its own training script."
            )
        self.num_channels = int(num_channels)
        self.initial_step = int(initial_step)
        self.in_width = self.initial_step * self.num_channels + 3
        self.width = int(width)
        self.padding = 0
        self.pad_both = bool(pad_both)
        width = self.width
        self.fc = nn.Linear(self.in_width, self.in_width * 2)
        self.fc0 = nn.Linear(self.in_width * 2, width)
        self.conv0 = OperatorBlock_3D(width, 1 * factor * width, 48, 48, 10, 20, 20, 8, Normalize=True)
        self.conv7 = OperatorBlock_3D(1 * factor * width, 1 * factor * width, 48, 48, 20, 14, 14, 8, Normalize=True)
        self.conv8 = OperatorBlock_3D(2 * factor * width, width, 64, 64, 20, 20, 20, 8)
        self.fc1 = nn.Linear(2 * width, 4 * width)
        self.fc2 = nn.Linear(4 * width, 3)

    @staticmethod
    def get_grid(shape: Tuple[int, ...], device: Any = None) -> torch.Tensor:
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.linspace(0, 1, size_x).reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.linspace(0, 1, size_y).reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.linspace(0, 1, size_z).reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        grid = torch.cat((gridx, gridy, gridz), dim=-1)
        return grid if device is None else grid.to(device)

    def _check_input(self, x: torch.Tensor) -> None:
        expected = self.initial_step * self.num_channels
        if x.ndim == 6:
            features = x.shape[4] * x.shape[5]
        elif x.ndim == 5:
            features = x.shape[4]
        else:
            raise ValueError(
                "UrbanFloodCast expects inputs of shape (batch, Sy, Sx, T, initial_step, num_channels) or "
                f"(batch, Sy, Sx, T, initial_step * num_channels), got ndim={x.ndim} {tuple(x.shape)}."
            )
        if features != expected:
            raise ValueError(
                f"UrbanFloodCast expects {self.initial_step} x {self.num_channels} input features per cell and step, "
                f"got shape {tuple(x.shape)}."
            )
        if min(x.shape[1], x.shape[2]) < _MIN_SPATIAL or x.shape[3] < _MIN_TIME:
            raise ValueError(
                f"UrbanFloodCast needs Sy, Sx >= {_MIN_SPATIAL} and T >= {_MIN_TIME} for its Fourier modes; got shape {tuple(x.shape)}."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        x = x.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3], -1)
        x = torch.cat((x, self.get_grid(x.shape, x.device)), dim=-1)
        x_fc0 = F.gelu(self.fc0(F.gelu(self.fc(x.float()))))
        x_fc0 = x_fc0.permute(0, 4, 1, 2, 3)
        d1, d2, d3 = x_fc0.shape[-3], x_fc0.shape[-2], x_fc0.shape[-1]
        x_c0 = self.conv0(x_fc0, int(3 * d1 / 4), int(3 * d2 / 4), d3)
        x_c7 = self.conv7(x_c0, int(3 * d1 / 4), int(3 * d2 / 4), int(2.0 * d3))
        x_c7 = torch.cat(
            [x_c7, F.interpolate(x_c0, size=tuple(x_c7.shape[2:]), mode="trilinear", align_corners=True)], dim=1
        )
        x_c8 = self.conv8(x_c7, d1, d2, d3)
        x_c8 = torch.cat(
            [x_c8, F.interpolate(x_fc0, size=tuple(x_c8.shape[2:]), mode="trilinear", align_corners=True)], dim=1
        )
        x_c8 = x_c8.permute(0, 2, 3, 4, 1)
        return self.fc2(F.gelu(self.fc1(x_c8)))


DNO = UrbanFloodCast


def urbanfloodcast_builder(
    task: str,
    num_channels: int = 5,
    width: int = 10,
    initial_step: int = 1,
    pad: int = 0,
    factor: int = 1,
    pad_both: bool = False,
    **kwargs: Any,
) -> UrbanFloodCast:
    """UrbanFloodCast DNO at the official configuration by default."""
    if task.lower() != "regression":
        raise ValueError(f"UrbanFloodCast only supports task='regression', got {task!r}.")
    kwargs.pop("name", None)
    if kwargs:
        raise ValueError(f"Unknown UrbanFloodCast arguments: {sorted(kwargs)}.")
    return UrbanFloodCast(
        num_channels=num_channels,
        width=width,
        initial_step=initial_step,
        pad=pad,
        factor=factor,
        pad_both=pad_both,
    )


__all__ = [
    "DNO",
    "MLP3d",
    "OperatorBlock_3D",
    "SpectralConv3d_Uno",
    "UrbanFloodCast",
    "pointwise_op_3D",
    "urbanfloodcast_builder",
]
