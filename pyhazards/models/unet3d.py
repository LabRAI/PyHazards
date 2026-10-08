"""MONAI's ``UNet``, with TS-SatFire's spatio-temporal (U-Net-3D) configuration.

Port of ``monai.networks.nets.UNet`` from MONAI 1.3.2 (``monai/networks/nets/unet.py``,
``monai/networks/blocks/convolutions.py`` ``ResidualUnit`` and
``monai/networks/layers/simplelayers.py`` ``SkipConnection``; Apache License 2.0, Copyright (c)
MONAI Consortium). The convolution blocks are shared with :mod:`pyhazards.models.attention_unet`.
Changes made for PyHazards: plain PyTorch (no MONAI dependency), invalid configurations and input
shapes raise ``ValueError`` up front, activations and norms are given by name (``"prelu"``,
``"relu"``; ``"instance"``, ``"batch"``), a ``stride_mode="shared"`` option reproduces the
TS-SatFire modification below, and :class:`TemporalUNet` adds the TS-SatFire input layout and
temporal read-out. Module names, creation order and defaults follow MONAI, so MONAI state dicts
load with ``strict=True`` and the same seed gives the same initial weights.

What MONAI's ``UNet`` is: MONAI documents it as the "enhanced" residual U-Net of Kerfoot et al.,
"Left-Ventricle Quantification Using Residual U-Net" (STACOM 2018, LNCS 11395,
doi:10.1007/978-3-030-12029-0_40). Each level has one encoder unit that down-samples with a
strided convolution (no pooling) and one decoder unit that up-samples with a strided transposed
convolution applied to the concatenation of the skip feature and the up-sampled deeper feature;
with ``num_res_units > 0`` these units are residual units. With ``num_res_units=0`` -- the
setting TS-SatFire uses -- every unit is a single ``Conv -> InstanceNorm -> Dropout -> PReLU``,
so the network is a plain strided-convolution encoder-decoder with concatenation skips, one
convolution per level and side, and no residual connections. It is neither Ronneberger et al.'s
2D U-Net (two unpadded 3x3 convolutions per level, max pooling) nor Cicek et al.'s 3D U-Net
(two 3x3x3 convolutions with batch norm per level, max pooling).

TS-SatFire (Zhao, Gerard & Ban, Scientific Data 12:1817, 2025) uses it in two ways: the 2D
"U-Net" baseline is stock MONAI ``UNet(spatial_dims=2, channels=(64, 128, 256, 512, 1024),
strides=(2, 2, 2, 2))``; "U-Net-3D" uses the repository's copy ``spatial_models/unet.py``, whose
only change is that the single ``strides`` argument -- ``(1, 2, 2)`` over (time, height, width)
-- is applied at every level instead of one entry per level, so time is never down-sampled. The
prediction script feeds ``(batch, channels, time, H, W)`` and averages the logits over time.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .attention_unet import _CONV, Convolution, _as_tuple, _same_padding
from .monai_blocks import TemporalReadout

IntOrSeq = Union[int, Sequence[int]]

# TS-SatFire configuration (run_spatial_temp_model_pred.py, run_spatial_model.py).
TS_SATFIRE_CHANNELS = (64, 128, 256, 512, 1024)
TS_SATFIRE_3D_STRIDES = (1, 2, 2)


class SkipConnection(nn.Module):
    """``cat([x, submodule(x)], dim=1)`` (``monai.networks.layers.SkipConnection``, mode ``"cat"``)."""

    def __init__(self, submodule: nn.Module, dim: int = 1):
        super().__init__()
        self.submodule = submodule
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.submodule(x)], dim=self.dim)


class ResidualUnit(nn.Module):
    """``subunits`` convolution units plus an additive residual (``monai.networks.blocks.ResidualUnit``).

    The residual path is the identity, or a convolution when the unit changes the channel count
    (1x1, no padding) or strides (same kernel and padding as the units).
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: IntOrSeq = 1,
        kernel_size: IntOrSeq = 3,
        subunits: int = 2,
        adn_ordering: str = "NDA",
        act: Optional[str] = "prelu",
        norm: Optional[str] = "instance",
        dropout: Optional[float] = None,
        bias: bool = True,
        last_conv_only: bool = False,
        padding: Optional[IntOrSeq] = None,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential()
        self.residual: nn.Module = nn.Identity()
        if not padding:
            padding = _same_padding(kernel_size)
        schannels = in_channels
        sstrides: IntOrSeq = strides
        subunits = max(1, subunits)
        for su in range(subunits):
            conv_only = last_conv_only and su == (subunits - 1)
            unit = Convolution(
                spatial_dims,
                schannels,
                out_channels,
                strides=sstrides,
                kernel_size=kernel_size,
                adn_ordering=adn_ordering,
                act=act,
                norm=norm,
                dropout=dropout,
                bias=bias,
                conv_only=conv_only,
                padding=padding,
            )
            self.conv.add_module(f"unit{su:d}", unit)
            schannels = out_channels
            sstrides = 1
        if np.prod(strides) != 1 or in_channels != out_channels:
            rkernel_size, rpadding = kernel_size, padding
            if np.prod(strides) == 1:  # only the channel count changes: 1x1 kernel without padding
                rkernel_size, rpadding = 1, 0
            self.residual = _CONV[spatial_dims](in_channels, out_channels, rkernel_size, strides, rpadding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        return self.conv(x) + res


class UNet(nn.Module):
    """MONAI ``UNet`` (1.3.2) in plain PyTorch.

    Args:
        spatial_dims: 1, 2 or 3.
        in_channels: input channels.
        out_channels: output channels (logits, no activation).
        channels: feature widths, top level first; at least two entries.
        strides: with ``stride_mode="per_level"`` (MONAI), one stride per level
            (``len(channels) - 1`` entries are used), each an int or one value per spatial
            dimension. With ``stride_mode="shared"`` (TS-SatFire's copy), a single stride -- an int
            or one value per spatial dimension -- used at every level.
        kernel_size: convolution kernel size (odd).
        up_kernel_size: transposed-convolution kernel size (odd).
        num_res_units: residual units per encoder unit (0: plain convolutions, MONAI's default).
        act: ``"prelu"`` (MONAI's default), ``"relu"`` or ``None``.
        norm: ``"instance"`` (MONAI's default), ``"batch"`` or ``None``.
        dropout: dropout rate in every unit.
        bias: convolution bias.
        adn_ordering: order of normalisation (N), dropout (D) and activation (A).
        stride_mode: ``"per_level"`` or ``"shared"``.

    Input ``(batch, in_channels, *spatial)``; output ``(batch, out_channels, *spatial)``. Every
    spatial size must be divisible by the product of that dimension's strides.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Union[IntOrSeq, Sequence[IntOrSeq]],
        kernel_size: IntOrSeq = 3,
        up_kernel_size: IntOrSeq = 3,
        num_res_units: int = 0,
        act: Optional[str] = "prelu",
        norm: Optional[str] = "instance",
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        stride_mode: str = "per_level",
    ):
        super().__init__()
        if spatial_dims not in (1, 2, 3):
            raise ValueError(f"spatial_dims must be 1, 2 or 3, got {spatial_dims}")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        channels = tuple(int(c) for c in channels)
        if len(channels) < 2:
            raise ValueError(f"channels needs at least two entries, got {channels!r}")
        n_levels = len(channels) - 1
        if stride_mode == "shared":
            level_strides = [strides] * n_levels  # TS-SatFire: the same stride at every level
        elif stride_mode == "per_level":
            if isinstance(strides, int):
                raise ValueError("stride_mode='per_level' needs one stride per level; use stride_mode='shared'.")
            level_strides = list(strides)
            if len(level_strides) < n_levels:
                raise ValueError(
                    f"strides needs {n_levels} entries (len(channels) - 1) with stride_mode='per_level', "
                    f"got {len(level_strides)}."
                )
            level_strides = level_strides[:n_levels]  # MONAI ignores (and warns about) extra entries
        else:
            raise ValueError(f"stride_mode must be 'per_level' or 'shared', got {stride_mode!r}")
        self.level_strides = [_as_tuple(s, spatial_dims, "each stride") for s in level_strides]
        for name, kernel in (("kernel_size", kernel_size), ("up_kernel_size", up_kernel_size)):
            _as_tuple(kernel, spatial_dims, name)
            _same_padding(kernel)
        if num_res_units < 0:
            raise ValueError(f"num_res_units must be >= 0, got {num_res_units}")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.stride_mode = stride_mode
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        def _create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[IntOrSeq], is_top: bool
        ) -> nn.Module:
            # Built from the bottom up, as in MONAI, so a seed gives the same initial weights.
            c = channels[0]
            s = strides[0]
            if len(channels) > 2:
                subblock = _create_block(c, c, channels[1:], strides[1:], False)
                upc = c * 2
            else:
                subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]
            down = self._get_down_layer(inc, c, s, is_top)
            up = self._get_up_layer(upc, outc, s, is_top)
            return nn.Sequential(down, SkipConnection(subblock), up)

        self.model = _create_block(in_channels, out_channels, self.channels, level_strides, True)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: IntOrSeq, is_top: bool) -> nn.Module:
        _ = is_top
        if self.num_res_units > 0:
            return ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
        return Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: IntOrSeq, is_top: bool) -> nn.Module:
        conv: nn.Module = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )
        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)
        return conv

    def size_multiple(self) -> Tuple[int, ...]:
        """Per-dimension factor that every input spatial size must be a multiple of."""
        factors = [1] * self.dimensions
        for stride in self.level_strides:
            factors = [f * s for f, s in zip(factors, stride)]
        return tuple(factors)

    def _check_input(self, x: torch.Tensor, layout: str) -> None:
        if x.ndim != self.dimensions + 2:
            raise ValueError(f"{type(self).__name__} expects input shape {layout}, got {tuple(x.shape)}.")
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"{type(self).__name__} expected {self.in_channels} input channels, got shape {tuple(x.shape)}."
            )
        multiple = self.size_multiple()
        if any(size % m for size, m in zip(x.shape[2:], multiple)):
            raise ValueError(
                f"Spatial shape {tuple(x.shape[2:])} must be divisible by {multiple} "
                "(the product of the strides in each dimension)."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = ", ".join(["D", "H", "W"][-self.dimensions :])
        self._check_input(x, f"(batch, channels, {spatial})")
        return self.model(x)


class TemporalUNet(TemporalReadout, UNet):
    """3D :class:`UNet` over ``(time, H, W)`` for raster time series (TS-SatFire's U-Net-3D).

    Takes ``(batch, time, channels, H, W)``, runs the network on ``(batch, channels, time, H, W)``
    and, with ``time_reduction="mean"``, averages the logits over time to
    ``(batch, out_channels, H, W)``. Parameter names are those of :class:`UNet`.
    """

    def __init__(self, *args, time_reduction: str = "mean", **kwargs):
        super().__init__(*args, **kwargs)
        if self.dimensions != 3:
            raise ValueError(f"TemporalUNet needs spatial_dims=3, got {self.dimensions}")
        self._set_time_reduction(time_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_network_layout(x, type(self).__name__)
        self._check_input(x, "(batch, time, channels, height, width)")
        return self._read_out(self.model(x))


def unet3d_builder(
    task: str,
    in_channels: int,
    out_channels: int = 2,
    spatial_dims: int = 3,
    channels: Sequence[int] = TS_SATFIRE_CHANNELS,
    strides: Optional[Union[IntOrSeq, Sequence[IntOrSeq]]] = None,
    stride_mode: str = "per_level",
    kernel_size: IntOrSeq = 3,
    up_kernel_size: IntOrSeq = 3,
    num_res_units: int = 0,
    act: Optional[str] = "prelu",
    norm: Optional[str] = "instance",
    dropout: float = 0.0,
    bias: bool = True,
    adn_ordering: str = "NDA",
    time_reduction: str = "mean",
    **kwargs,
) -> nn.Module:
    """Build MONAI's U-Net; the defaults are TS-SatFire's U-Net-3D prediction model.

    ``spatial_dims=3`` returns a :class:`TemporalUNet` (input ``(B, T, C, H, W)``);
    ``spatial_dims=2`` returns a :class:`UNet` (input ``(B, C, H, W)``). Without ``strides``,
    TS-SatFire's strides are used: ``(1, 2, 2)`` at every level in 3D, ``2`` at every level in 2D.
    """
    _ = kwargs
    if task.lower() != "segmentation":
        raise ValueError(f"unet3d supports task='segmentation', got {task!r}.")
    if strides is None:
        strides = TS_SATFIRE_3D_STRIDES if spatial_dims == 3 else 2
        stride_mode = "shared"
    common = dict(
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        kernel_size=kernel_size,
        up_kernel_size=up_kernel_size,
        num_res_units=num_res_units,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        adn_ordering=adn_ordering,
        stride_mode=stride_mode,
    )
    if spatial_dims == 3:
        return TemporalUNet(spatial_dims=3, time_reduction=time_reduction, **common)
    if spatial_dims == 2:
        return UNet(spatial_dims=2, **common)
    raise ValueError(f"unet3d supports spatial_dims 2 or 3, got {spatial_dims}.")


__all__ = ["ResidualUnit", "SkipConnection", "TemporalUNet", "UNet", "unet3d_builder"]
