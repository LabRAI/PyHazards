"""Attention U-Net as implemented by MONAI, with the TS-SatFire spatio-temporal configuration.

Port of ``monai.networks.nets.AttentionUnet`` from MONAI 1.3.2
(``monai/networks/nets/attentionunet.py``, Apache License 2.0, Copyright (c) MONAI Consortium),
together with the subset of ``monai.networks.blocks.Convolution`` / ``ADN`` and the layer
factories it relies on. Changes made for PyHazards: the MONAI blocks are re-implemented in plain
PyTorch (no MONAI dependency), invalid configurations raise ``ValueError`` up front, a
``stride_mode="shared"`` option reproduces the TS-SatFire modification described below, and
:class:`TemporalAttentionUnet` adds the TS-SatFire input layout and temporal read-out. Module
names, creation order and defaults follow MONAI, so MONAI state dicts load with ``strict=True``
and the same seed gives the same initial weights.

MONAI's network is "based on" Oktay et al., "Attention U-Net: Learning Where to Look for the
Pancreas" (MIDL 2018, arXiv:1804.03999) but is not the authors' implementation
(ozan-oktay/Attention-Gated-Networks): its additive gate is computed at the skip resolution from
the already up-sampled decoder feature, without the sub-sampled grid gating, the multiple gates
per level or the deep supervision of the official code.

TS-SatFire (Zhao, Gerard & Ban, Scientific Data 12:1817, 2025) uses this network in two ways:
the 2D baseline is stock MONAI ``AttentionUnet(spatial_dims=2, channels=(64, 128, 256, 512, 1024),
strides=(2, 2, 2, 2))``; the 3D baseline ("Attention-U-Net-3D") uses the repository's copy
``spatial_models/attentionunet.py`` (MONAI 1.2/1.3.0 code), whose only change is that the single
``strides`` argument -- ``(1, 2, 2)`` over (time, height, width) -- is applied at every level
instead of one entry per level, so time is never down-sampled. The prediction script feeds
``(batch, channels, time, H, W)`` and averages the logits over time.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

IntOrSeq = Union[int, Sequence[int]]

_CONV = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
_CONV_TRANSPOSE = {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}
_BATCH_NORM = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}
_INSTANCE_NORM = {1: nn.InstanceNorm1d, 2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d}
_ACTIVATIONS = {"relu": nn.ReLU, "prelu": nn.PReLU}

# TS-SatFire configuration (run_spatial_temp_model_pred.py).
TS_SATFIRE_CHANNELS = (64, 128, 256, 512, 1024)
TS_SATFIRE_3D_STRIDES = (1, 2, 2)


def _as_tuple(value: IntOrSeq, spatial_dims: int, name: str) -> Tuple[int, ...]:
    if isinstance(value, int):
        return (int(value),) * spatial_dims
    values = tuple(int(v) for v in value)
    if len(values) != spatial_dims:
        raise ValueError(f"{name} must be an int or have {spatial_dims} entries, got {value!r}")
    return values


def _same_padding(kernel_size: IntOrSeq) -> IntOrSeq:
    """``monai.networks.layers.convutils.same_padding`` for dilation 1."""
    kernels = (kernel_size,) if isinstance(kernel_size, int) else tuple(kernel_size)
    if any(int(k) % 2 == 0 for k in kernels):
        raise ValueError(f"Same padding needs odd kernel sizes, got {kernel_size!r}")
    padding = tuple((int(k) - 1) // 2 for k in kernels)
    return padding if len(padding) > 1 else padding[0]


def _stride_minus_kernel_padding(kernel_size: IntOrSeq, stride: IntOrSeq) -> IntOrSeq:
    """``monai.networks.layers.convutils.stride_minus_kernel_padding``."""
    kernels = (kernel_size,) if isinstance(kernel_size, int) else tuple(kernel_size)
    strides = (stride,) if isinstance(stride, int) else tuple(stride)
    if len(kernels) == 1 and len(strides) > 1:
        kernels = kernels * len(strides)
    if len(strides) == 1 and len(kernels) > 1:
        strides = strides * len(kernels)
    padding = tuple(int(s) - int(k) for s, k in zip(strides, kernels))
    return padding if len(padding) > 1 else padding[0]


class ADN(nn.Sequential):
    """Normalisation, dropout and activation in a given order (``monai.networks.blocks.ADN``).

    Only the layers used by :class:`AttentionUnet` are supported: batch or instance norm,
    ``nn.Dropout`` (MONAI's default ``dropout_dim=1``) and ReLU or PReLU with default arguments.
    """

    def __init__(
        self,
        ordering: str,
        in_channels: int,
        spatial_dims: int,
        act: Optional[str] = "relu",
        norm: Optional[str] = None,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        layers = {"A": None, "D": None, "N": None}
        if norm is not None:
            if norm == "batch":
                layers["N"] = _BATCH_NORM[spatial_dims](in_channels)
            elif norm == "instance":
                layers["N"] = _INSTANCE_NORM[spatial_dims](in_channels)
            else:
                raise ValueError(f"norm must be 'batch', 'instance' or None, got {norm!r}")
        if act is not None:
            if act not in _ACTIVATIONS:
                raise ValueError(f"act must be one of {sorted(_ACTIVATIONS)} or None, got {act!r}")
            layers["A"] = _ACTIVATIONS[act]()
        if dropout is not None:
            layers["D"] = nn.Dropout(p=float(dropout))
        for item in ordering.upper():
            if item not in layers:
                raise ValueError(f"ordering must only contain 'A', 'D' and 'N', got {ordering!r}")
            if layers[item] is not None:
                self.add_module(item, layers[item])


class Convolution(nn.Sequential):
    """``(Conv | ConvTranspose) -> ADN`` block (``monai.networks.blocks.Convolution``).

    Defaults match MONAI: kernel 3, "same" padding, PReLU activation, instance norm, no dropout.
    Transposed convolutions use ``output_padding = stride - 1``, so they scale sizes exactly by
    the stride.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: IntOrSeq = 1,
        kernel_size: IntOrSeq = 3,
        adn_ordering: str = "NDA",
        act: Optional[str] = "prelu",
        norm: Optional[str] = "instance",
        dropout: Optional[float] = None,
        bias: bool = True,
        conv_only: bool = False,
        is_transposed: bool = False,
        padding: Optional[IntOrSeq] = None,
        output_padding: Optional[IntOrSeq] = None,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_transposed = is_transposed
        if padding is None:
            padding = _same_padding(kernel_size)
        if is_transposed:
            if output_padding is None:
                output_padding = _stride_minus_kernel_padding(1, strides)
            conv: nn.Module = _CONV_TRANSPOSE[spatial_dims](
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                output_padding=output_padding,
                bias=bias,
            )
        else:
            conv = _CONV[spatial_dims](
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                bias=bias,
            )
        self.add_module("conv", conv)
        if conv_only:
            return
        if act is None and norm is None and dropout is None:
            return
        self.add_module(
            "adn",
            ADN(
                ordering=adn_ordering,
                in_channels=out_channels,
                spatial_dims=spatial_dims,
                act=act,
                norm=norm,
                dropout=dropout,
            ),
        )


class ConvBlock(nn.Module):
    """Two ``Conv -> BatchNorm -> Dropout -> ReLU`` units; the first one carries the stride."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrSeq = 3,
        strides: IntOrSeq = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = [
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=None,
                adn_ordering="NDA",
                act="relu",
                norm="batch",
                dropout=dropout,
            ),
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=1,
                padding=None,
                adn_ordering="NDA",
                act="relu",
                norm="batch",
                dropout=dropout,
            ),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpConv(nn.Module):
    """Transposed ``Conv -> BatchNorm -> Dropout -> ReLU`` that undoes one level's stride."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrSeq = 3,
        strides: IntOrSeq = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.up = Convolution(
            spatial_dims,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=kernel_size,
            act="relu",
            adn_ordering="NDA",
            norm="batch",
            dropout=dropout,
            is_transposed=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class AttentionBlock(nn.Module):
    """Additive attention gate ``x * sigmoid(BN(psi(relu(BN(W_g g) + BN(W_x x)))))``.

    ``g`` (gating signal) and ``x`` (skip feature) have the same resolution, as in MONAI.
    """

    def __init__(self, spatial_dims: int, f_int: int, f_g: int, f_l: int, dropout: float = 0.0):
        super().__init__()
        self.W_g = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_g,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            _BATCH_NORM[spatial_dims](f_int),
        )
        self.W_x = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_l,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            _BATCH_NORM[spatial_dims](f_int),
        )
        self.psi = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_int,
                out_channels=1,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            _BATCH_NORM[spatial_dims](1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionLayer(nn.Module):
    """One U-Net level: ``submodule`` (encoder + deeper levels), up-convolution, gate and merge.

    As in MONAI, the gate and the up-convolution never receive the dropout rate, and the merge
    convolution uses MONAI's ``Convolution`` defaults (instance norm and PReLU).
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        submodule: nn.Module,
        up_kernel_size: IntOrSeq = 3,
        strides: IntOrSeq = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attention = AttentionBlock(
            spatial_dims=spatial_dims, f_g=in_channels, f_l=in_channels, f_int=in_channels // 2
        )
        self.upconv = UpConv(
            spatial_dims=spatial_dims,
            in_channels=out_channels,
            out_channels=in_channels,
            strides=strides,
            kernel_size=up_kernel_size,
        )
        self.merge = Convolution(
            spatial_dims=spatial_dims, in_channels=2 * in_channels, out_channels=in_channels, dropout=dropout
        )
        self.submodule = submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fromlower = self.upconv(self.submodule(x))
        att = self.attention(g=fromlower, x=x)
        return self.merge(torch.cat((att, fromlower), dim=1))


class AttentionUnet(nn.Module):
    """MONAI ``AttentionUnet`` (1.3.2) in plain PyTorch.

    Args:
        spatial_dims: 1, 2 or 3.
        in_channels: input channels.
        out_channels: output channels (logits, no activation).
        channels: feature widths, top level first; at least two entries.
        strides: with ``stride_mode="per_level"`` (MONAI), one stride per level
            (``len(channels) - 1`` entries are used), each an int or one value per spatial
            dimension. With ``stride_mode="shared"`` (the TS-SatFire copy), a single stride -- an
            int or one value per spatial dimension -- used at every level.
        kernel_size: convolution kernel size (odd).
        up_kernel_size: transposed-convolution kernel size (odd).
        dropout: dropout rate inside the encoder/decoder convolution blocks.
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
        dropout: float = 0.0,
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
            level_strides = level_strides[:n_levels]  # MONAI ignores extra entries as well
        else:
            raise ValueError(f"stride_mode must be 'per_level' or 'shared', got {stride_mode!r}")
        self.level_strides = [_as_tuple(s, spatial_dims, "each stride") for s in level_strides]
        for name, kernel in (("kernel_size", kernel_size), ("up_kernel_size", up_kernel_size)):
            _as_tuple(kernel, spatial_dims, name)
            _same_padding(kernel)

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.stride_mode = stride_mode
        self.kernel_size = kernel_size
        self.dropout = dropout

        # Creation order follows MONAI (head, output conv, then the levels from the bottom up),
        # so a given seed produces the same initial weights.
        head = ConvBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=channels[0],
            dropout=dropout,
            kernel_size=self.kernel_size,
        )
        reduce_channels = Convolution(
            spatial_dims=spatial_dims,
            in_channels=channels[0],
            out_channels=out_channels,
            kernel_size=1,
            strides=1,
            padding=0,
            conv_only=True,
        )
        self.up_kernel_size = up_kernel_size

        def _create_block(channels: Sequence[int], strides: Sequence[IntOrSeq]) -> nn.Module:
            if len(channels) > 2:
                subblock = _create_block(channels[1:], strides[1:])
                return AttentionLayer(
                    spatial_dims=spatial_dims,
                    in_channels=channels[0],
                    out_channels=channels[1],
                    submodule=nn.Sequential(
                        ConvBlock(
                            spatial_dims=spatial_dims,
                            in_channels=channels[0],
                            out_channels=channels[1],
                            strides=strides[0],
                            dropout=self.dropout,
                            kernel_size=self.kernel_size,
                        ),
                        subblock,
                    ),
                    up_kernel_size=self.up_kernel_size,
                    strides=strides[0],
                    dropout=dropout,
                )
            # The next level is the bottom one: stop the recursion.
            return self._get_bottom_layer(channels[0], channels[1], strides[0])

        encdec = _create_block(self.channels, level_strides)
        self.model = nn.Sequential(head, encdec, reduce_channels)

    def _get_bottom_layer(self, in_channels: int, out_channels: int, strides: IntOrSeq) -> nn.Module:
        return AttentionLayer(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            submodule=ConvBlock(
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                dropout=self.dropout,
                kernel_size=self.kernel_size,
            ),
            up_kernel_size=self.up_kernel_size,
            strides=strides,
            dropout=self.dropout,
        )

    def size_multiple(self) -> Tuple[int, ...]:
        """Per-dimension factor that every input spatial size must be a multiple of."""
        factors = [1] * self.dimensions
        for stride in self.level_strides:
            factors = [f * s for f, s in zip(factors, stride)]
        return tuple(factors)

    def _check_input(self, x: torch.Tensor, layout: str) -> None:
        if x.ndim != self.dimensions + 2:
            raise ValueError(
                f"{type(self).__name__} expects input shape {layout}, got {tuple(x.shape)}."
            )
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
        spatial = ", ".join(["H", "W"] if self.dimensions == 2 else ["D", "H", "W"][-self.dimensions :])
        self._check_input(x, f"(batch, channels, {spatial})")
        return self.model(x)


class TemporalAttentionUnet(AttentionUnet):
    """3D :class:`AttentionUnet` over ``(time, H, W)`` for raster time series (TS-SatFire usage).

    Takes PyHazards' ``(batch, time, channels, H, W)`` layout, runs the 3D network on
    ``(batch, channels, time, H, W)`` and, with ``time_reduction="mean"`` (TS-SatFire's
    prediction task), averages the logits over time to ``(batch, out_channels, H, W)``.
    ``time_reduction="none"`` returns ``(batch, out_channels, time, H, W)``, the per-date maps
    TS-SatFire uses for its active-fire and burned-area tasks. Parameter names are those of
    :class:`AttentionUnet`, so 3D MONAI / TS-SatFire state dicts load unchanged.
    """

    def __init__(self, *args, time_reduction: str = "mean", **kwargs):
        super().__init__(*args, **kwargs)
        if self.dimensions != 3:
            raise ValueError(f"TemporalAttentionUnet needs spatial_dims=3, got {self.dimensions}")
        if time_reduction not in {"mean", "none"}:
            raise ValueError(f"time_reduction must be 'mean' or 'none', got {time_reduction!r}")
        self.time_reduction = time_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                "TemporalAttentionUnet expects input shape (batch, time, channels, height, width), "
                f"got {tuple(x.shape)}."
            )
        x = x.transpose(1, 2).contiguous()
        self._check_input(x, "(batch, time, channels, height, width)")
        out = self.model(x)
        return out.mean(2) if self.time_reduction == "mean" else out


def attention_unet_builder(
    task: str,
    in_channels: int,
    out_channels: int = 2,
    spatial_dims: int = 3,
    channels: Sequence[int] = TS_SATFIRE_CHANNELS,
    strides: Optional[Union[IntOrSeq, Sequence[IntOrSeq]]] = None,
    stride_mode: str = "per_level",
    kernel_size: IntOrSeq = 3,
    up_kernel_size: IntOrSeq = 3,
    dropout: float = 0.0,
    time_reduction: str = "mean",
    **kwargs,
) -> nn.Module:
    """Build the Attention U-Net; the defaults are TS-SatFire's prediction model.

    ``spatial_dims=3`` returns a :class:`TemporalAttentionUnet` (input ``(B, T, C, H, W)``);
    ``spatial_dims=2`` returns an :class:`AttentionUnet` (input ``(B, C, H, W)``). Without
    ``strides``, TS-SatFire's strides are used: ``(1, 2, 2)`` at every level in 3D, ``2`` at every
    level in 2D.
    """
    _ = kwargs
    if task.lower() != "segmentation":
        raise ValueError(f"attention_unet supports task='segmentation', got {task!r}.")
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
        dropout=dropout,
        stride_mode=stride_mode,
    )
    if spatial_dims == 3:
        return TemporalAttentionUnet(spatial_dims=3, time_reduction=time_reduction, **common)
    if spatial_dims == 2:
        return AttentionUnet(spatial_dims=2, **common)
    raise ValueError(f"attention_unet supports spatial_dims 2 or 3, got {spatial_dims}.")


__all__ = [
    "ADN",
    "AttentionBlock",
    "AttentionLayer",
    "AttentionUnet",
    "ConvBlock",
    "Convolution",
    "TemporalAttentionUnet",
    "UpConv",
    "attention_unet_builder",
]
