"""MONAI building blocks shared by the ``unetr`` and ``swin_unetr`` ports.

Ported from MONAI 1.3.2 (Apache License 2.0, Copyright (c) MONAI Consortium):
``monai/networks/blocks/dynunet_block.py`` (``get_conv_layer``, ``UnetResBlock``,
``UnetBasicBlock``, ``UnetOutBlock``), ``monai/networks/blocks/unetr_block.py``
(``UnetrBasicBlock``, ``UnetrPrUpBlock``, ``UnetrUpBlock``), ``monai/networks/blocks/mlp.py``
(``MLPBlock``), ``monai/networks/layers/drop_path.py`` (``DropPath``) and
``monai/networks/layers/utils.py`` (``get_norm_layer``). The blocks are rewritten in plain
PyTorch with MONAI's module names and creation order, so MONAI state dicts load with
``strict=True`` and a given seed gives the same initial weights. Only the options that UNETR and
SwinUNETR use are supported: normalisation ``"batch"`` or ``"instance"`` (optionally as a
``(name, kwargs)`` tuple), LeakyReLU(0.01) activations and no dropout in the convolution blocks.

``trunc_normal_`` is MONAI's ``monai.networks.layers.trunc_normal_``, which uses the same
inverse-CDF algorithm as timm 0.4.12, so it is shared with :mod:`pyhazards.models.swin_blocks`.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .attention_unet import Convolution
from .swin_blocks import trunc_normal_

IntOrSeq = Union[int, Sequence[int]]
NormSpec = Union[str, Tuple[str, Dict[str, Any]]]

_BATCH_NORM = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}
_INSTANCE_NORM = {1: nn.InstanceNorm1d, 2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d}


def as_tuple(value: IntOrSeq, dims: int, name: str) -> Tuple[int, ...]:
    """``monai.utils.ensure_tuple_rep`` for ints, with a ``ValueError`` for wrong lengths."""
    if isinstance(value, (int, np.integer)):
        return (int(value),) * dims
    values = tuple(int(v) for v in value)
    if len(values) != dims:
        raise ValueError(f"{name} must be an int or have {dims} entries, got {value!r}")
    return values


def get_norm_layer(name: NormSpec, spatial_dims: int, channels: int) -> nn.Module:
    """``monai.networks.layers.get_norm_layer`` for batch and instance norm."""
    if isinstance(name, str):
        norm_name, norm_args = name, {}
    elif isinstance(name, (tuple, list)) and len(name) == 2 and isinstance(name[0], str):
        norm_name, norm_args = name[0], dict(name[1])
    else:
        raise ValueError(f"norm_name must be 'batch', 'instance' or (name, kwargs), got {name!r}")
    norm_name = norm_name.lower()
    if norm_name == "batch":
        return _BATCH_NORM[spatial_dims](**{"num_features": channels, **norm_args})
    if norm_name == "instance":
        return _INSTANCE_NORM[spatial_dims](**{"num_features": channels, **norm_args})
    raise ValueError(f"norm_name must be 'batch' or 'instance', got {name!r}")


def check_norm_name(name: NormSpec) -> None:
    get_norm_layer(name, 1, 1)


def get_padding(kernel_size: IntOrSeq, stride: IntOrSeq) -> IntOrSeq:
    kernel = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise ValueError("padding value should not be negative, please change the kernel size and/or stride.")
    padding = tuple(int(p) for p in padding_np)
    return padding if len(padding) > 1 else padding[0]


def get_output_padding(kernel_size: IntOrSeq, stride: IntOrSeq, padding: IntOrSeq) -> IntOrSeq:
    kernel = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)
    out_padding_np = 2 * padding_np + stride_np - kernel
    if np.min(out_padding_np) < 0:
        raise ValueError("out_padding value should not be negative, please change the kernel size and/or stride.")
    out_padding = tuple(int(p) for p in out_padding_np)
    return out_padding if len(out_padding) > 1 else out_padding[0]


def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: IntOrSeq = 3,
    stride: IntOrSeq = 1,
    bias: bool = False,
    is_transposed: bool = False,
) -> Convolution:
    """``monai.networks.blocks.dynunet_block.get_conv_layer`` without act/norm/dropout.

    Every call in UNETR and SwinUNETR builds a bare (transposed) convolution; MONAI's default
    ``bias=False`` is kept.
    """
    padding = get_padding(kernel_size, stride)
    output_padding = get_output_padding(kernel_size, stride, padding) if is_transposed else None
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=None,
        norm=None,
        dropout=None,
        bias=bias,
        conv_only=True,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )


def _leaky_relu() -> nn.Module:
    # MONAI's default act_name ("leakyrelu", {"inplace": True, "negative_slope": 0.01}).
    return nn.LeakyReLU(negative_slope=0.01, inplace=True)


class UnetResBlock(nn.Module):
    """Residual ``conv -> norm -> lrelu -> conv -> norm (+ 1x1 projection)`` block (MONAI)."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrSeq,
        stride: IntOrSeq,
        norm_name: NormSpec,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.conv2 = get_conv_layer(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1)
        self.lrelu = _leaky_relu()
        self.norm1 = get_norm_layer(norm_name, spatial_dims, out_channels)
        self.norm2 = get_norm_layer(norm_name, spatial_dims, out_channels)
        self.downsample = in_channels != out_channels
        if not np.all(np.atleast_1d(stride) == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=1, stride=stride)
            self.norm3 = get_norm_layer(norm_name, spatial_dims, out_channels)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out


class UnetBasicBlock(nn.Module):
    """``conv -> norm -> lrelu -> conv -> norm -> lrelu`` block (MONAI)."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrSeq,
        stride: IntOrSeq,
        norm_name: NormSpec,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.conv2 = get_conv_layer(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1)
        self.lrelu = _leaky_relu()
        self.norm1 = get_norm_layer(norm_name, spatial_dims, out_channels)
        self.norm2 = get_norm_layer(norm_name, spatial_dims, out_channels)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        out = self.lrelu(self.norm1(self.conv1(inp)))
        return self.lrelu(self.norm2(self.conv2(out)))


class UnetOutBlock(nn.Module):
    """1x1 output convolution with bias (MONAI)."""

    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=1, stride=1, bias=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.conv(inp)


class UnetrBasicBlock(nn.Module):
    """UNETR encoder block: a :class:`UnetResBlock` or :class:`UnetBasicBlock` named ``layer``."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrSeq,
        stride: IntOrSeq,
        norm_name: NormSpec,
        res_block: bool = False,
    ):
        super().__init__()
        block = UnetResBlock if res_block else UnetBasicBlock
        self.layer = block(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm_name=norm_name,
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.layer(inp)


class UnetrPrUpBlock(nn.Module):
    """UNETR projection-upsampling block: one transposed convolution, then ``num_layer`` more.

    With ``conv_block=True`` each extra up-sampling is followed by a residual or basic block.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_layer: int,
        kernel_size: IntOrSeq,
        stride: IntOrSeq,
        upsample_kernel_size: IntOrSeq,
        norm_name: NormSpec,
        conv_block: bool = False,
        res_block: bool = False,
    ):
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv_init = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            is_transposed=True,
        )

        def _up() -> Convolution:
            return get_conv_layer(
                spatial_dims,
                out_channels,
                out_channels,
                kernel_size=upsample_kernel_size,
                stride=upsample_stride,
                is_transposed=True,
            )

        if conv_block:
            block = UnetResBlock if res_block else UnetBasicBlock
            self.blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        _up(),
                        block(
                            spatial_dims=spatial_dims,
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            norm_name=norm_name,
                        ),
                    )
                    for _ in range(num_layer)
                ]
            )
        else:
            self.blocks = nn.ModuleList([_up() for _ in range(num_layer)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transp_conv_init(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class UnetrUpBlock(nn.Module):
    """UNETR decoder block: transposed convolution, concatenation with the skip, conv block."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrSeq,
        upsample_kernel_size: IntOrSeq,
        norm_name: NormSpec,
        res_block: bool = False,
    ):
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            is_transposed=True,
        )
        block = UnetResBlock if res_block else UnetBasicBlock
        self.conv_block = block(
            spatial_dims,
            out_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
        )

    def forward(self, inp: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        return self.conv_block(out)


class MLPBlock(nn.Module):
    """``Linear -> GELU -> Dropout -> Linear -> Dropout`` (MONAI ``MLPBlock``).

    ``dropout_mode="vit"`` uses two dropout modules, ``"swin"`` reuses the first one.
    """

    def __init__(self, hidden_size: int, mlp_dim: int, dropout_rate: float = 0.0, dropout_mode: str = "vit"):
        super().__init__()
        if not 0 <= dropout_rate <= 1:
            raise ValueError("dropout_rate should be between 0 and 1.")
        mlp_dim = mlp_dim or hidden_size
        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        self.fn = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)
        if dropout_mode == "vit":
            self.drop2 = nn.Dropout(dropout_rate)
        elif dropout_mode == "swin":
            self.drop2 = self.drop1
        else:
            raise ValueError(f"dropout_mode should be 'vit' or 'swin', got {dropout_mode!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.fn(self.linear1(x)))
        return self.drop2(self.linear2(x))


class DropPath(nn.Module):
    """Per-sample stochastic depth (MONAI ``DropPath``: Bernoulli mask, scaled by the keep rate)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        if not 0 <= drop_prob <= 1:
            raise ValueError("Drop path prob should be between 0 and 1.")
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class TemporalReadout:
    """Mixin for 3D networks over ``(time, H, W)`` fed with PyHazards' ``(B, T, C, H, W)`` layout.

    Subclasses call :meth:`_to_network_layout` before the network and :meth:`_read_out` after it.
    With ``time_reduction="mean"`` (TS-SatFire's prediction script, ``outputs.mean(2)``) the
    logits are averaged over time to ``(B, classes, H, W)``; ``"none"`` keeps
    ``(B, classes, T, H, W)``.
    """

    time_reduction: str

    def _set_time_reduction(self, time_reduction: str) -> None:
        if time_reduction not in {"mean", "none"}:
            raise ValueError(f"time_reduction must be 'mean' or 'none', got {time_reduction!r}")
        self.time_reduction = time_reduction

    @staticmethod
    def _to_network_layout(x: torch.Tensor, name: str) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                f"{name} expects input shape (batch, time, channels, height, width), got {tuple(x.shape)}."
            )
        return x.transpose(1, 2).contiguous()

    def _read_out(self, out: torch.Tensor) -> torch.Tensor:
        return out.mean(2) if self.time_reduction == "mean" else out


__all__ = [
    "DropPath",
    "MLPBlock",
    "TemporalReadout",
    "UnetBasicBlock",
    "UnetOutBlock",
    "UnetResBlock",
    "UnetrBasicBlock",
    "UnetrPrUpBlock",
    "UnetrUpBlock",
    "get_conv_layer",
    "get_norm_layer",
    "trunc_normal_",
]
