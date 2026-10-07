"""Swin Transformer building blocks shared by ``swin_unet`` and ``asufm``.

Attribution:

- Window attention, the Swin block, patch embedding and patch merging are ported from
  microsoft/Swin-Transformer ``models/swin_transformer.py`` at commit
  ``f82860bfb5225915aca09c3227159ee9e1df874d`` (MIT License, Copyright (c) 2021 Microsoft).
- ``DropPath`` and ``trunc_normal_`` follow ``timm.models.layers`` from timm 0.4.12 (Apache-2.0,
  Copyright 2019 Ross Wightman; ``trunc_normal_`` is itself taken from PyTorch, BSD-3-Clause),
  the version Swin-Transformer pins, so initialisation and stochastic depth draw the same random
  numbers as the reference implementations. (``torch.nn.init.trunc_normal_`` switched to
  rejection sampling in recent PyTorch releases and no longer reproduces them.)
- ``FocalModulation`` is ported from microsoft/FocalNet ``classification/focalnet.py`` at commit
  ``e8514bdcfe1eb6e7e110403e9f2972c67dfed1c3`` (MIT License, Copyright (c) Microsoft Corporation).
- The decoder pieces (``PatchExpand``, ``FinalPatchExpand_X4``, ``BasicLayer_up``) are written for
  PyHazards from the description in the Swin-Unet paper (Cao et al., ECCV Workshops 2022,
  arXiv:2105.05537, Sec. 3.4-3.5): a linear layer doubles (or, for the last layer, multiplies by
  16) the channel dimension and the result is rearranged into a 2x (4x) larger token grid. The
  official Swin-Unet repository has no license, so none of its code is copied; module and
  attribute names follow it so that its state dicts load with ``strict=True``.

All blocks operate on token sequences ``(batch, height * width, channels)`` whose spatial size is
fixed at construction (``input_resolution``), as in the references.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint as torch_checkpoint


def to_2tuple(value: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(value, (tuple, list)):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0) -> torch.Tensor:
    """Truncated normal fill by inverse-CDF sampling (timm 0.4.12 / PyTorch 1.x algorithm)."""

    def norm_cdf(value: float) -> float:
        return (1.0 + math.erf(value / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        low, high = norm_cdf((a - mean) / std), norm_cdf((b - mean) / std)
        tensor.uniform_(2 * low - 1, 2 * high - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Per-sample stochastic depth with the random draw of timm 0.4.12."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


def stochastic_depth_rates(drop_path_rate: float, depths: Sequence[int]) -> List[float]:
    """Linearly increasing drop-path rates over all encoder blocks (Swin's decay rule)."""
    return [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """``(B, H, W, C)`` -> ``(B * num_windows, window_size, window_size, C)``."""
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    """Inverse of :func:`window_partition`."""
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class WindowAttention(nn.Module):
    """Window multi-head self-attention with a learned relative position bias (W-MSA / SW-MSA)."""

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        coords = torch.stack(
            torch.meshgrid([torch.arange(window_size[0]), torch.arange(window_size[1])], indexing="ij")
        )
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = (coords_flatten[:, :, None] - coords_flatten[:, None, :]).permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # Initialised after the projections, as in the reference (keeps seeded init identical).
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)

        area = self.window_size[0] * self.window_size[1]
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(area, area, -1)
        attn = attn + bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        if mask is not None:
            n_windows = mask.shape[0]
            attn = attn.view(b_ // n_windows, n_windows, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.attn_drop(self.softmax(attn))

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        return self.proj_drop(self.proj(x))


class FocalModulation(nn.Module):
    """Focal modulation (Yang et al., NeurIPS 2022) on ``(B, H, W, C)`` features."""

    def __init__(
        self,
        dim: int,
        focal_window: int,
        focal_level: int,
        focal_factor: int = 2,
        bias: bool = True,
        proj_drop: float = 0.0,
        use_postln_in_modulation: bool = False,
        normalize_modulator: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        self.f = nn.Linear(dim, 2 * dim + (focal_level + 1), bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)
        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()
        self.kernel_sizes = []
        for k in range(focal_level):
            kernel_size = focal_factor * k + focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, groups=dim, padding=kernel_size // 2, bias=False),
                    nn.GELU(),
                )
            )
            self.kernel_sizes.append(kernel_size)
        if use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[-1]
        x = self.f(x).permute(0, 3, 1, 2).contiguous()
        q, ctx, gates = torch.split(x, (c, c, self.focal_level + 1), 1)

        ctx_all = 0
        for level in range(self.focal_level):
            ctx = self.focal_layers[level](ctx)
            ctx_all = ctx_all + ctx * gates[:, level : level + 1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level :]
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)

        x_out = (q * self.h(ctx_all)).permute(0, 2, 3, 1).contiguous()
        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)
        return self.proj_drop(self.proj(x_out))


class SwinTransformerBlock(nn.Module):
    """Swin block: (shifted) window attention and an MLP, each with a pre-norm residual.

    ``modulation`` optionally inserts a focal-modulation module between ``norm1`` and the
    attention (ASUFM). It is registered whenever given; ``use_modulation`` decides whether the
    forward pass applies it (ASUFM builds it in its decoder blocks but never uses it there).
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        modulation: Optional[nn.Module] = None,
        use_modulation: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = tuple(input_resolution)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # A window covering the whole map is neither partitioned nor shifted.
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        if not 0 <= self.shift_size < self.window_size:
            raise ValueError(f"shift_size must be in [0, window_size), got {self.shift_size}")
        self.use_modulation = bool(use_modulation and modulation is not None)

        self.norm1 = norm_layer(dim)
        if modulation is not None:
            self.modulation = modulation
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.register_buffer("attn_mask", self._shifted_window_mask() if self.shift_size > 0 else None)

    def _shifted_window_mask(self) -> torch.Tensor:
        """-100 between tokens that come from different regions after the cyclic shift."""
        h, w = self.input_resolution
        regions = torch.zeros((1, h, w, 1))
        bounds = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        label = 0
        for rows in bounds:
            for cols in bounds:
                regions[:, rows, cols, :] = label
                label += 1
        labels = window_partition(regions, self.window_size).view(-1, self.window_size * self.window_size)
        mask = labels.unsqueeze(1) - labels.unsqueeze(2)
        return mask.masked_fill(mask != 0, float(-100.0)).masked_fill(mask == 0, float(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = self.input_resolution
        b, length, c = x.shape
        if length != h * w:
            raise ValueError(f"Swin block expected {h * w} tokens for resolution {(h, w)}, got {length}.")

        shortcut = x
        x = self.norm1(x).view(b, h, w, c)
        if self.use_modulation:
            # ASUFM applies norm1 a second time to the modulated features.
            x = self.norm1(self.modulation(x))
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        windows = window_partition(x, self.window_size).view(-1, self.window_size * self.window_size, c)
        windows = self.attn(windows, mask=self.attn_mask).view(-1, self.window_size, self.window_size, c)
        x = window_reverse(windows, self.window_size, h, w)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = shortcut + self.drop_path(x.view(b, h * w, c))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    """Non-overlapping ``patch_size`` convolution from an image to a token sequence."""

    def __init__(self, img_size=224, patch_size=4, in_chans: int = 3, embed_dim: int = 96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x) if self.norm is not None else x


class PatchMerging(nn.Module):
    """Concatenate each 2x2 neighbourhood (4C channels), normalise, and project to 2C."""

    def __init__(self, input_resolution: Tuple[int, int], dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = tuple(input_resolution)
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = self.input_resolution
        b, _, c = x.shape
        x = x.view(b, h, w, c)
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1)
        return self.reduction(self.norm(x.view(b, -1, 4 * c)))


def _tokens_to_finer_grid(x: torch.Tensor, h: int, w: int, scale: int) -> torch.Tensor:
    """Split each token's channels into a ``scale x scale`` block of tokens (row-major).

    ``(B, h * w, scale * scale * c)`` -> ``(B, h * scale * w * scale, c)``; channel index
    ``(i * scale + j) * c + k`` of token ``(r, s)`` becomes channel ``k`` of token
    ``(r * scale + i, s * scale + j)``.
    """
    b, length, channels = x.shape
    if length != h * w:
        raise ValueError(f"expected {h * w} tokens for resolution {(h, w)}, got {length}.")
    c = channels // (scale * scale)
    x = x.view(b, h, w, scale, scale, c).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(b, h * scale * w * scale, c)


class PatchExpand(nn.Module):
    """Swin-Unet patch expanding: 2x more tokens with half the channels."""

    def __init__(self, input_resolution: Tuple[int, int], dim: int, dim_scale: int = 2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = tuple(input_resolution)
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = self.input_resolution
        return self.norm(_tokens_to_finer_grid(self.expand(x), h, w, 2))


class FinalPatchExpand_X4(nn.Module):
    """Last Swin-Unet expanding layer: 4x more tokens per side, ``dim`` channels each."""

    def __init__(self, input_resolution: Tuple[int, int], dim: int, dim_scale: int = 4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = tuple(input_resolution)
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = self.input_resolution
        return self.norm(_tokens_to_finer_grid(self.expand(x), h, w, self.dim_scale))


def _run_blocks(blocks: nn.ModuleList, x: torch.Tensor, use_checkpoint: bool) -> torch.Tensor:
    for block in blocks:
        if use_checkpoint:
            x = torch_checkpoint.checkpoint(block, x, use_reentrant=False)
        else:
            x = block(x)
    return x


def _make_blocks(
    dim: int,
    input_resolution: Tuple[int, int],
    depth: int,
    num_heads: int,
    window_size: int,
    mlp_ratio: float,
    qkv_bias: bool,
    qk_scale: Optional[float],
    drop: float,
    attn_drop: float,
    drop_path: Union[float, Sequence[float]],
    norm_layer,
    focal_modulation: Optional[str],
) -> nn.ModuleList:
    blocks = []
    for i in range(depth):
        modulation = None
        if focal_modulation is not None:
            # Built before the block's attention, matching the reference creation order.
            modulation = FocalModulation(dim, focal_window=3, focal_level=1, proj_drop=drop)
        blocks.append(
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                norm_layer=norm_layer,
                modulation=modulation,
                use_modulation=focal_modulation == "applied",
            )
        )
    return nn.ModuleList(blocks)


class BasicLayer(nn.Module):
    """One encoder stage: ``depth`` Swin blocks (alternating shift) and optional patch merging.

    ``focal_modulation`` is None (plain Swin), ``"applied"`` or ``"unused"`` (see
    :class:`SwinTransformerBlock`).
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Union[float, Sequence[float]] = 0.0,
        norm_layer=nn.LayerNorm,
        downsample: bool = False,
        use_checkpoint: bool = False,
        focal_modulation: Optional[str] = None,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = tuple(input_resolution)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = _make_blocks(
            dim, input_resolution, depth, num_heads, window_size, mlp_ratio, qkv_bias, qk_scale,
            drop, attn_drop, drop_path, norm_layer, focal_modulation,
        )
        self.downsample = PatchMerging(input_resolution, dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _run_blocks(self.blocks, x, self.use_checkpoint)
        return self.downsample(x) if self.downsample is not None else x


class BasicLayer_up(nn.Module):
    """One decoder stage: ``depth`` Swin blocks and optional patch expanding."""

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Union[float, Sequence[float]] = 0.0,
        norm_layer=nn.LayerNorm,
        upsample: bool = False,
        use_checkpoint: bool = False,
        focal_modulation: Optional[str] = None,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = tuple(input_resolution)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = _make_blocks(
            dim, input_resolution, depth, num_heads, window_size, mlp_ratio, qkv_bias, qk_scale,
            drop, attn_drop, drop_path, norm_layer, focal_modulation,
        )
        self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer) if upsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _run_blocks(self.blocks, x, self.use_checkpoint)
        return self.upsample(x) if self.upsample is not None else x


def init_swin_weights(module: nn.Module) -> None:
    """Truncated-normal(0.02) linear weights, zero biases, unit LayerNorms (``model.apply``)."""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)


def check_swin_geometry(img_size: int, patch_size: int, window_size: int, num_layers: int) -> None:
    """Raise ``ValueError`` unless every stage's token grid splits into whole windows."""
    if img_size % patch_size:
        raise ValueError(f"img_size {img_size} must be divisible by patch_size {patch_size}.")
    resolution = img_size // patch_size
    for stage in range(num_layers):
        side = resolution // (2 ** stage)
        if stage < num_layers - 1 and side % 2:
            raise ValueError(
                f"img_size {img_size}: stage {stage} has an odd token grid ({side}) and cannot be merged."
            )
        window = min(window_size, side)
        if side % window:
            raise ValueError(
                f"img_size {img_size}: stage {stage} token grid ({side}) is not divisible by window_size {window}."
            )


__all__ = [
    "BasicLayer",
    "BasicLayer_up",
    "DropPath",
    "FinalPatchExpand_X4",
    "FocalModulation",
    "Mlp",
    "PatchEmbed",
    "PatchExpand",
    "PatchMerging",
    "SwinTransformerBlock",
    "WindowAttention",
    "check_swin_geometry",
    "drop_path",
    "init_swin_weights",
    "stochastic_depth_rates",
    "to_2tuple",
    "trunc_normal_",
    "window_partition",
    "window_reverse",
]
