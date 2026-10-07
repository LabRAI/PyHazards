"""SwinUNETR as implemented by MONAI, with TS-SatFire's spatio-temporal (SwinUNETR-3D) modifications.

Port of ``monai.networks.nets.SwinUNETR`` from MONAI 1.3.2 (``monai/networks/nets/swin_unetr.py``,
``monai/networks/blocks/patchembedding.py`` ``PatchEmbed`` and the UNETR blocks in
:mod:`pyhazards.models.monai_blocks`; Apache License 2.0, Copyright (c) MONAI Consortium).
TS-SatFire's copy (``spatial_models/swinunetr/``, itself an adaptation of MONAI's file) is
reproduced through the options described below. The TS-SatFire repository has no license, so its
code is not copied: its changes are re-implemented on top of the MONAI port from their behaviour
and checked against it as a test oracle. The Swin-V2 attention follows the algorithm of
microsoft/Swin-Transformer ``models/swin_transformer_v2.py`` (``WindowAttention``; MIT License,
Copyright (c) Microsoft Corporation), extended to 3D windows as in TS-SatFire. Changes made for
PyHazards: plain PyTorch (no MONAI or einops dependency), invalid configurations and input shapes
raise ``ValueError`` up front, and :class:`TemporalSwinUNETR` adds the TS-SatFire input layout and
temporal read-out. Module names, creation order and defaults follow MONAI (or TS-SatFire, for its
options), so their state dicts load with ``strict=True`` and the same seed gives the same initial
weights.

SwinUNETR is Hatamizadeh et al., "Swin UNETR: Swin Transformers for Semantic Segmentation of
Brain Tumors in MRI Images" (BrainLes 2021, LNCS 12962, arXiv:2201.01266): a four-stage Swin
Transformer encoder (shifted-window attention, patch merging) whose five feature levels feed a
convolutional U-Net decoder of residual blocks. The authors' official code
(Project-MONAI/research-contributions) uses MONAI's implementation.

TS-SatFire (Zhao, Gerard & Ban, Scientific Data 12:1817, 2025) uses it in two ways:
"SwinUNETR-2D" is stock MONAI ``SwinUNETR(img_size=(256, 256), spatial_dims=2, feature_size=48,
norm_name="batch")`` (25,151,996 parameters with 8 input channels, Table 3: 25.2M);
"SwinUNETR-3D" uses the repository's copy, which differs from MONAI 1.3.2 in that

- ``patch_size`` and ``window_size`` are arguments -- ``(1, 2, 2)`` and ``(T, 4, 4)`` over
  (time, height, width), so patches never span several days and every attention window covers
  the whole time series (MONAI hard-codes 2 and 7);
- patch merging (``downsample="ts_satfire"`` here) concatenates neighbours only along the
  dimensions whose size is even and down-samples only those; the decision is made at
  construction from ``img_size`` (before patch embedding), and the decoder up-samples by the same
  per-stage factors. With T = 6 the time axis is therefore halved once, by the first merging
  (6 -> 3), and kept at 3 afterwards; with T = 4 it is halved twice; with T = 2 once. Its
  eight-way concatenation order also differs from MONAI's ``"merging"`` (which repeats two of the
  eight slices) and ``"mergingv2"``;
- ``attn_version`` selects MONAI's window attention (``"v1"``, used by the prediction script) or
  Swin-V2 cosine attention with a continuous relative-position MLP (``"v2"``);
- the output block is wrapped in ``nn.Sequential`` (keys ``out.0.*``; ``wrap_out=True``), and
  MONAI's ``feature_size % 12`` check is dropped.
"""

from __future__ import annotations

import itertools
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint

from .monai_blocks import (
    DropPath,
    MLPBlock,
    NormSpec,
    TemporalReadout,
    UnetOutBlock,
    UnetrBasicBlock,
    UnetrUpBlock,
    as_tuple,
    check_norm_name,
    trunc_normal_,
)

IntOrSeq = Union[int, Sequence[int]]

_CONV = {2: nn.Conv2d, 3: nn.Conv3d}
DOWNSAMPLE_MODES = ("merging", "mergingv2", "ts_satfire")
ATTENTION_VERSIONS = ("v1", "v2")

# TS-SatFire SwinUNETR-3D (run_spatial_temp_model_pred.py): window (T, 4, 4), patch (1, 2, 2).
TS_SATFIRE_SWIN3D_PATCH = (1, 2, 2)
TS_SATFIRE_SWIN3D_WINDOW_HW = (4, 4)


def _channels_last(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, *range(2, x.ndim), 1)


def _channels_first(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, x.ndim - 1, *range(1, x.ndim - 1))


def window_partition(x: torch.Tensor, window_size: Sequence[int]) -> torch.Tensor:
    """Split ``(b, [d,] h, w, c)`` into ``(num_windows * b, window_volume, c)`` (MONAI)."""
    x_shape = x.size()
    if len(x_shape) == 5:
        b, d, h, w, c = x_shape
        x = x.view(
            b,
            d // window_size[0],
            window_size[0],
            h // window_size[1],
            window_size[1],
            w // window_size[2],
            window_size[2],
            c,
        )
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)
    else:
        b, h, w, c = x.shape
        x = x.view(b, h // window_size[0], window_size[0], w // window_size[1], window_size[1], c)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0] * window_size[1], c)
    return windows


def window_reverse(windows: torch.Tensor, window_size: Sequence[int], dims: Sequence[int]) -> torch.Tensor:
    """Inverse of :func:`window_partition` (MONAI)."""
    if len(dims) == 4:
        b, d, h, w = dims
        x = windows.view(
            b,
            d // window_size[0],
            h // window_size[1],
            w // window_size[2],
            window_size[0],
            window_size[1],
            window_size[2],
            -1,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)
    else:
        b, h, w = dims
        x = windows.view(b, h // window_size[0], w // window_size[1], window_size[0], window_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    """Shrink the window (and drop the shift) along dimensions not larger than the window (MONAI)."""
    use_window_size = list(window_size)
    use_shift_size = list(shift_size) if shift_size is not None else None
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if use_shift_size is not None:
                use_shift_size[i] = 0
    if use_shift_size is None:
        return tuple(use_window_size)
    return tuple(use_window_size), tuple(use_shift_size)


def compute_mask(dims: Sequence[int], window_size, shift_size, device) -> torch.Tensor:
    """Attention mask for shifted windows (MONAI)."""
    cnt = 0
    if len(dims) == 3:
        d, h, w = dims
        img_mask = torch.zeros((1, d, h, w, 1), device=device)
        for ds in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for hs in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                for ws in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                    img_mask[:, ds, hs, ws, :] = cnt
                    cnt += 1
    else:
        h, w = dims
        img_mask = torch.zeros((1, h, w, 1), device=device)
        for hs in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for ws in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                img_mask[:, hs, ws, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size).squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))


def _relative_position_index(window_size: Sequence[int]) -> torch.Tensor:
    coords = torch.stack(torch.meshgrid(*[torch.arange(w) for w in window_size], indexing="ij"))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    for i, w in enumerate(window_size):
        relative_coords[:, :, i] += w - 1
    if len(window_size) == 3:
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * window_size[2] - 1
    else:
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    return relative_coords.sum(-1)


def _apply_mask_and_softmax(attn: torch.Tensor, mask: Optional[torch.Tensor], num_heads: int, softmax: nn.Module):
    if mask is not None:
        b, n = attn.shape[0], attn.shape[-1]
        nw = mask.shape[0]
        attn = attn.view(b // nw, nw, num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, n, n)
    return softmax(attn)


class WindowAttention(nn.Module):
    """Window attention with a learned relative-position bias table (MONAI; TS-SatFire ``"v1"``)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        table_size = int(np.prod([2 * w - 1 for w in window_size]))
        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))
        self.register_buffer("relative_position_index", _relative_position_index(window_size))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        # MONAI indexes the full-window table with [:n, :n] also when the window was shrunk.
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        attn = attn + relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        attn = _apply_mask_and_softmax(attn, mask, self.num_heads, self.softmax)
        attn = self.attn_drop(attn).to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class WindowAttentionV2(nn.Module):
    """Swin-V2 window attention: scaled cosine attention and a log-spaced continuous position bias.

    TS-SatFire's ``WindowAttentionV2`` (``attn_version="v2"``), after Liu et al., "Swin Transformer
    V2" (CVPR 2022): ``cos(q, k) * exp(min(logit_scale, log 100)) + 16 * sigmoid(cpb_mlp(coords))``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if any(w < 2 for w in window_size):
            # The reference divides the relative coordinates by (window - 1): NaN for a window of 1.
            raise ValueError(f"attn_version='v2' needs window sizes >= 2, got {tuple(window_size)}")
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        self.cpb_mlp = nn.Sequential(
            nn.Linear(len(window_size), 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False)
        )
        axes = [torch.arange(-(w - 1), w, dtype=torch.float32) for w in window_size]
        table = torch.stack(torch.meshgrid(*axes, indexing="ij"))
        table = table.permute(*range(1, len(window_size) + 1), 0).contiguous().unsqueeze(0)
        for i, w in enumerate(window_size):
            # TS-SatFire indexes [:, :, :, i] -- the coordinate axis in 2D, but in 3D the i-th entry
            # of the width axis (all three coordinates there are divided by window[i] - 1). Kept.
            table[:, :, :, i] /= w - 1
        table *= 8  # normalise to [-8, 8]
        table = torch.sign(table) * torch.log2(torch.abs(table) + 1.0) / np.log2(8)
        self.register_buffer("relative_coords_table", table)
        self.register_buffer("relative_position_index", _relative_position_index(window_size))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        max_scale = torch.log(torch.tensor(1.0 / 0.01, device=x.device))
        attn = attn * torch.clamp(self.logit_scale, max=max_scale).exp()
        table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = table[self.relative_position_index.clone()[:n, :n].reshape(-1)].reshape(n, n, -1)
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias.permute(2, 0, 1).contiguous())
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = _apply_mask_and_softmax(attn, mask, self.num_heads, self.softmax)
        attn = self.attn_drop(attn).to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class SwinTransformerBlock(nn.Module):
    """(Shifted-)window attention and MLP with pre-norm residuals (MONAI)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        use_checkpoint: bool = False,
        attn_version: str = "v1",
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.norm1 = nn.LayerNorm(dim)
        attention = WindowAttention if attn_version == "v1" else WindowAttentionV2
        self.attn = attention(
            dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPBlock(hidden_size=dim, mlp_dim=int(dim * mlp_ratio), dropout_rate=drop, dropout_mode="swin")

    def forward_part1(self, x: torch.Tensor, mask_matrix: Optional[torch.Tensor]) -> torch.Tensor:
        x_shape = x.size()
        x = self.norm1(x)
        if len(x_shape) == 5:
            b, d, h, w, c = x.shape
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
            pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
            pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_d1))
            _, dp, hp, wp, _ = x.shape
            dims = [b, dp, hp, wp]
        else:
            b, h, w, c = x.shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            pad_d1 = 0
            pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
            pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
            _, hp, wp, _ = x.shape
            dims = [b, hp, wp]
        roll_dims = tuple(range(1, len(x_shape) - 1))
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=tuple(-s for s in shift_size), dims=roll_dims)
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        attn_windows = self.attn(window_partition(shifted_x, window_size), mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=tuple(shift_size), dims=roll_dims)
        else:
            x = shifted_x
        if len(x_shape) == 5:
            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x = x[:, :d, :h, :w, :].contiguous()
        elif pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()
        return x

    def forward_part2(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x: torch.Tensor, mask_matrix: Optional[torch.Tensor]) -> torch.Tensor:
        shortcut = x
        if self.use_checkpoint:
            x = torch_checkpoint.checkpoint(self.forward_part1, x, mask_matrix, use_reentrant=False)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + torch_checkpoint.checkpoint(self.forward_part2, x, use_reentrant=False)
        else:
            x = x + self.forward_part2(x)
        return x


class PatchMergingV2(nn.Module):
    """Concatenate each 2x2(x2) neighbourhood, ``LayerNorm``, linear to ``2 * dim`` (MONAI ``"mergingv2"``)."""

    def __init__(self, dim: int, spatial_dims: int = 3):
        super().__init__()
        self.dim = dim
        if spatial_dims == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = nn.LayerNorm(8 * dim)
        else:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            _, d, h, w, _ = x.shape
            if (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1):
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
            x = torch.cat([x[:, i::2, j::2, k::2, :] for i, j, k in itertools.product(range(2), range(2), range(2))], -1)
        else:
            _, h, w, _ = x.shape
            if (h % 2 == 1) or (w % 2 == 1):
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            x = torch.cat([x[:, j::2, i::2, :] for i, j in itertools.product(range(2), range(2))], -1)
        return self.reduction(self.norm(x))


class PatchMerging(PatchMergingV2):
    """MONAI's ``"merging"`` mode, "the PatchMerging module previously defined in v0.9.0".

    In 3D its eight slices are ``(0,0,0) (1,0,0) (0,1,0) (0,0,1) (1,0,1) (0,1,0) (0,0,1) (1,1,1)``:
    ``(0,1,0)`` and ``(0,0,1)`` appear twice and ``(1,1,0)``, ``(0,1,1)`` never (kept for
    compatibility with MONAI's pretrained weights). 2D inputs use :class:`PatchMergingV2`.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return super().forward(x)
        _, d, h, w, _ = x.shape
        if (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1):
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        return self.reduction(self.norm(x))


class AdaptivePatchMerging(nn.Module):
    """TS-SatFire's 3D patch merging: merge only the dimensions in ``merge`` (``downsample="ts_satfire"``).

    ``merge`` is fixed at construction (TS-SatFire derives it from the stage resolution computed
    from ``img_size``). The channels of the 2^k merged neighbours are concatenated in TS-SatFire's
    order, normalised and projected to ``2 * dim``; without any merged dimension only the norm and
    the projection are applied.
    """

    def __init__(self, dim: int, merge: Sequence[bool]):
        super().__init__()
        self.dim = dim
        self.merge = tuple(bool(m) for m in merge)
        dims_tmp = dim * 2 ** sum(self.merge)
        self.resample_scale = [2 if m else 1 for m in self.merge]
        self.norm = nn.LayerNorm(dims_tmp)
        self.output_dims = 2 * dim
        self.reduction = nn.Linear(dims_tmp, self.output_dims, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, H, W, C)
        merged = [axis + 1 for axis, m in enumerate(self.merge) if m]
        if len(merged) == 3:
            offsets = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
        elif len(merged) == 2:
            offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]
        elif len(merged) == 1:
            offsets = [(0,), (1,)]
        else:
            offsets = []
        if offsets:
            parts = []
            for offset in offsets:
                index: List[slice] = [slice(None)] * x.ndim
                for axis, start in zip(merged, offset):
                    index[axis] = slice(start, None, 2)
                parts.append(x[tuple(index)])
            x = torch.cat(parts, dim=-1)
        return self.reduction(self.norm(x))


class BasicLayer(nn.Module):
    """One Swin stage: ``depth`` blocks alternating unshifted / half-window-shifted, then merging (MONAI)."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: List[float],
        mlp_ratio: float,
        qkv_bias: bool,
        drop: float,
        attn_drop: float,
        downsample: Callable[[], nn.Module],
        use_checkpoint: bool = False,
        attn_version: str = "v1",
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for _ in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                    use_checkpoint=use_checkpoint,
                    attn_version=attn_version,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        spatial = tuple(x.shape[2:])
        window_size, shift_size = get_window_size(spatial, self.window_size, self.shift_size)
        x = _channels_last(x)
        padded = [int(np.ceil(s / w)) * w for s, w in zip(spatial, window_size)]
        attn_mask = compute_mask(padded, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(b, *spatial, -1)
        x = self.downsample(x)
        return _channels_first(x)


class PatchEmbed(nn.Module):
    """Strided-convolution patch embedding, zero-padding inputs to a multiple of the patch (MONAI)."""

    def __init__(self, patch_size: Sequence[int], in_chans: int, embed_dim: int, spatial_dims: int = 3):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.embed_dim = embed_dim
        self.proj = _CONV[spatial_dims](in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = None  # MONAI's SwinTransformer uses patch_norm=False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pads: List[int] = []
        for size, patch in zip(reversed(x.shape[2:]), reversed(self.patch_size)):
            pads += [0, (patch - size % patch) % patch]
        if any(pads):
            x = F.pad(x, pads)
        return self.proj(x)


class SwinTransformer(nn.Module):
    """Four-stage Swin encoder returning the (optionally layer-normalised) features of every level.

    ``stage_scales`` holds each stage's down-sampling factor per dimension (2 everywhere for MONAI's
    merging; TS-SatFire's plan for ``downsample="ts_satfire"``); ``resamples`` lists them deepest
    first, as TS-SatFire's ``SwinTransformer.resamples``.
    """

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        window_size: Sequence[int],
        patch_size: Sequence[int],
        depths: Sequence[int],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample: str = "merging",
        use_v2: bool = False,
        img_size: Optional[Sequence[int]] = None,
        attn_version: str = "v1",
    ):
        super().__init__()
        if downsample not in DOWNSAMPLE_MODES:
            raise ValueError(f"downsample must be one of {DOWNSAMPLE_MODES}, got {downsample!r}")
        if downsample == "ts_satfire" and img_size is None:
            raise ValueError("downsample='ts_satfire' needs img_size: its merging plan is fixed at construction.")
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = False
        self.window_size = tuple(window_size)
        self.patch_size = tuple(patch_size)
        self.patch_embed = PatchEmbed(self.patch_size, in_chans, embed_dim, spatial_dims=spatial_dims)
        self.pos_drop = nn.Dropout(p=drop_rate)
        # torch.linspace(...).item() as in the references, on the CPU so meta-device builds work too.
        dpr = torch.linspace(0, drop_path_rate, sum(depths), device="cpu").tolist()
        self.use_v2 = use_v2
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()
        if self.use_v2:
            self.layers1c = nn.ModuleList()
            self.layers2c = nn.ModuleList()
            self.layers3c = nn.ModuleList()
            self.layers4c = nn.ModuleList()
        self.stage_scales: List[Tuple[int, ...]] = []
        resolution = list(img_size) if img_size is not None else None
        for i_layer in range(self.num_layers):
            dim = int(embed_dim * 2**i_layer)
            if downsample == "ts_satfire":
                # TS-SatFire decides from the stage resolution derived from img_size (not divided by the patch).
                merge = [r % 2 == 0 for r in resolution]
                scale = tuple(2 if m else 1 for m in merge)
                resolution = [r // s for r, s in zip(resolution, scale)]

                def factory(dim=dim, merge=merge) -> nn.Module:
                    return AdaptivePatchMerging(dim, merge)

            else:
                scale = (2,) * spatial_dims
                merging = PatchMerging if downsample == "merging" else PatchMergingV2

                def factory(dim=dim, merging=merging) -> nn.Module:
                    return merging(dim=dim, spatial_dims=spatial_dims)

            self.stage_scales.append(scale)
            layer = BasicLayer(
                dim=dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                downsample=factory,
                use_checkpoint=use_checkpoint,
                attn_version=attn_version,
            )
            getattr(self, f"layers{i_layer + 1}").append(layer)
            if self.use_v2:
                layerc = UnetrBasicBlock(
                    spatial_dims=spatial_dims,
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=3,
                    stride=1,
                    norm_name="instance",
                    res_block=True,
                )
                getattr(self, f"layers{i_layer + 1}c").append(layerc)
        self.resamples = [list(s) for s in reversed(self.stage_scales)]
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

    @staticmethod
    def proj_out(x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        if normalize:
            x = _channels_first(F.layer_norm(_channels_last(x), [x.shape[1]]))
        return x

    def forward(self, x: torch.Tensor, normalize: bool = True) -> List[torch.Tensor]:
        x0 = self.pos_drop(self.patch_embed(x))
        outs = [self.proj_out(x0, normalize)]
        for i in range(1, 5):
            if self.use_v2:
                x0 = getattr(self, f"layers{i}c")[0](x0.contiguous())
            x0 = getattr(self, f"layers{i}")[0](x0.contiguous())
            outs.append(self.proj_out(x0, normalize))
        return outs


class SwinUNETR(nn.Module):
    """MONAI ``SwinUNETR`` (1.3.2) in plain PyTorch, with TS-SatFire's options.

    Args:
        img_size: spatial input size. Optional for MONAI's merging modes (only checked, as in
            MONAI); required for ``downsample="ts_satfire"``, whose merging plan depends on it.
        in_channels: input channels.
        out_channels: output channels (logits, no activation).
        depths: Swin blocks per stage (four stages).
        num_heads: attention heads per stage.
        feature_size: width of the first stage; doubled at every stage.
        norm_name: decoder normalisation, ``"instance"`` (MONAI) or ``"batch"`` (TS-SatFire).
        drop_rate, attn_drop_rate, dropout_path_rate: dropout, attention dropout, stochastic depth.
        normalize: layer-normalise the encoder features passed to the decoder.
        use_checkpoint: gradient checkpointing in the Swin blocks.
        spatial_dims: 2 or 3 (``"ts_satfire"`` merging: 3 only).
        downsample: ``"merging"`` (MONAI default, v0.9.0 slice order), ``"mergingv2"`` or
            ``"ts_satfire"`` (merge only even-sized dimensions).
        use_v2: MONAI's SwinUNETR-v2 residual convolution block before every stage.
        patch_size: patch size (MONAI: 2; TS-SatFire: ``(1, 2, 2)``).
        window_size: attention window (MONAI: 7; TS-SatFire: ``(T, 4, 4)``).
        attn_version: ``"v1"`` (MONAI window attention) or ``"v2"`` (Swin-V2 cosine attention).
        wrap_out: wrap the output block in ``nn.Sequential`` (TS-SatFire's parameter names).

    Input ``(batch, in_channels, *spatial)``; output ``(batch, out_channels, *spatial)``.
    """

    def __init__(
        self,
        img_size: Optional[IntOrSeq],
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: NormSpec = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample: str = "merging",
        use_v2: bool = False,
        patch_size: IntOrSeq = 2,
        window_size: IntOrSeq = 7,
        attn_version: str = "v1",
        wrap_out: bool = False,
    ):
        super().__init__()
        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")
        if downsample not in DOWNSAMPLE_MODES:
            raise ValueError(f"downsample must be one of {DOWNSAMPLE_MODES}, got {downsample!r}")
        if attn_version not in ATTENTION_VERSIONS:
            raise ValueError(
                f"attn_version must be one of {ATTENTION_VERSIONS}, got {attn_version!r} (TS-SatFire's "
                "autoregressive 'ar' attention is not part of its prediction model and is not ported)"
            )
        if in_channels <= 0 or out_channels <= 0 or feature_size <= 0:
            raise ValueError("in_channels, out_channels and feature_size must be positive.")
        for name, rate in (("dropout", drop_rate), ("attention dropout", attn_drop_rate), ("drop path", dropout_path_rate)):
            if not 0 <= rate <= 1:
                raise ValueError(f"{name} rate should be between 0 and 1.")
        depths, num_heads = tuple(depths), tuple(num_heads)
        if len(depths) != 4 or len(num_heads) != 4 or min(depths) < 1:
            raise ValueError(f"depths and num_heads need four entries (depths >= 1), got {depths} and {num_heads}")
        if any((feature_size * 2**i) % h for i, h in enumerate(num_heads)):
            raise ValueError(f"Every stage width feature_size * 2**i must be divisible by num_heads[i], got {feature_size} and {num_heads}")
        if downsample != "ts_satfire" and feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")  # MONAI's check (dropped by TS-SatFire)
        check_norm_name(norm_name)
        patch = as_tuple(patch_size, spatial_dims, "patch_size")
        window = as_tuple(window_size, spatial_dims, "window_size")
        if min(patch) < 1 or min(window) < 1:
            raise ValueError(f"patch_size and window_size must be positive, got {patch} and {window}")
        size = as_tuple(img_size, spatial_dims, "img_size") if img_size is not None else None
        if downsample == "ts_satfire":
            if spatial_dims != 3:
                raise ValueError("downsample='ts_satfire' supports spatial_dims=3 only, as in TS-SatFire.")
            if size is None:
                raise ValueError("downsample='ts_satfire' needs img_size: its merging plan is fixed at construction.")
            for m, p in zip(size, patch):  # TS-SatFire's check
                if any(m % p ** (i + 1) for i in range(5)):
                    raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")
        elif size is not None:
            self._check_monai_size(size, patch)

        self.img_size = size
        self.in_channels = in_channels
        self.spatial_dims = spatial_dims
        self.downsample = downsample
        self.patch_size = patch
        self.window_size = window
        self.normalize = normalize

        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window,
            patch_size=patch,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=downsample,
            use_v2=use_v2,
            img_size=size,
            attn_version=attn_version,
        )
        resamples = self.swinViT.resamples

        def basic(in_mult: int) -> UnetrBasicBlock:
            return UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=in_mult * feature_size,
                out_channels=in_mult * feature_size,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=True,
            )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder2 = basic(1)
        self.encoder3 = basic(2)
        self.encoder4 = basic(4)
        self.encoder10 = basic(16)
        decoders = (
            ("decoder5", 16, 8, resamples[0]),
            ("decoder4", 8, 4, resamples[1]),
            ("decoder3", 4, 2, resamples[2]),
            ("decoder2", 2, 1, resamples[3]),
            ("decoder1", 1, 1, patch),
        )
        for name, in_mult, out_mult, up in decoders:
            setattr(
                self,
                name,
                UnetrUpBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_mult * feature_size,
                    out_channels=out_mult * feature_size,
                    kernel_size=3,
                    upsample_kernel_size=tuple(up),
                    norm_name=norm_name,
                    res_block=True,
                ),
            )
        out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.out = nn.Sequential(out) if wrap_out else out

    @staticmethod
    def _check_monai_size(spatial: Sequence[int], patch: Sequence[int]) -> None:
        # MONAI: divisible by patch_size**5 (= 2 * 2**4 with its fixed patch 2).
        multiple = tuple(p * 2**4 for p in patch)
        if any(s % m for s, m in zip(spatial, multiple)):
            raise ValueError(f"spatial dimensions {tuple(spatial)} of input image must be divisible by {multiple}.")

    def _check_input(self, x: torch.Tensor, layout: str) -> None:
        if x.ndim != self.spatial_dims + 2:
            raise ValueError(f"{type(self).__name__} expects input shape {layout}, got {tuple(x.shape)}.")
        if x.size(1) != self.in_channels:
            raise ValueError(f"{type(self).__name__} expected {self.in_channels} input channels, got shape {tuple(x.shape)}.")
        spatial = tuple(x.shape[2:])
        if self.downsample != "ts_satfire":
            self._check_monai_size(spatial, self.patch_size)
            return
        if any(s % p for s, p in zip(spatial, self.patch_size)):
            raise ValueError(f"Spatial shape {spatial} must be divisible by patch_size {self.patch_size}.")
        sizes = [s // p for s, p in zip(spatial, self.patch_size)]
        for stage, scale in enumerate(self.swinViT.stage_scales):
            if any(f == 2 and n % 2 for n, f in zip(sizes, scale)):
                raise ValueError(
                    f"Spatial shape {spatial} does not fit the merging plan {self.swinViT.stage_scales} fixed by "
                    f"img_size={self.img_size}: stage {stage + 1} would merge an odd-sized dimension."
                )
            sizes = [n // f for n, f in zip(sizes, scale)]

    def _forward_network(self, x_in: torch.Tensor) -> torch.Tensor:
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        return self.out(out)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        spatial = ", ".join(["D", "H", "W"][-self.spatial_dims :])
        self._check_input(x_in, f"(batch, channels, {spatial})")
        return self._forward_network(x_in)


class TemporalSwinUNETR(TemporalReadout, SwinUNETR):
    """3D :class:`SwinUNETR` over ``(time, H, W)`` for raster time series (TS-SatFire's SwinUNETR-3D).

    Takes ``(batch, time, channels, H, W)``, runs the network on ``(batch, channels, time, H, W)``
    and, with ``time_reduction="mean"``, averages the logits over time to
    ``(batch, out_channels, H, W)``. Parameter names are those of :class:`SwinUNETR`.
    """

    def __init__(self, *args, time_reduction: str = "mean", **kwargs):
        super().__init__(*args, **kwargs)
        if self.spatial_dims != 3:
            raise ValueError(f"TemporalSwinUNETR needs spatial_dims=3, got {self.spatial_dims}")
        self._set_time_reduction(time_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_network_layout(x, type(self).__name__)
        self._check_input(x, "(batch, time, channels, height, width)")
        return self._read_out(self._forward_network(x))


def swin_unetr_builder(
    task: str,
    in_channels: int,
    out_channels: int = 2,
    spatial_dims: int = 3,
    img_size: Optional[IntOrSeq] = None,
    history: int = 6,
    image_size: int = 256,
    feature_size: Optional[int] = None,
    num_heads: Optional[Union[int, Sequence[int]]] = None,
    depths: Sequence[int] = (2, 2, 2, 2),
    norm_name: NormSpec = "batch",
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    dropout_path_rate: float = 0.0,
    normalize: bool = True,
    use_checkpoint: bool = False,
    patch_size: Optional[IntOrSeq] = None,
    window_size: Optional[IntOrSeq] = None,
    attn_version: str = "v1",
    downsample: Optional[str] = None,
    use_v2: bool = False,
    time_reduction: str = "mean",
    **kwargs,
) -> nn.Module:
    """Build MONAI's SwinUNETR; the defaults are TS-SatFire's SwinUNETR models.

    ``spatial_dims=3`` returns TS-SatFire's SwinUNETR-3D as a :class:`TemporalSwinUNETR` (input
    ``(B, T, C, H, W)``): ``img_size = (history, image_size, image_size)``, patch ``(1, 2, 2)``,
    window ``(history, 4, 4)``, TS-SatFire's merging and output naming, feature size 36 and
    3 heads per stage. ``spatial_dims=2`` returns TS-SatFire's SwinUNETR-2D, stock MONAI
    (patch 2, window 7, MONAI merging, feature size 48, heads (3, 6, 12, 24)), input
    ``(B, C, H, W)``. ``num_heads`` may be one int for every stage, as in TS-SatFire's scripts.
    """
    _ = kwargs
    if task.lower() != "segmentation":
        raise ValueError(f"swin_unetr supports task='segmentation', got {task!r}.")
    if spatial_dims not in (2, 3):
        raise ValueError(f"swin_unetr supports spatial_dims 2 or 3, got {spatial_dims}.")
    ts_3d = spatial_dims == 3
    if num_heads is None:
        num_heads = 3 if ts_3d else (3, 6, 12, 24)
    heads = (int(num_heads),) * 4 if isinstance(num_heads, int) else tuple(num_heads)
    if img_size is None:
        img_size = (history, image_size, image_size) if ts_3d else (image_size, image_size)
    common = dict(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        depths=depths,
        num_heads=heads,
        feature_size=(36 if ts_3d else 48) if feature_size is None else feature_size,
        norm_name=norm_name,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        dropout_path_rate=dropout_path_rate,
        normalize=normalize,
        use_checkpoint=use_checkpoint,
        spatial_dims=spatial_dims,
        use_v2=use_v2,
        attn_version=attn_version,
    )
    if ts_3d:
        img = as_tuple(img_size, 3, "img_size")
        return TemporalSwinUNETR(
            downsample="ts_satfire" if downsample is None else downsample,
            patch_size=TS_SATFIRE_SWIN3D_PATCH if patch_size is None else patch_size,
            window_size=(img[0], *TS_SATFIRE_SWIN3D_WINDOW_HW) if window_size is None else window_size,
            wrap_out=downsample in (None, "ts_satfire"),
            time_reduction=time_reduction,
            **common,
        )
    return SwinUNETR(
        downsample="merging" if downsample is None else downsample,
        patch_size=2 if patch_size is None else patch_size,
        window_size=7 if window_size is None else window_size,
        **common,
    )


__all__ = [
    "AdaptivePatchMerging",
    "BasicLayer",
    "PatchEmbed",
    "PatchMerging",
    "PatchMergingV2",
    "SwinTransformer",
    "SwinTransformerBlock",
    "SwinUNETR",
    "TemporalSwinUNETR",
    "WindowAttention",
    "WindowAttentionV2",
    "swin_unetr_builder",
]
