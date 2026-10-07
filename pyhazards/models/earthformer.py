"""Earthformer: a space-time Transformer built from cuboid attention.

Gao, Shi, Wang, Zhu, Wang, Li & Yeung, "Earthformer: Exploring Space-Time Transformers for Earth
System Forecasting", NeurIPS 2022 (https://arxiv.org/abs/2207.05833).

Port of the official code, amazon-science/earth-forecasting-transformer at commit
``7732b03bdb366110563516c3502315deab4c2026``:
``src/earthformer/cuboid_transformer/cuboid_transformer.py``, ``cuboid_transformer_patterns.py`` and
``utils.py``. Licensed under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0). NOTICE of the original work: "Copyright Amazon.com,
Inc. or its affiliates. All Rights Reserved."

Changes made for PyHazards (the network itself is unchanged):

- the cuboid patterns are kept in plain dictionaries instead of the GluonNLP ``Registry`` class,
  the unused ``einops`` import is dropped, and ``DownSampling3D`` (not used by
  ``CuboidTransformerModel``) is not ported;
- ``torch.meshgrid`` gets ``indexing="ij"`` and gradient checkpointing ``use_reentrant=True``
  explicitly (the defaults of the PyTorch versions the official code was written for; current
  PyTorch warns when they are omitted), and cuboid strategies are stored as tuples so that the
  cached mask functions also accept lists;
- three code paths that only raise errors in the official code work here: ``PatchMerging3D`` pads
  when only the time axis needs padding (official: ``if pad_h or pad_h or pad_w``), the
  ``CTHW`` layout of ``Upsample3DLayer`` (official: ``self.output_size``), and cross-attention
  masks for ``padding_type`` "zeros"/"nearest" when padding is needed (official: undefined mask);
- ``CuboidTransformerModel.forward`` checks the input shape and raises ``ValueError``;
- :class:`Earthformer` / :class:`EarthformerSegmenter` (PyHazards wrappers: ``(batch, time,
  channels, height, width)`` layout, next-fire-mask head), the configuration presets and
  :func:`earthformer_builder` are additions.

Parameter names, creation order and initialisation follow ``CuboidTransformerModel``, so its state
dicts (including the official SEVIR and ICAR-ENSO checkpoints) load with ``strict=True`` and the
same seed gives the same initial weights.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import urllib.error
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


# --------------------------------------------------------------------------------------------------
# utils.py
# --------------------------------------------------------------------------------------------------


def round_to(dat, c):
    return dat + (dat - dat % c) % c


def get_activation(act, inplace=False, **kwargs):
    """Activation layer by name ('leaky' is LeakyReLU with slope 0.1)."""
    if act is None:
        return lambda x: x
    if isinstance(act, str):
        if act == "leaky":
            negative_slope = kwargs.get("negative_slope", 0.1)
            return nn.LeakyReLU(negative_slope, inplace=inplace)
        elif act == "identity":
            return nn.Identity()
        elif act == "elu":
            return nn.ELU(inplace=inplace)
        elif act == "gelu":
            return nn.GELU()
        elif act == "relu":
            return nn.ReLU()
        elif act == "sigmoid":
            return nn.Sigmoid()
        elif act == "tanh":
            return nn.Tanh()
        elif act == "softrelu" or act == "softplus":
            return nn.Softplus()
        elif act == "softsign":
            return nn.Softsign()
        else:
            raise NotImplementedError(f'act="{act}" is not supported.')
    else:
        return act


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, NeurIPS 2019)."""

    def __init__(self, d, p=-1.0, eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)
            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset
        return self.scale * x_normed


def get_norm_layer(normalization: str = "layer_norm", axis: int = -1, epsilon: float = 1e-5, in_channels: int = 0, **kwargs):
    """Normalization layer over the last axis: 'layer_norm', 'rms_norm' or None (identity)."""
    if isinstance(normalization, str):
        if normalization == "layer_norm":
            assert in_channels > 0
            assert axis == -1
            norm_layer = nn.LayerNorm(normalized_shape=in_channels, eps=epsilon, **kwargs)
        elif normalization == "rms_norm":
            assert axis == -1
            norm_layer = RMSNorm(d=in_channels, eps=epsilon, **kwargs)
        else:
            raise NotImplementedError(f"normalization={normalization} is not supported")
        return norm_layer
    elif normalization is None:
        return nn.Identity()
    else:
        raise NotImplementedError("The type of normalization must be str")


def _generalize_padding(x, pad_t, pad_h, pad_w, padding_type, t_pad_left=False):
    """Pad ``x`` of shape (B, T, H, W, C) to (B, T + pad_t, H + pad_h, W + pad_w, C)."""
    if pad_t == 0 and pad_h == 0 and pad_w == 0:
        return x

    assert padding_type in ["zeros", "ignore", "nearest"]
    B, T, H, W, C = x.shape

    if padding_type == "nearest":
        return F.interpolate(x.permute(0, 4, 1, 2, 3), size=(T + pad_t, H + pad_h, W + pad_w)).permute(0, 2, 3, 4, 1)
    else:
        if t_pad_left:
            return F.pad(x, (0, 0, 0, pad_w, 0, pad_h, pad_t, 0))
        else:
            return F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))


def _generalize_unpadding(x, pad_t, pad_h, pad_w, padding_type):
    assert padding_type in ["zeros", "ignore", "nearest"]
    B, T, H, W, C = x.shape
    if pad_t == 0 and pad_h == 0 and pad_w == 0:
        return x

    if padding_type == "nearest":
        return F.interpolate(x.permute(0, 4, 1, 2, 3), size=(T - pad_t, H - pad_h, W - pad_w)).permute(0, 2, 3, 4, 1)
    else:
        return x[:, : (T - pad_t), : (H - pad_h), : (W - pad_w), :].contiguous()


def apply_initialization(m, linear_mode="0", conv_mode="0", norm_mode="0", embed_mode="0"):
    if isinstance(m, nn.Linear):
        if linear_mode in ("0",):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
        elif linear_mode in ("1",):
            nn.init.kaiming_normal_(m.weight, a=0.1, mode="fan_out", nonlinearity="leaky_relu")
        else:
            raise NotImplementedError
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        if conv_mode in ("0",):
            nn.init.kaiming_normal_(m.weight, a=0.1, mode="fan_out", nonlinearity="leaky_relu")
        else:
            raise NotImplementedError
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        if norm_mode in ("0",):
            if m.elementwise_affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        else:
            raise NotImplementedError
    elif isinstance(m, nn.GroupNorm):
        if norm_mode in ("0",):
            if m.affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        else:
            raise NotImplementedError
    elif isinstance(m, nn.Embedding):
        if embed_mode in ("0",):
            nn.init.trunc_normal_(m.weight.data, std=0.02)
        else:
            raise NotImplementedError
    else:
        pass


def _checkpoint(function, *args):
    # The official code calls checkpoint.checkpoint(function, *args), i.e. the reentrant variant.
    return checkpoint.checkpoint(function, *args, use_reentrant=True)


# --------------------------------------------------------------------------------------------------
# cuboid_transformer_patterns.py
# --------------------------------------------------------------------------------------------------


def full_attention(input_shape):
    T, H, W, _ = input_shape
    cuboid_size = [(T, H, W)]
    strategy = [("l", "l", "l")]
    shift_size = [(0, 0, 0)]
    return cuboid_size, strategy, shift_size


def self_axial(input_shape):
    """Axial attention (Ho et al., arXiv:1912.12180): (T, 1, 1) -> (1, H, 1) -> (1, 1, W)."""
    T, H, W, _ = input_shape
    cuboid_size = [(T, 1, 1), (1, H, 1), (1, 1, W)]
    strategy = [("l", "l", "l"), ("l", "l", "l"), ("l", "l", "l")]
    shift_size = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
    return cuboid_size, strategy, shift_size


def self_video_swin(input_shape, P=2, M=4):
    """Video Swin Transformer windows (Liu et al., arXiv:2106.13230)."""
    T, H, W, _ = input_shape
    P = min(P, T)
    M = min(M, H, W)
    cuboid_size = [(P, M, M), (P, M, M)]
    strategy = [("l", "l", "l"), ("l", "l", "l")]
    shift_size = [(0, 0, 0), (P // 2, M // 2, M // 2)]
    return cuboid_size, strategy, shift_size


def self_divided_space_time(input_shape):
    T, H, W, _ = input_shape
    cuboid_size = [(T, 1, 1), (1, H, W)]
    strategy = [("l", "l", "l"), ("l", "l", "l")]
    shift_size = [(0, 0, 0), (0, 0, 0)]
    return cuboid_size, strategy, shift_size


def self_spatial_lg_v1(input_shape, M=4):
    T, H, W, _ = input_shape
    if H <= M and W <= M:
        cuboid_size = [(T, 1, 1), (1, H, W)]
        strategy = [("l", "l", "l"), ("l", "l", "l")]
        shift_size = [(0, 0, 0), (0, 0, 0)]
    else:
        cuboid_size = [(T, 1, 1), (1, M, M), (1, M, M)]
        strategy = [("l", "l", "l"), ("l", "l", "l"), ("d", "d", "d")]
        shift_size = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
    return cuboid_size, strategy, shift_size


def self_axial_space_dilate_K(input_shape, K=2):
    T, H, W, _ = input_shape
    K = min(K, H, W)
    cuboid_size = [(T, 1, 1), (1, H // K, 1), (1, H // K, 1), (1, 1, W // K), (1, 1, W // K)]
    strategy = [("l", "l", "l"), ("d", "d", "d"), ("l", "l", "l"), ("d", "d", "d"), ("l", "l", "l")]
    shift_size = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]
    return cuboid_size, strategy, shift_size


def cross_KxK(mem_shape, K):
    T_mem, H, W, _ = mem_shape
    K = min(K, H, W)
    cuboid_hw = [(K, K)]
    shift_hw = [(0, 0)]
    strategy = [("l", "l", "l")]
    n_temporal = [1]
    return cuboid_hw, shift_hw, strategy, n_temporal


def cross_KxK_lg(mem_shape, K):
    T_mem, H, W, _ = mem_shape
    K = min(K, H, W)
    cuboid_hw = [(K, K), (K, K)]
    shift_hw = [(0, 0), (0, 0)]
    strategy = [("l", "l", "l"), ("d", "d", "d")]
    n_temporal = [1, 1]
    return cuboid_hw, shift_hw, strategy, n_temporal


def cross_KxK_heter(mem_shape, K):
    T_mem, H, W, _ = mem_shape
    K = min(K, H, W)
    cuboid_hw = [(K, K), (K, K), (K, K)]
    shift_hw = [(0, 0), (0, 0), (K // 2, K // 2)]
    strategy = [("l", "l", "l"), ("d", "d", "d"), ("l", "l", "l")]
    n_temporal = [1, 1, 1]
    return cuboid_hw, shift_hw, strategy, n_temporal


CuboidSelfAttentionPatterns: Dict[str, Any] = {
    "full": full_attention,
    "axial": self_axial,
    "video_swin": self_video_swin,
    "divided_st": self_divided_space_time,
}
for _p in [1, 2, 4, 8, 10]:
    for _m in [1, 2, 4, 8, 16, 32]:
        CuboidSelfAttentionPatterns[f"video_swin_{_p}x{_m}"] = functools.partial(self_video_swin, P=_p, M=_m)
CuboidSelfAttentionPatterns["spatial_lg_v1"] = self_spatial_lg_v1
for _m in [1, 2, 4, 8, 16, 32]:
    CuboidSelfAttentionPatterns[f"spatial_lg_{_m}"] = functools.partial(self_spatial_lg_v1, M=_m)
for _k in [2, 4, 8]:
    CuboidSelfAttentionPatterns[f"axial_space_dilate_{_k}"] = functools.partial(self_axial_space_dilate_K, K=_k)

CuboidCrossAttentionPatterns: Dict[str, Any] = {}
for _k in [1, 2, 4, 8]:
    CuboidCrossAttentionPatterns[f"cross_{_k}x{_k}"] = functools.partial(cross_KxK, K=_k)
    CuboidCrossAttentionPatterns[f"cross_{_k}x{_k}_lg"] = functools.partial(cross_KxK_lg, K=_k)
    CuboidCrossAttentionPatterns[f"cross_{_k}x{_k}_heter"] = functools.partial(cross_KxK_heter, K=_k)


def _get_pattern(patterns: Mapping[str, Any], key: str, kind: str):
    if key not in patterns:
        raise ValueError(f"Unknown cuboid {kind} pattern {key!r}; expected one of {sorted(patterns)}.")
    return patterns[key]


# --------------------------------------------------------------------------------------------------
# cuboid_transformer.py
# --------------------------------------------------------------------------------------------------


class PosEmbed(nn.Module):
    """Learned space-time positional embedding added to (B, T, H, W, C): 't+h+w' or 't+hw'."""

    def __init__(self, embed_dim, maxT, maxH, maxW, typ="t+h+w"):
        super(PosEmbed, self).__init__()
        self.typ = typ

        assert self.typ in ["t+h+w", "t+hw"]
        self.maxT = maxT
        self.maxH = maxH
        self.maxW = maxW
        self.embed_dim = embed_dim
        if self.typ == "t+h+w":
            self.T_embed = nn.Embedding(num_embeddings=maxT, embedding_dim=embed_dim)
            self.H_embed = nn.Embedding(num_embeddings=maxH, embedding_dim=embed_dim)
            self.W_embed = nn.Embedding(num_embeddings=maxW, embedding_dim=embed_dim)
        elif self.typ == "t+hw":
            self.T_embed = nn.Embedding(num_embeddings=maxT, embedding_dim=embed_dim)
            self.HW_embed = nn.Embedding(num_embeddings=maxH * maxW, embedding_dim=embed_dim)
        else:
            raise NotImplementedError
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            apply_initialization(m, embed_mode="0")

    def forward(self, x):
        _, T, H, W, _ = x.shape
        t_idx = torch.arange(T, device=x.device)
        h_idx = torch.arange(H, device=x.device)
        w_idx = torch.arange(W, device=x.device)
        if self.typ == "t+h+w":
            return (
                x
                + self.T_embed(t_idx).reshape(T, 1, 1, self.embed_dim)
                + self.H_embed(h_idx).reshape(1, H, 1, self.embed_dim)
                + self.W_embed(w_idx).reshape(1, 1, W, self.embed_dim)
            )
        elif self.typ == "t+hw":
            spatial_idx = h_idx.unsqueeze(-1) * self.maxW + w_idx
            return x + self.T_embed(t_idx).reshape(T, 1, 1, self.embed_dim) + self.HW_embed(spatial_idx)
        else:
            raise NotImplementedError


class PositionwiseFFN(nn.Module):
    """Position-wise FFN.

    pre_norm=True:  norm(data) -> fc1 -> act -> act_dropout -> fc2 -> dropout -> res(+data)
    pre_norm=False: data -> fc1 -> act -> act_dropout -> fc2 -> dropout -> norm(res(+data))
    gated_proj uses act(fc1_gate(data)) * fc1(data).
    """

    def __init__(
        self,
        units: int = 512,
        hidden_size: int = 2048,
        activation_dropout: float = 0.0,
        dropout: float = 0.1,
        gated_proj: bool = False,
        activation="relu",
        normalization: str = "layer_norm",
        layer_norm_eps: float = 1e-5,
        pre_norm: bool = False,
        linear_init_mode="0",
        norm_init_mode="0",
    ):
        super().__init__()
        self.linear_init_mode = linear_init_mode
        self.norm_init_mode = norm_init_mode

        self._pre_norm = pre_norm
        self._gated_proj = gated_proj
        self._kwargs = OrderedDict(
            [
                ("units", units),
                ("hidden_size", hidden_size),
                ("activation_dropout", activation_dropout),
                ("activation", activation),
                ("dropout", dropout),
                ("normalization", normalization),
                ("layer_norm_eps", layer_norm_eps),
                ("gated_proj", gated_proj),
                ("pre_norm", pre_norm),
            ]
        )
        self.dropout_layer = nn.Dropout(dropout)
        self.activation_dropout_layer = nn.Dropout(activation_dropout)
        self.ffn_1 = nn.Linear(in_features=units, out_features=hidden_size, bias=True)
        if self._gated_proj:
            self.ffn_1_gate = nn.Linear(in_features=units, out_features=hidden_size, bias=True)
        self.activation = get_activation(activation)
        self.ffn_2 = nn.Linear(in_features=hidden_size, out_features=units, bias=True)
        self.layer_norm = get_norm_layer(normalization=normalization, in_channels=units, epsilon=layer_norm_eps)
        self.reset_parameters()

    def reset_parameters(self):
        apply_initialization(self.ffn_1, linear_mode=self.linear_init_mode)
        if self._gated_proj:
            apply_initialization(self.ffn_1_gate, linear_mode=self.linear_init_mode)
        apply_initialization(self.ffn_2, linear_mode=self.linear_init_mode)
        apply_initialization(self.layer_norm, norm_mode=self.norm_init_mode)

    def forward(self, data):
        residual = data
        if self._pre_norm:
            data = self.layer_norm(data)
        if self._gated_proj:
            out = self.activation(self.ffn_1_gate(data)) * self.ffn_1(data)
        else:
            out = self.activation(self.ffn_1(data))
        out = self.activation_dropout_layer(out)
        out = self.ffn_2(out)
        out = self.dropout_layer(out)
        out = out + residual
        if not self._pre_norm:
            out = self.layer_norm(out)
        return out


class PatchMerging3D(nn.Module):
    """Patch merging: (B, T, H, W, C) -> (B, T/dt, H/dh, W/dw, out_dim) via norm + linear."""

    def __init__(
        self,
        dim,
        out_dim=None,
        downsample=(1, 2, 2),
        norm_layer="layer_norm",
        padding_type="nearest",
        linear_init_mode="0",
        norm_init_mode="0",
    ):
        super().__init__()
        self.linear_init_mode = linear_init_mode
        self.norm_init_mode = norm_init_mode
        self.dim = dim
        if out_dim is None:
            out_dim = max(downsample) * dim
        self.out_dim = out_dim
        self.downsample = downsample
        self.padding_type = padding_type
        self.reduction = nn.Linear(downsample[0] * downsample[1] * downsample[2] * dim, out_dim, bias=False)
        self.norm = get_norm_layer(norm_layer, in_channels=downsample[0] * downsample[1] * downsample[2] * dim)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            apply_initialization(m, linear_mode=self.linear_init_mode, norm_mode=self.norm_init_mode)

    def get_out_shape(self, data_shape):
        T, H, W, C_in = data_shape
        pad_t = (self.downsample[0] - T % self.downsample[0]) % self.downsample[0]
        pad_h = (self.downsample[1] - H % self.downsample[1]) % self.downsample[1]
        pad_w = (self.downsample[2] - W % self.downsample[2]) % self.downsample[2]
        return (
            (T + pad_t) // self.downsample[0],
            (H + pad_h) // self.downsample[1],
            (W + pad_w) // self.downsample[2],
            self.out_dim,
        )

    def forward(self, x):
        B, T, H, W, C = x.shape

        pad_t = (self.downsample[0] - T % self.downsample[0]) % self.downsample[0]
        pad_h = (self.downsample[1] - H % self.downsample[1]) % self.downsample[1]
        pad_w = (self.downsample[2] - W % self.downsample[2]) % self.downsample[2]
        # The official condition is ``pad_h or pad_h or pad_w``, which skips the padding (and then
        # fails in reshape) when only the time axis needs it; identical whenever pad_t == 0.
        if pad_t or pad_h or pad_w:
            T += pad_t
            H += pad_h
            W += pad_w
            x = _generalize_padding(x, pad_t, pad_h, pad_w, padding_type=self.padding_type)

        x = (
            x.reshape(
                (
                    B,
                    T // self.downsample[0],
                    self.downsample[0],
                    H // self.downsample[1],
                    self.downsample[1],
                    W // self.downsample[2],
                    self.downsample[2],
                    C,
                )
            )
            .permute(0, 1, 3, 5, 2, 4, 6, 7)
            .reshape(
                B,
                T // self.downsample[0],
                H // self.downsample[1],
                W // self.downsample[2],
                self.downsample[0] * self.downsample[1] * self.downsample[2] * C,
            )
        )
        x = self.norm(x)
        x = self.reduction(x)
        return x


class Upsample3DLayer(nn.Module):
    """Nearest-neighbour upsampling followed by a 3x3 convolution.

    temporal_upsample=False: interpolation-2d (nearest) -> conv3x3(dim, out_dim) per frame;
    otherwise interpolation-3d (nearest) -> conv (the official code uses a Conv2d here too).
    """

    def __init__(
        self,
        dim,
        out_dim,
        target_size,
        temporal_upsample=False,
        kernel_size=3,
        layout="THWC",
        conv_init_mode="0",
    ):
        super(Upsample3DLayer, self).__init__()
        self.conv_init_mode = conv_init_mode
        self.target_size = target_size
        self.out_dim = out_dim
        self.temporal_upsample = temporal_upsample
        if temporal_upsample:
            self.up = nn.Upsample(size=target_size, mode="nearest")
        else:
            self.up = nn.Upsample(size=(target_size[1], target_size[2]), mode="nearest")
        self.conv = nn.Conv2d(
            in_channels=dim,
            out_channels=out_dim,
            kernel_size=(kernel_size, kernel_size),
            padding=(kernel_size // 2, kernel_size // 2),
        )
        assert layout in ["THWC", "CTHW"]
        self.layout = layout

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            apply_initialization(m, conv_mode=self.conv_init_mode)

    def forward(self, x):
        if self.layout == "THWC":
            B, T, H, W, C = x.shape
            if self.temporal_upsample:
                x = x.permute(0, 4, 1, 2, 3)
                return self.conv(self.up(x)).permute(0, 2, 3, 4, 1)
            else:
                assert self.target_size[0] == T
                x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
                x = self.up(x)
                return self.conv(x).permute(0, 2, 3, 1).reshape((B,) + tuple(self.target_size) + (self.out_dim,))
        elif self.layout == "CTHW":
            B, C, T, H, W = x.shape
            if self.temporal_upsample:
                return self.conv(self.up(x))
            else:
                assert self.target_size[0] == T  # official: self.output_size, an AttributeError
                x = x.permute(0, 2, 1, 3, 4)
                x = x.reshape(B * T, C, H, W)
                return (
                    self.conv(self.up(x))
                    .reshape(B, self.target_size[0], self.out_dim, self.target_size[1], self.target_size[2])
                    .permute(0, 2, 1, 3, 4)
                )


def cuboid_reorder(data, cuboid_size, strategy):
    """Reorder (B, T, H, W, C) into (B, num_cuboids, bT * bH * bW, C).

    The tensor shape must be divisible by the cuboid size. Strategy 'l' groups neighbouring
    elements (local), 'd' strided ones (dilated).
    """
    B, T, H, W, C = data.shape
    num_cuboids = T // cuboid_size[0] * H // cuboid_size[1] * W // cuboid_size[2]
    cuboid_volume = cuboid_size[0] * cuboid_size[1] * cuboid_size[2]
    intermediate_shape = []

    nblock_axis = []
    block_axis = []
    for i, (block_size, total_size, ele_strategy) in enumerate(zip(cuboid_size, (T, H, W), strategy)):
        if ele_strategy == "l":
            intermediate_shape.extend([total_size // block_size, block_size])
            nblock_axis.append(2 * i + 1)
            block_axis.append(2 * i + 2)
        elif ele_strategy == "d":
            intermediate_shape.extend([block_size, total_size // block_size])
            nblock_axis.append(2 * i + 2)
            block_axis.append(2 * i + 1)
        else:
            raise NotImplementedError
    data = data.reshape((B,) + tuple(intermediate_shape) + (C,))
    reordered_data = data.permute((0,) + tuple(nblock_axis) + tuple(block_axis) + (7,))
    reordered_data = reordered_data.reshape((B, num_cuboids, cuboid_volume, C))
    return reordered_data


def cuboid_reorder_reverse(data, cuboid_size, strategy, orig_data_shape):
    """Inverse of :func:`cuboid_reorder`."""
    B, num_cuboids, cuboid_volume, C = data.shape
    T, H, W = orig_data_shape

    permutation_axis = [0]
    for i, (block_size, total_size, ele_strategy) in enumerate(zip(cuboid_size, (T, H, W), strategy)):
        if ele_strategy == "l":
            permutation_axis.append(i + 1)
            permutation_axis.append(i + 4)
        elif ele_strategy == "d":
            permutation_axis.append(i + 4)
            permutation_axis.append(i + 1)
        else:
            raise NotImplementedError
    permutation_axis.append(7)
    data = data.reshape(
        B, T // cuboid_size[0], H // cuboid_size[1], W // cuboid_size[2], cuboid_size[0], cuboid_size[1], cuboid_size[2], C
    )
    data = data.permute(permutation_axis)
    data = data.reshape((B, T, H, W, C))
    return data


@lru_cache()
def compute_cuboid_self_attention_mask(data_shape, cuboid_size, shift_size, strategy, padding_type, device):
    """Shifted-window attention mask of shape (num_cuboid, cuboid_vol, cuboid_vol).

    Padded positions are masked when padding_type is 'ignore'; the shift mask keeps shifted
    windows from attending across the wrap-around.
    """
    T, H, W = data_shape
    pad_t = (cuboid_size[0] - T % cuboid_size[0]) % cuboid_size[0]
    pad_h = (cuboid_size[1] - H % cuboid_size[1]) % cuboid_size[1]
    pad_w = (cuboid_size[2] - W % cuboid_size[2]) % cuboid_size[2]
    data_mask = None
    if pad_t > 0 or pad_h > 0 or pad_w > 0:
        if padding_type == "ignore":
            data_mask = torch.ones((1, T, H, W, 1), dtype=torch.bool, device=device)
            data_mask = F.pad(data_mask, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))
    else:
        data_mask = torch.ones((1, T + pad_t, H + pad_h, W + pad_w, 1), dtype=torch.bool, device=device)
    if any(i > 0 for i in shift_size):
        if padding_type == "ignore":
            data_mask = torch.roll(data_mask, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
    if padding_type == "ignore":
        data_mask = cuboid_reorder(data_mask, cuboid_size, strategy=strategy)
        data_mask = data_mask.squeeze(-1).squeeze(0)
    shift_mask = torch.zeros((1, T + pad_t, H + pad_h, W + pad_w, 1), device=device)
    cnt = 0
    for t in slice(-cuboid_size[0]), slice(-cuboid_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-cuboid_size[1]), slice(-cuboid_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-cuboid_size[2]), slice(-cuboid_size[2], -shift_size[2]), slice(-shift_size[2], None):
                shift_mask[:, t, h, w, :] = cnt
                cnt += 1
    shift_mask = cuboid_reorder(shift_mask, cuboid_size, strategy=strategy)
    shift_mask = shift_mask.squeeze(-1).squeeze(0)
    attn_mask = (shift_mask.unsqueeze(1) - shift_mask.unsqueeze(2)) == 0
    if padding_type == "ignore":
        attn_mask = data_mask.unsqueeze(1) * data_mask.unsqueeze(2) * attn_mask
    return attn_mask


def masked_softmax(att_score, mask, axis: int = -1):
    """Softmax that ignores masked elements (mask: 1 keep, 0 masked; broadcastable)."""
    if mask is not None:
        if att_score.dtype == torch.float16:
            att_score = att_score.masked_fill(torch.logical_not(mask), -1e4)
        else:
            att_score = att_score.masked_fill(torch.logical_not(mask), -1e18)
        att_weights = torch.softmax(att_score, dim=axis) * mask
    else:
        att_weights = torch.softmax(att_score, dim=axis)
    return att_weights


def update_cuboid_size_shift_size(data_shape, cuboid_size, shift_size, strategy):
    """Shrink the cuboid to the data where the data is smaller; no shift on dilated axes."""
    new_cuboid_size = list(cuboid_size)
    new_shift_size = list(shift_size)
    for i in range(len(data_shape)):
        if strategy[i] == "d":
            new_shift_size[i] = 0
        if data_shape[i] <= cuboid_size[i]:
            new_cuboid_size[i] = data_shape[i]
            new_shift_size[i] = 0
    return tuple(new_cuboid_size), tuple(new_shift_size)


class CuboidSelfAttentionLayer(nn.Module):
    """Cuboid self-attention: decompose (T, H, W) into cuboids and attend within each cuboid.

    Optional global vectors attend to the whole tensor and every cuboid attends to them.
    """

    def __init__(
        self,
        dim,
        num_heads,
        cuboid_size=(2, 7, 7),
        shift_size=(0, 0, 0),
        strategy=("l", "l", "l"),
        padding_type="ignore",
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_final_proj=True,
        norm_layer="layer_norm",
        use_global_vector=False,
        use_global_self_attn=False,
        separate_global_qkv=False,
        global_dim_ratio=1,
        checkpoint_level=True,
        use_relative_pos=True,
        attn_linear_init_mode="0",
        ffn_linear_init_mode="0",
        norm_init_mode="0",
    ):
        super(CuboidSelfAttentionLayer, self).__init__()
        self.attn_linear_init_mode = attn_linear_init_mode
        self.ffn_linear_init_mode = ffn_linear_init_mode
        self.norm_init_mode = norm_init_mode

        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.dim = dim
        self.cuboid_size = tuple(cuboid_size)
        self.shift_size = tuple(shift_size)
        self.strategy = tuple(strategy)
        self.padding_type = padding_type
        self.use_final_proj = use_final_proj
        self.use_relative_pos = use_relative_pos
        self.use_global_vector = use_global_vector
        self.use_global_self_attn = use_global_self_attn
        self.separate_global_qkv = separate_global_qkv
        if global_dim_ratio != 1:
            assert separate_global_qkv is True, "Setting global_dim_ratio != 1 requires separate_global_qkv == True."
        self.global_dim_ratio = global_dim_ratio

        assert self.padding_type in ["ignore", "zeros", "nearest"]
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        if use_relative_pos:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * cuboid_size[0] - 1) * (2 * cuboid_size[1] - 1) * (2 * cuboid_size[2] - 1), num_heads)
            )
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

            coords_t = torch.arange(self.cuboid_size[0])
            coords_h = torch.arange(self.cuboid_size[1])
            coords_w = torch.arange(self.cuboid_size[2])
            coords = torch.stack(torch.meshgrid(coords_t, coords_h, coords_w, indexing="ij"))

            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.cuboid_size[0] - 1
            relative_coords[:, :, 1] += self.cuboid_size[1] - 1
            relative_coords[:, :, 2] += self.cuboid_size[2] - 1

            relative_coords[:, :, 0] *= (2 * self.cuboid_size[1] - 1) * (2 * self.cuboid_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.cuboid_size[2] - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        if self.use_global_vector:
            if self.separate_global_qkv:
                self.l2g_q_net = nn.Linear(dim, dim, bias=qkv_bias)
                self.l2g_global_kv_net = nn.Linear(in_features=global_dim_ratio * dim, out_features=dim * 2, bias=qkv_bias)
                self.g2l_global_q_net = nn.Linear(in_features=global_dim_ratio * dim, out_features=dim, bias=qkv_bias)
                self.g2l_k_net = nn.Linear(in_features=dim, out_features=dim, bias=qkv_bias)
                self.g2l_v_net = nn.Linear(in_features=dim, out_features=global_dim_ratio * dim, bias=qkv_bias)
                if self.use_global_self_attn:
                    self.g2g_global_qkv_net = nn.Linear(
                        in_features=global_dim_ratio * dim, out_features=global_dim_ratio * dim * 3, bias=qkv_bias
                    )
            else:
                self.global_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.global_attn_drop = nn.Dropout(attn_drop)

        if use_final_proj:
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)

            if self.use_global_vector:
                self.global_proj = nn.Linear(in_features=global_dim_ratio * dim, out_features=global_dim_ratio * dim)

        self.norm = get_norm_layer(norm_layer, in_channels=dim)
        if self.use_global_vector:
            self.global_vec_norm = get_norm_layer(norm_layer, in_channels=global_dim_ratio * dim)

        self.checkpoint_level = checkpoint_level
        self.reset_parameters()

    def reset_parameters(self):
        apply_initialization(self.qkv, linear_mode=self.attn_linear_init_mode)
        if self.use_final_proj:
            apply_initialization(self.proj, linear_mode=self.ffn_linear_init_mode)
        apply_initialization(self.norm, norm_mode=self.norm_init_mode)
        if self.use_global_vector:
            if self.separate_global_qkv:
                apply_initialization(self.l2g_q_net, linear_mode=self.attn_linear_init_mode)
                apply_initialization(self.l2g_global_kv_net, linear_mode=self.attn_linear_init_mode)
                apply_initialization(self.g2l_global_q_net, linear_mode=self.attn_linear_init_mode)
                apply_initialization(self.g2l_k_net, linear_mode=self.attn_linear_init_mode)
                apply_initialization(self.g2l_v_net, linear_mode=self.attn_linear_init_mode)
                if self.use_global_self_attn:
                    apply_initialization(self.g2g_global_qkv_net, linear_mode=self.attn_linear_init_mode)
            else:
                apply_initialization(self.global_qkv, linear_mode=self.attn_linear_init_mode)
            apply_initialization(self.global_vec_norm, norm_mode=self.norm_init_mode)

    def forward(self, x, global_vectors=None):
        x = self.norm(x)

        B, T, H, W, C_in = x.shape
        assert C_in == self.dim
        if self.use_global_vector:
            _, num_global, _ = global_vectors.shape
            global_vectors = self.global_vec_norm(global_vectors)

        cuboid_size, shift_size = update_cuboid_size_shift_size((T, H, W), self.cuboid_size, self.shift_size, self.strategy)
        # Step-1: pad the input
        pad_t = (cuboid_size[0] - T % cuboid_size[0]) % cuboid_size[0]
        pad_h = (cuboid_size[1] - H % cuboid_size[1]) % cuboid_size[1]
        pad_w = (cuboid_size[2] - W % cuboid_size[2]) % cuboid_size[2]

        x = _generalize_padding(x, pad_t, pad_h, pad_w, self.padding_type)

        # Step-2: shift the tensor (shifted-window attention)
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x
        # Step-3: reorder into (B, num_cuboids, cuboid_volume, C)
        reordered_x = cuboid_reorder(shifted_x, cuboid_size=cuboid_size, strategy=self.strategy)
        _, num_cuboids, cuboid_volume, _ = reordered_x.shape
        # Step-4: self-attention, mask of shape (num_cuboids, cuboid_volume, cuboid_volume)
        attn_mask = compute_cuboid_self_attention_mask(
            (T, H, W), cuboid_size, shift_size=shift_size, strategy=self.strategy, padding_type=self.padding_type, device=x.device
        )
        head_C = C_in // self.num_heads
        qkv = (
            self.qkv(reordered_x)
            .reshape(B, num_cuboids, cuboid_volume, 3, self.num_heads, head_C)
            .permute(3, 0, 4, 1, 2, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn_score = q @ k.transpose(-2, -1)

        if self.use_relative_pos:
            # Sliced to the (possibly shrunk) cuboid volume exactly as in the official code.
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index[:cuboid_volume, :cuboid_volume].reshape(-1)
            ].reshape(cuboid_volume, cuboid_volume, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(1)
            attn_score = attn_score + relative_position_bias

        # Local-to-global attention
        if self.use_global_vector:
            global_head_C = self.global_dim_ratio * head_C
            if self.separate_global_qkv:
                l2g_q = (
                    self.l2g_q_net(reordered_x)
                    .reshape(B, num_cuboids, cuboid_volume, self.num_heads, head_C)
                    .permute(0, 3, 1, 2, 4)
                )
                l2g_q = l2g_q * self.scale
                l2g_global_kv = (
                    self.l2g_global_kv_net(global_vectors)
                    .reshape(B, 1, num_global, 2, self.num_heads, head_C)
                    .permute(3, 0, 4, 1, 2, 5)
                )
                l2g_global_k, l2g_global_v = l2g_global_kv[0], l2g_global_kv[1]
                g2l_global_q = (
                    self.g2l_global_q_net(global_vectors).reshape(B, num_global, self.num_heads, head_C).permute(0, 2, 1, 3)
                )
                g2l_global_q = g2l_global_q * self.scale
                g2l_k = (
                    self.g2l_k_net(reordered_x)
                    .reshape(B, num_cuboids, cuboid_volume, self.num_heads, head_C)
                    .permute(0, 3, 1, 2, 4)
                )
                g2l_v = (
                    self.g2l_v_net(reordered_x)
                    .reshape(B, num_cuboids, cuboid_volume, self.num_heads, global_head_C)
                    .permute(0, 3, 1, 2, 4)
                )
                if self.use_global_self_attn:
                    g2g_global_qkv = (
                        self.g2g_global_qkv_net(global_vectors)
                        .reshape(B, 1, num_global, 3, self.num_heads, global_head_C)
                        .permute(3, 0, 4, 1, 2, 5)
                    )
                    g2g_global_q, g2g_global_k, g2g_global_v = g2g_global_qkv[0], g2g_global_qkv[1], g2g_global_qkv[2]
                    g2g_global_q = g2g_global_q.squeeze(2) * self.scale
            else:
                q_global, k_global, v_global = (
                    self.global_qkv(global_vectors)
                    .reshape(B, 1, num_global, 3, self.num_heads, head_C)
                    .permute(3, 0, 4, 1, 2, 5)
                )
                q_global = q_global.squeeze(2) * self.scale
                l2g_q, g2l_k, g2l_v = q, k, v
                g2l_global_q, l2g_global_k, l2g_global_v = q_global, k_global, v_global
                if self.use_global_self_attn:
                    g2g_global_q, g2g_global_k, g2g_global_v = q_global, k_global, v_global
            l2g_attn_score = l2g_q @ l2g_global_k.transpose(-2, -1)
            attn_score_l2l_l2g = torch.cat((attn_score, l2g_attn_score), dim=-1)
            attn_mask_l2l_l2g = F.pad(attn_mask, (0, num_global), "constant", 1)
            v_l_g = torch.cat((v, l2g_global_v.expand(B, self.num_heads, num_cuboids, num_global, head_C)), dim=3)
            attn_score_l2l_l2g = masked_softmax(attn_score_l2l_l2g, mask=attn_mask_l2l_l2g)
            attn_score_l2l_l2g = self.attn_drop(attn_score_l2l_l2g)
            reordered_x = (attn_score_l2l_l2g @ v_l_g).permute(0, 2, 3, 1, 4).reshape(B, num_cuboids, cuboid_volume, self.dim)
            # Update the global vectors
            if self.padding_type == "ignore":
                g2l_attn_mask = torch.ones((1, T, H, W, 1), device=x.device)
                if pad_t > 0 or pad_h > 0 or pad_w > 0:
                    g2l_attn_mask = F.pad(g2l_attn_mask, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))
                if any(i > 0 for i in shift_size):
                    g2l_attn_mask = torch.roll(
                        g2l_attn_mask, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3)
                    )
                g2l_attn_mask = g2l_attn_mask.reshape((-1,))
            else:
                g2l_attn_mask = None
            g2l_attn_score = g2l_global_q @ g2l_k.reshape(B, self.num_heads, num_cuboids * cuboid_volume, head_C).transpose(-2, -1)
            if self.use_global_self_attn:
                g2g_attn_score = g2g_global_q @ g2g_global_k.squeeze(2).transpose(-2, -1)
                g2all_attn_score = torch.cat((g2l_attn_score, g2g_attn_score), dim=-1)
                if g2l_attn_mask is not None:
                    g2all_attn_mask = F.pad(g2l_attn_mask, (0, num_global), "constant", 1)
                else:
                    g2all_attn_mask = None
                new_v = torch.cat(
                    (
                        g2l_v.reshape(B, self.num_heads, num_cuboids * cuboid_volume, global_head_C),
                        g2g_global_v.reshape(B, self.num_heads, num_global, global_head_C),
                    ),
                    dim=2,
                )
            else:
                g2all_attn_score = g2l_attn_score
                g2all_attn_mask = g2l_attn_mask
                new_v = g2l_v.reshape(B, self.num_heads, num_cuboids * cuboid_volume, global_head_C)
            g2all_attn_score = masked_softmax(g2all_attn_score, mask=g2all_attn_mask)
            g2all_attn_score = self.global_attn_drop(g2all_attn_score)
            new_global_vector = (
                (g2all_attn_score @ new_v).permute(0, 2, 1, 3).reshape(B, num_global, self.global_dim_ratio * self.dim)
            )
        else:
            attn_score = masked_softmax(attn_score, mask=attn_mask)
            attn_score = self.attn_drop(attn_score)
            reordered_x = (attn_score @ v).permute(0, 2, 3, 1, 4).reshape(B, num_cuboids, cuboid_volume, self.dim)

        if self.use_final_proj:
            reordered_x = self.proj_drop(self.proj(reordered_x))
            if self.use_global_vector:
                new_global_vector = self.proj_drop(self.global_proj(new_global_vector))
        # Step-5: shift back and slice
        shifted_x = cuboid_reorder_reverse(
            reordered_x, cuboid_size=cuboid_size, strategy=self.strategy, orig_data_shape=(T + pad_t, H + pad_h, W + pad_w)
        )
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x
        x = _generalize_unpadding(x, pad_t=pad_t, pad_h=pad_h, pad_w=pad_w, padding_type=self.padding_type)
        if self.use_global_vector:
            return x, new_global_vector
        else:
            return x


class StackCuboidSelfAttentionBlock(nn.Module):
    """A stack of cuboid self-attention layers (one per pattern), each followed by an FFN when
    ``use_inter_ffn`` (pre-LN residual), else one FFN after the stack."""

    def __init__(
        self,
        dim,
        num_heads,
        block_cuboid_size=[(4, 4, 4), (4, 4, 4)],
        block_shift_size=[(0, 0, 0), (2, 2, 2)],
        block_strategy=[("d", "d", "d"), ("l", "l", "l")],
        padding_type="ignore",
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        ffn_drop=0.0,
        activation="leaky",
        gated_ffn=False,
        norm_layer="layer_norm",
        use_inter_ffn=False,
        use_global_vector=False,
        use_global_vector_ffn=True,
        use_global_self_attn=False,
        separate_global_qkv=False,
        global_dim_ratio=1,
        checkpoint_level=True,
        use_relative_pos=True,
        use_final_proj=True,
        attn_linear_init_mode="0",
        ffn_linear_init_mode="0",
        norm_init_mode="0",
    ):
        super(StackCuboidSelfAttentionBlock, self).__init__()
        self.attn_linear_init_mode = attn_linear_init_mode
        self.ffn_linear_init_mode = ffn_linear_init_mode
        self.norm_init_mode = norm_init_mode

        assert len(block_cuboid_size[0]) > 0 and len(block_shift_size) > 0 and len(block_strategy) > 0, (
            f"Format of the block cuboid size is not correct. block_cuboid_size={block_cuboid_size}"
        )
        assert len(block_cuboid_size) == len(block_shift_size) == len(block_strategy)
        self.num_attn = len(block_cuboid_size)
        self.checkpoint_level = checkpoint_level
        self.use_inter_ffn = use_inter_ffn
        self.use_global_vector = use_global_vector
        self.use_global_vector_ffn = use_global_vector_ffn
        self.use_global_self_attn = use_global_self_attn
        self.global_dim_ratio = global_dim_ratio

        def ffn(units, hidden_size):
            return PositionwiseFFN(
                units=units,
                hidden_size=hidden_size,
                activation_dropout=ffn_drop,
                dropout=ffn_drop,
                gated_proj=gated_ffn,
                activation=activation,
                normalization=norm_layer,
                pre_norm=True,
                linear_init_mode=ffn_linear_init_mode,
                norm_init_mode=norm_init_mode,
            )

        num_ffn = self.num_attn if self.use_inter_ffn else 1
        self.ffn_l = nn.ModuleList([ffn(dim, 4 * dim) for _ in range(num_ffn)])
        if self.use_global_vector_ffn and self.use_global_vector:
            self.global_ffn_l = nn.ModuleList(
                [ffn(global_dim_ratio * dim, global_dim_ratio * 4 * dim) for _ in range(num_ffn)]
            )
        self.attn_l = nn.ModuleList(
            [
                CuboidSelfAttentionLayer(
                    dim=dim,
                    num_heads=num_heads,
                    cuboid_size=ele_cuboid_size,
                    shift_size=ele_shift_size,
                    strategy=ele_strategy,
                    padding_type=padding_type,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    norm_layer=norm_layer,
                    use_global_vector=use_global_vector,
                    use_global_self_attn=use_global_self_attn,
                    separate_global_qkv=separate_global_qkv,
                    global_dim_ratio=global_dim_ratio,
                    checkpoint_level=checkpoint_level,
                    use_relative_pos=use_relative_pos,
                    use_final_proj=use_final_proj,
                    attn_linear_init_mode=attn_linear_init_mode,
                    ffn_linear_init_mode=ffn_linear_init_mode,
                    norm_init_mode=norm_init_mode,
                )
                for ele_cuboid_size, ele_shift_size, ele_strategy in zip(block_cuboid_size, block_shift_size, block_strategy)
            ]
        )

    def reset_parameters(self):
        for m in self.ffn_l:
            m.reset_parameters()
        if self.use_global_vector_ffn and self.use_global_vector:
            for m in self.global_ffn_l:
                m.reset_parameters()
        for m in self.attn_l:
            m.reset_parameters()

    def forward(self, x, global_vectors=None):
        if self.use_inter_ffn:
            if self.use_global_vector:
                for idx, (attn, ffn) in enumerate(zip(self.attn_l, self.ffn_l)):
                    if self.checkpoint_level >= 2 and self.training:
                        x_out, global_vectors_out = _checkpoint(attn, x, global_vectors)
                    else:
                        x_out, global_vectors_out = attn(x, global_vectors)
                    x = x + x_out
                    global_vectors = global_vectors + global_vectors_out

                    if self.checkpoint_level >= 1 and self.training:
                        x = _checkpoint(ffn, x)
                        if self.use_global_vector_ffn:
                            global_vectors = _checkpoint(self.global_ffn_l[idx], global_vectors)
                    else:
                        x = ffn(x)
                        if self.use_global_vector_ffn:
                            global_vectors = self.global_ffn_l[idx](global_vectors)
                return x, global_vectors
            else:
                for idx, (attn, ffn) in enumerate(zip(self.attn_l, self.ffn_l)):
                    if self.checkpoint_level >= 2 and self.training:
                        x = x + _checkpoint(attn, x)
                    else:
                        x = x + attn(x)
                    if self.checkpoint_level >= 1 and self.training:
                        x = _checkpoint(ffn, x)
                    else:
                        x = ffn(x)
                return x
        else:
            if self.use_global_vector:
                for idx, attn in enumerate(self.attn_l):
                    if self.checkpoint_level >= 2 and self.training:
                        x_out, global_vectors_out = _checkpoint(attn, x, global_vectors)
                    else:
                        x_out, global_vectors_out = attn(x, global_vectors)
                    x = x + x_out
                    global_vectors = global_vectors + global_vectors_out
                if self.checkpoint_level >= 1 and self.training:
                    x = _checkpoint(self.ffn_l[0], x)
                    if self.use_global_vector_ffn:
                        global_vectors = _checkpoint(self.global_ffn_l[0], global_vectors)
                else:
                    x = self.ffn_l[0](x)
                    if self.use_global_vector_ffn:
                        global_vectors = self.global_ffn_l[0](global_vectors)
                return x, global_vectors
            else:
                for idx, attn in enumerate(self.attn_l):
                    if self.checkpoint_level >= 2 and self.training:
                        out = _checkpoint(attn, x)
                    else:
                        out = attn(x)
                    x = x + out
                if self.checkpoint_level >= 1 and self.training:
                    x = _checkpoint(self.ffn_l[0], x)
                else:
                    x = self.ffn_l[0](x)
                return x


@lru_cache()
def compute_cuboid_cross_attention_mask(T_x, T_mem, H, W, n_temporal, cuboid_hw, shift_hw, strategy, padding_type, device):
    """Cross-attention mask of shape (num_cuboid, x_cuboid_vol, mem_cuboid_vol)."""
    pad_t_mem = (n_temporal - T_mem % n_temporal) % n_temporal
    pad_t_x = (n_temporal - T_x % n_temporal) % n_temporal
    pad_h = (cuboid_hw[0] - H % cuboid_hw[0]) % cuboid_hw[0]
    pad_w = (cuboid_hw[1] - W % cuboid_hw[1]) % cuboid_hw[1]

    mem_cuboid_size = ((T_mem + pad_t_mem) // n_temporal,) + cuboid_hw
    x_cuboid_size = ((T_x + pad_t_x) // n_temporal,) + cuboid_hw
    # The official code leaves mem_mask / x_mask undefined when padding is needed and padding_type
    # is not 'ignore'; here the padded entries then count as tokens, as in the self-attention mask.
    if (pad_t_mem > 0 or pad_h > 0 or pad_w > 0) and padding_type == "ignore":
        mem_mask = torch.ones((1, T_mem, H, W, 1), dtype=torch.bool, device=device)
        mem_mask = F.pad(mem_mask, (0, 0, 0, pad_w, 0, pad_h, pad_t_mem, 0))
    else:
        mem_mask = torch.ones((1, T_mem + pad_t_mem, H + pad_h, W + pad_w, 1), dtype=torch.bool, device=device)
    if (pad_t_x > 0 or pad_h > 0 or pad_w > 0) and padding_type == "ignore":
        x_mask = torch.ones((1, T_x, H, W, 1), dtype=torch.bool, device=device)
        x_mask = F.pad(x_mask, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t_x))
    else:
        x_mask = torch.ones((1, T_x + pad_t_x, H + pad_h, W + pad_w, 1), dtype=torch.bool, device=device)

    if any(i > 0 for i in shift_hw):
        if padding_type == "ignore":
            x_mask = torch.roll(x_mask, shifts=(-shift_hw[0], -shift_hw[1]), dims=(2, 3))
            mem_mask = torch.roll(mem_mask, shifts=(-shift_hw[0], -shift_hw[1]), dims=(2, 3))
    x_mask = cuboid_reorder(x_mask, x_cuboid_size, strategy=strategy)
    x_mask = x_mask.squeeze(-1).squeeze(0)
    num_cuboids, x_cuboid_volume = x_mask.shape
    mem_mask = cuboid_reorder(mem_mask, mem_cuboid_size, strategy=strategy)
    mem_mask = mem_mask.squeeze(-1).squeeze(0)
    _, mem_cuboid_volume = mem_mask.shape

    shift_mask = torch.zeros((1, n_temporal, H + pad_h, W + pad_w, 1), device=device)

    cnt = 0
    for h in slice(-cuboid_hw[0]), slice(-cuboid_hw[0], -shift_hw[0]), slice(-shift_hw[0], None):
        for w in slice(-cuboid_hw[1]), slice(-cuboid_hw[1], -shift_hw[1]), slice(-shift_hw[1], None):
            shift_mask[:, :, h, w, :] = cnt
            cnt += 1
    shift_mask = cuboid_reorder(shift_mask, (1,) + cuboid_hw, strategy=strategy)
    shift_mask = shift_mask.squeeze(-1).squeeze(0)
    shift_mask = (shift_mask.unsqueeze(1) - shift_mask.unsqueeze(2)) == 0
    bh_bw = cuboid_hw[0] * cuboid_hw[1]
    attn_mask = (
        shift_mask.reshape((num_cuboids, 1, bh_bw, 1, bh_bw))
        * x_mask.reshape((num_cuboids, -1, bh_bw, 1, 1))
        * mem_mask.reshape(num_cuboids, 1, 1, -1, bh_bw)
    )
    attn_mask = attn_mask.reshape(num_cuboids, x_cuboid_volume, mem_cuboid_volume)
    return attn_mask


class CuboidCrossAttentionLayer(nn.Module):
    """Cuboid cross-attention between a query tensor (T2, H, W, C) and a memory (T1, H, W, C).

    Both are cut into the same number of cuboids (``n_temporal`` along time, ``cuboid_hw`` in
    space); the memory is padded on the left in time, the query on the right.
    """

    def __init__(
        self,
        dim,
        num_heads,
        n_temporal=1,
        cuboid_hw=(7, 7),
        shift_hw=(0, 0),
        strategy=("d", "l", "l"),
        padding_type="ignore",
        cross_last_n_frames=None,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        max_temporal_relative=50,
        norm_layer="layer_norm",
        use_global_vector=True,
        separate_global_qkv=False,
        global_dim_ratio=1,
        checkpoint_level=1,
        use_relative_pos=True,
        attn_linear_init_mode="0",
        ffn_linear_init_mode="0",
        norm_init_mode="0",
    ):
        super(CuboidCrossAttentionLayer, self).__init__()
        self.attn_linear_init_mode = attn_linear_init_mode
        self.ffn_linear_init_mode = ffn_linear_init_mode
        self.norm_init_mode = norm_init_mode

        self.dim = dim
        self.num_heads = num_heads
        self.n_temporal = n_temporal
        assert n_temporal > 0
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        shift_hw = list(shift_hw)
        if strategy[1] == "d":
            shift_hw[0] = 0
        if strategy[2] == "d":
            shift_hw[1] = 0
        self.cuboid_hw = tuple(cuboid_hw)
        self.shift_hw = tuple(shift_hw)
        self.strategy = tuple(strategy)
        self.padding_type = padding_type
        self.max_temporal_relative = max_temporal_relative
        self.cross_last_n_frames = cross_last_n_frames
        self.use_relative_pos = use_relative_pos
        self.use_global_vector = use_global_vector
        self.separate_global_qkv = separate_global_qkv
        if global_dim_ratio != 1:
            assert separate_global_qkv is True, "Setting global_dim_ratio != 1 requires separate_global_qkv == True."
        self.global_dim_ratio = global_dim_ratio

        assert self.padding_type in ["ignore", "zeros", "nearest"]

        if use_relative_pos:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * max_temporal_relative - 1) * (2 * cuboid_hw[0] - 1) * (2 * cuboid_hw[1] - 1), num_heads)
            )
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

            coords_t = torch.arange(max_temporal_relative)
            coords_h = torch.arange(self.cuboid_hw[0])
            coords_w = torch.arange(self.cuboid_hw[1])
            coords = torch.stack(torch.meshgrid(coords_t, coords_h, coords_w, indexing="ij"))

            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += max_temporal_relative - 1
            relative_coords[:, :, 1] += self.cuboid_hw[0] - 1
            relative_coords[:, :, 2] += self.cuboid_hw[1] - 1
            relative_position_index = (
                relative_coords[:, :, 0] * (2 * self.cuboid_hw[0] - 1) * (2 * self.cuboid_hw[1] - 1)
                + relative_coords[:, :, 1] * (2 * self.cuboid_hw[1] - 1)
                + relative_coords[:, :, 2]
            )
            self.register_buffer("relative_position_index", relative_position_index)

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.use_global_vector:
            if self.separate_global_qkv:
                self.l2g_q_net = nn.Linear(dim, dim, bias=qkv_bias)
                self.l2g_global_kv_net = nn.Linear(in_features=global_dim_ratio * dim, out_features=dim * 2, bias=qkv_bias)

        self.norm = get_norm_layer(norm_layer, in_channels=dim)

        self._checkpoint_level = checkpoint_level

        self.reset_parameters()

    def reset_parameters(self):
        apply_initialization(self.q_proj, linear_mode=self.attn_linear_init_mode)
        apply_initialization(self.kv_proj, linear_mode=self.attn_linear_init_mode)
        apply_initialization(self.proj, linear_mode=self.ffn_linear_init_mode)
        apply_initialization(self.norm, norm_mode=self.norm_init_mode)
        if self.use_global_vector:
            if self.separate_global_qkv:
                apply_initialization(self.l2g_q_net, linear_mode=self.attn_linear_init_mode)
                apply_initialization(self.l2g_global_kv_net, linear_mode=self.attn_linear_init_mode)

    def forward(self, x, mem, mem_global_vectors=None):
        if self.cross_last_n_frames is not None:
            cross_last_n_frames = int(min(self.cross_last_n_frames, mem.shape[1]))
            mem = mem[:, -cross_last_n_frames:, ...]
        if self.use_global_vector:
            _, num_global, _ = mem_global_vectors.shape
        x = self.norm(x)
        B, T_x, H, W, C_in = x.shape
        B_mem, T_mem, H_mem, W_mem, C_mem = mem.shape
        assert T_x < self.max_temporal_relative and T_mem < self.max_temporal_relative
        cuboid_hw = self.cuboid_hw
        n_temporal = self.n_temporal
        shift_hw = self.shift_hw
        assert B_mem == B and H == H_mem and W == W_mem and C_in == C_mem, (
            f"Shape of memory and the input tensor does not match. x.shape={x.shape}, mem.shape={mem.shape}"
        )
        pad_t_mem = (n_temporal - T_mem % n_temporal) % n_temporal
        pad_t_x = (n_temporal - T_x % n_temporal) % n_temporal
        pad_h = (cuboid_hw[0] - H % cuboid_hw[0]) % cuboid_hw[0]
        pad_w = (cuboid_hw[1] - W % cuboid_hw[1]) % cuboid_hw[1]

        # Step-1: pad the memory and x
        mem = _generalize_padding(mem, pad_t_mem, pad_h, pad_w, self.padding_type, t_pad_left=True)
        x = _generalize_padding(x, pad_t_x, pad_h, pad_w, self.padding_type, t_pad_left=False)

        # Step-2: shift the tensors (shifted-window attention)
        if any(i > 0 for i in shift_hw):
            shifted_x = torch.roll(x, shifts=(-shift_hw[0], -shift_hw[1]), dims=(2, 3))
            shifted_mem = torch.roll(mem, shifts=(-shift_hw[0], -shift_hw[1]), dims=(2, 3))
        else:
            shifted_x = x
            shifted_mem = mem

        # Step-3: reorder the tensors
        mem_cuboid_size = (mem.shape[1] // n_temporal,) + cuboid_hw
        x_cuboid_size = (x.shape[1] // n_temporal,) + cuboid_hw

        reordered_mem = cuboid_reorder(shifted_mem, cuboid_size=mem_cuboid_size, strategy=self.strategy)
        reordered_x = cuboid_reorder(shifted_x, cuboid_size=x_cuboid_size, strategy=self.strategy)
        _, num_cuboids_mem, mem_cuboid_volume, _ = reordered_mem.shape
        _, num_cuboids, x_cuboid_volume, _ = reordered_x.shape
        assert num_cuboids_mem == num_cuboids, (
            f"Number of cuboids do not match. num_cuboids={num_cuboids}, num_cuboids_mem={num_cuboids_mem}"
        )

        # Step-4: cross-attention, mask of shape (num_cuboids, x_cuboid_volume, mem_cuboid_volume)
        attn_mask = compute_cuboid_cross_attention_mask(
            T_x, T_mem, H, W, n_temporal, cuboid_hw, shift_hw, strategy=self.strategy, padding_type=self.padding_type, device=x.device
        )
        head_C = C_in // self.num_heads

        kv = (
            self.kv_proj(reordered_mem)
            .reshape(B, num_cuboids, mem_cuboid_volume, 2, self.num_heads, head_C)
            .permute(3, 0, 4, 1, 2, 5)
        )
        k, v = kv[0], kv[1]
        q = self.q_proj(reordered_x).reshape(B, num_cuboids, x_cuboid_volume, self.num_heads, head_C).permute(0, 3, 1, 2, 4)
        q = q * self.scale
        attn_score = q @ k.transpose(-2, -1)

        if self.use_relative_pos:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index[:x_cuboid_volume, :mem_cuboid_volume].reshape(-1)
            ].reshape(x_cuboid_volume, mem_cuboid_volume, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(1)
            attn_score = attn_score + relative_position_bias

        if self.use_global_vector:
            if self.separate_global_qkv:
                l2g_q = (
                    self.l2g_q_net(reordered_x)
                    .reshape(B, num_cuboids, x_cuboid_volume, self.num_heads, head_C)
                    .permute(0, 3, 1, 2, 4)
                )
                l2g_q = l2g_q * self.scale
                l2g_global_kv = (
                    self.l2g_global_kv_net(mem_global_vectors)
                    .reshape(B, 1, num_global, 2, self.num_heads, head_C)
                    .permute(3, 0, 4, 1, 2, 5)
                )
                l2g_global_k, l2g_global_v = l2g_global_kv[0], l2g_global_kv[1]
            else:
                kv_global = (
                    self.kv_proj(mem_global_vectors).reshape(B, 1, num_global, 2, self.num_heads, head_C).permute(3, 0, 4, 1, 2, 5)
                )
                l2g_global_k, l2g_global_v = kv_global[0], kv_global[1]
                l2g_q = q
            l2g_attn_score = l2g_q @ l2g_global_k.transpose(-2, -1)
            attn_score_l2l_l2g = torch.cat((attn_score, l2g_attn_score), dim=-1)
            attn_mask_l2l_l2g = F.pad(attn_mask, (0, num_global), "constant", 1)
            v_l_g = torch.cat((v, l2g_global_v.expand(B, self.num_heads, num_cuboids, num_global, head_C)), dim=3)
            attn_score_l2l_l2g = masked_softmax(attn_score_l2l_l2g, mask=attn_mask_l2l_l2g)
            attn_score_l2l_l2g = self.attn_drop(attn_score_l2l_l2g)
            reordered_x = (attn_score_l2l_l2g @ v_l_g).permute(0, 2, 3, 1, 4).reshape(B, num_cuboids, x_cuboid_volume, self.dim)
        else:
            attn_score = masked_softmax(attn_score, mask=attn_mask)
            attn_score = self.attn_drop(attn_score)
            reordered_x = (attn_score @ v).permute(0, 2, 3, 1, 4).reshape(B, num_cuboids, x_cuboid_volume, self.dim)
        reordered_x = self.proj_drop(self.proj(reordered_x))
        # Step-5: shift back and slice
        shifted_x = cuboid_reorder_reverse(
            reordered_x, cuboid_size=x_cuboid_size, strategy=self.strategy, orig_data_shape=(x.shape[1], x.shape[2], x.shape[3])
        )
        if any(i > 0 for i in shift_hw):
            x = torch.roll(shifted_x, shifts=(shift_hw[0], shift_hw[1]), dims=(2, 3))
        else:
            x = shifted_x
        x = _generalize_unpadding(x, pad_t=pad_t_x, pad_h=pad_h, pad_w=pad_w, padding_type=self.padding_type)
        return x


class CuboidTransformerEncoder(nn.Module):
    """x --> attn_block --> patch_merge --> attn_block --> patch_merge --> ... --> out."""

    def __init__(
        self,
        input_shape,
        base_units=128,
        block_units=None,
        scale_alpha=1.0,
        depth=[4, 4, 4],
        downsample=2,
        downsample_type="patch_merge",
        block_attn_patterns=None,
        block_cuboid_size=[(4, 4, 4), (4, 4, 4)],
        block_strategy=[("l", "l", "l"), ("d", "d", "d")],
        block_shift_size=[(0, 0, 0), (0, 0, 0)],
        num_heads=4,
        attn_drop=0.0,
        proj_drop=0.0,
        ffn_drop=0.0,
        activation="leaky",
        ffn_activation="leaky",
        gated_ffn=False,
        norm_layer="layer_norm",
        use_inter_ffn=True,
        padding_type="ignore",
        checkpoint_level=True,
        use_relative_pos=True,
        self_attn_use_final_proj=True,
        use_global_vector=False,
        use_global_vector_ffn=True,
        use_global_self_attn=False,
        separate_global_qkv=False,
        global_dim_ratio=1,
        attn_linear_init_mode="0",
        ffn_linear_init_mode="0",
        conv_init_mode="0",
        down_linear_init_mode="0",
        norm_init_mode="0",
    ):
        super(CuboidTransformerEncoder, self).__init__()
        self.attn_linear_init_mode = attn_linear_init_mode
        self.ffn_linear_init_mode = ffn_linear_init_mode
        self.conv_init_mode = conv_init_mode
        self.down_linear_init_mode = down_linear_init_mode
        self.norm_init_mode = norm_init_mode

        self.input_shape = input_shape
        self.depth = depth
        self.num_blocks = len(depth)
        self.base_units = base_units
        self.scale_alpha = scale_alpha
        if not isinstance(downsample, (tuple, list)):
            downsample = (1, downsample, downsample)
        self.downsample = downsample
        self.downsample_type = downsample_type
        self.num_heads = num_heads
        self.use_global_vector = use_global_vector
        self.checkpoint_level = checkpoint_level
        if block_units is None:
            block_units = [round_to(base_units * int((max(downsample) ** scale_alpha) ** i), 4) for i in range(self.num_blocks)]
        else:
            assert len(block_units) == self.num_blocks and block_units[0] == base_units
        self.block_units = block_units

        if self.num_blocks > 1:
            if downsample_type == "patch_merge":
                self.down_layers = nn.ModuleList(
                    [
                        PatchMerging3D(
                            dim=self.block_units[i],
                            downsample=downsample,
                            padding_type=padding_type,
                            out_dim=self.block_units[i + 1],
                            linear_init_mode=down_linear_init_mode,
                            norm_init_mode=norm_init_mode,
                        )
                        for i in range(self.num_blocks - 1)
                    ]
                )
            else:
                raise NotImplementedError
            if self.use_global_vector:
                self.down_layer_global_proj = nn.ModuleList(
                    [
                        nn.Linear(
                            in_features=global_dim_ratio * self.block_units[i],
                            out_features=global_dim_ratio * self.block_units[i + 1],
                        )
                        for i in range(self.num_blocks - 1)
                    ]
                )

        if block_attn_patterns is not None:
            mem_shapes = self.get_mem_shapes()
            if isinstance(block_attn_patterns, (tuple, list)):
                assert len(block_attn_patterns) == self.num_blocks
            else:
                block_attn_patterns = [block_attn_patterns for _ in range(self.num_blocks)]
            block_cuboid_size = []
            block_strategy = []
            block_shift_size = []
            for idx, key in enumerate(block_attn_patterns):
                func = _get_pattern(CuboidSelfAttentionPatterns, key, "self-attention")
                cuboid_size, strategy, shift_size = func(mem_shapes[idx])
                block_cuboid_size.append(cuboid_size)
                block_strategy.append(strategy)
                block_shift_size.append(shift_size)
        else:
            if not isinstance(block_cuboid_size[0][0], (list, tuple)):
                block_cuboid_size = [block_cuboid_size for _ in range(self.num_blocks)]
            else:
                assert len(block_cuboid_size) == self.num_blocks, (
                    f"Incorrect input format! Received block_cuboid_size={block_cuboid_size}"
                )

            if not isinstance(block_strategy[0][0], (list, tuple)):
                block_strategy = [block_strategy for _ in range(self.num_blocks)]
            else:
                assert len(block_strategy) == self.num_blocks, f"Incorrect input format! Received block_strategy={block_strategy}"

            if not isinstance(block_shift_size[0][0], (list, tuple)):
                block_shift_size = [block_shift_size for _ in range(self.num_blocks)]
            else:
                assert len(block_shift_size) == self.num_blocks, (
                    f"Incorrect input format! Received block_shift_size={block_shift_size}"
                )
        self.block_cuboid_size = block_cuboid_size
        self.block_strategy = block_strategy
        self.block_shift_size = block_shift_size

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        StackCuboidSelfAttentionBlock(
                            dim=self.block_units[i],
                            num_heads=num_heads,
                            block_cuboid_size=block_cuboid_size[i],
                            block_strategy=block_strategy[i],
                            block_shift_size=block_shift_size[i],
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            ffn_drop=ffn_drop,
                            activation=ffn_activation,
                            gated_ffn=gated_ffn,
                            norm_layer=norm_layer,
                            use_inter_ffn=use_inter_ffn,
                            padding_type=padding_type,
                            use_global_vector=use_global_vector,
                            use_global_vector_ffn=use_global_vector_ffn,
                            use_global_self_attn=use_global_self_attn,
                            separate_global_qkv=separate_global_qkv,
                            global_dim_ratio=global_dim_ratio,
                            checkpoint_level=checkpoint_level,
                            use_relative_pos=use_relative_pos,
                            use_final_proj=self_attn_use_final_proj,
                            attn_linear_init_mode=attn_linear_init_mode,
                            ffn_linear_init_mode=ffn_linear_init_mode,
                            norm_init_mode=norm_init_mode,
                        )
                        for _ in range(depth[i])
                    ]
                )
                for i in range(self.num_blocks)
            ]
        )
        self.reset_parameters()

    def reset_parameters(self):
        if self.num_blocks > 1:
            for m in self.down_layers:
                m.reset_parameters()
            if self.use_global_vector:
                apply_initialization(self.down_layer_global_proj, linear_mode=self.down_linear_init_mode)
        for ms in self.blocks:
            for m in ms:
                m.reset_parameters()

    def get_mem_shapes(self):
        """Shapes (T, H, W, C) of the encoder outputs, used to build the decoder."""
        if self.num_blocks == 1:
            return [self.input_shape]
        else:
            mem_shapes = [self.input_shape]
            curr_shape = self.input_shape
            for down_layer in self.down_layers:
                curr_shape = down_layer.get_out_shape(curr_shape)
                mem_shapes.append(curr_shape)
            return mem_shapes

    def forward(self, x, global_vectors=None):
        B, T, H, W, C_in = x.shape
        assert (T, H, W, C_in) == tuple(self.input_shape)

        if self.use_global_vector:
            out = []
            global_mem_out = []
            for i in range(self.num_blocks):
                for l in self.blocks[i]:
                    x, global_vectors = l(x, global_vectors)
                out.append(x)
                global_mem_out.append(global_vectors)
                if self.num_blocks > 1 and i < self.num_blocks - 1:
                    x = self.down_layers[i](x)
                    global_vectors = self.down_layer_global_proj[i](global_vectors)
            return out, global_mem_out
        else:
            out = []
            for i in range(self.num_blocks):
                x = self.blocks[i](x)
                out.append(x)
                if self.num_blocks > 1 and i < self.num_blocks - 1:
                    x = self.down_layers[i](x)
            return out


class StackCuboidCrossAttentionBlock(nn.Module):
    """A stack of cuboid cross-attention layers, each followed by an FFN when ``use_inter_ffn``."""

    def __init__(
        self,
        dim,
        num_heads,
        block_cuboid_hw=[(4, 4), (4, 4)],
        block_shift_hw=[(0, 0), (2, 2)],
        block_n_temporal=[1, 2],
        block_strategy=[("d", "d", "d"), ("l", "l", "l")],
        padding_type="ignore",
        cross_last_n_frames=None,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        ffn_drop=0.0,
        activation="leaky",
        gated_ffn=False,
        norm_layer="layer_norm",
        use_inter_ffn=True,
        max_temporal_relative=50,
        checkpoint_level=1,
        use_relative_pos=True,
        use_global_vector=False,
        separate_global_qkv=False,
        global_dim_ratio=1,
        attn_linear_init_mode="0",
        ffn_linear_init_mode="0",
        norm_init_mode="0",
    ):
        super(StackCuboidCrossAttentionBlock, self).__init__()
        self.attn_linear_init_mode = attn_linear_init_mode
        self.ffn_linear_init_mode = ffn_linear_init_mode
        self.norm_init_mode = norm_init_mode

        assert len(block_cuboid_hw[0]) > 0 and len(block_shift_hw) > 0 and len(block_strategy) > 0, (
            f"Incorrect format. block_cuboid_hw={block_cuboid_hw}, block_shift_hw={block_shift_hw}, "
            f"block_strategy={block_strategy}"
        )
        assert len(block_cuboid_hw) == len(block_shift_hw) == len(block_strategy)
        self.num_attn = len(block_cuboid_hw)
        self.checkpoint_level = checkpoint_level
        self.use_inter_ffn = use_inter_ffn
        self.use_global_vector = use_global_vector
        num_ffn = self.num_attn if self.use_inter_ffn else 1
        self.ffn_l = nn.ModuleList(
            [
                PositionwiseFFN(
                    units=dim,
                    hidden_size=4 * dim,
                    activation_dropout=ffn_drop,
                    dropout=ffn_drop,
                    gated_proj=gated_ffn,
                    activation=activation,
                    normalization=norm_layer,
                    pre_norm=True,
                    linear_init_mode=ffn_linear_init_mode,
                    norm_init_mode=norm_init_mode,
                )
                for _ in range(num_ffn)
            ]
        )
        self.attn_l = nn.ModuleList(
            [
                CuboidCrossAttentionLayer(
                    dim=dim,
                    num_heads=num_heads,
                    cuboid_hw=ele_cuboid_hw,
                    shift_hw=ele_shift_hw,
                    strategy=ele_strategy,
                    n_temporal=ele_n_temporal,
                    cross_last_n_frames=cross_last_n_frames,
                    padding_type=padding_type,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    norm_layer=norm_layer,
                    max_temporal_relative=max_temporal_relative,
                    use_global_vector=use_global_vector,
                    separate_global_qkv=separate_global_qkv,
                    global_dim_ratio=global_dim_ratio,
                    checkpoint_level=checkpoint_level,
                    use_relative_pos=use_relative_pos,
                    attn_linear_init_mode=attn_linear_init_mode,
                    ffn_linear_init_mode=ffn_linear_init_mode,
                    norm_init_mode=norm_init_mode,
                )
                for ele_cuboid_hw, ele_shift_hw, ele_strategy, ele_n_temporal in zip(
                    block_cuboid_hw, block_shift_hw, block_strategy, block_n_temporal
                )
            ]
        )

    def reset_parameters(self):
        for m in self.ffn_l:
            m.reset_parameters()
        for m in self.attn_l:
            m.reset_parameters()

    def forward(self, x, mem, mem_global_vector=None):
        if self.use_inter_ffn:
            for attn, ffn in zip(self.attn_l, self.ffn_l):
                if self.checkpoint_level >= 2 and self.training:
                    x = x + _checkpoint(attn, x, mem, mem_global_vector)
                else:
                    x = x + attn(x, mem, mem_global_vector)
                if self.checkpoint_level >= 1 and self.training:
                    x = _checkpoint(ffn, x)
                else:
                    x = ffn(x)
            return x
        else:
            for attn in self.attn_l:
                if self.checkpoint_level >= 2 and self.training:
                    x = x + _checkpoint(attn, x, mem, mem_global_vector)
                else:
                    x = x + attn(x, mem, mem_global_vector)
            if self.checkpoint_level >= 1 and self.training:
                x = _checkpoint(self.ffn_l[0], x)
            else:
                x = self.ffn_l[0](x)
        return x


class CuboidTransformerDecoder(nn.Module):
    """For each hierarchy (top to bottom): StackCuboidSelfAttention, then StackCuboidCrossAttention
    with the encoder memory of that hierarchy, then upsampling."""

    def __init__(
        self,
        target_temporal_length,
        mem_shapes,
        cross_start=0,
        depth=[2, 2],
        upsample_type="upsample",
        upsample_kernel_size=3,
        block_self_attn_patterns=None,
        block_self_cuboid_size=[(4, 4, 4), (4, 4, 4)],
        block_self_cuboid_strategy=[("l", "l", "l"), ("d", "d", "d")],
        block_self_shift_size=[(1, 1, 1), (0, 0, 0)],
        block_cross_attn_patterns=None,
        block_cross_cuboid_hw=[(4, 4), (4, 4)],
        block_cross_cuboid_strategy=[("l", "l", "l"), ("d", "l", "l")],
        block_cross_shift_hw=[(0, 0), (0, 0)],
        block_cross_n_temporal=[1, 2],
        cross_last_n_frames=None,
        num_heads=4,
        attn_drop=0.0,
        proj_drop=0.0,
        ffn_drop=0.0,
        ffn_activation="leaky",
        gated_ffn=False,
        norm_layer="layer_norm",
        use_inter_ffn=False,
        hierarchical_pos_embed=False,
        pos_embed_type="t+hw",
        max_temporal_relative=50,
        padding_type="ignore",
        checkpoint_level=True,
        use_relative_pos=True,
        self_attn_use_final_proj=True,
        use_first_self_attn=False,
        use_self_global=False,
        self_update_global=True,
        use_cross_global=False,
        use_global_vector_ffn=True,
        use_global_self_attn=False,
        separate_global_qkv=False,
        global_dim_ratio=1,
        attn_linear_init_mode="0",
        ffn_linear_init_mode="0",
        conv_init_mode="0",
        up_linear_init_mode="0",
        norm_init_mode="0",
    ):
        super(CuboidTransformerDecoder, self).__init__()
        self.attn_linear_init_mode = attn_linear_init_mode
        self.ffn_linear_init_mode = ffn_linear_init_mode
        self.conv_init_mode = conv_init_mode
        self.up_linear_init_mode = up_linear_init_mode
        self.norm_init_mode = norm_init_mode

        assert len(depth) == len(mem_shapes)
        self.target_temporal_length = target_temporal_length
        self.num_blocks = len(mem_shapes)
        self.cross_start = cross_start
        self.mem_shapes = mem_shapes
        self.depth = depth
        self.upsample_type = upsample_type
        self.hierarchical_pos_embed = hierarchical_pos_embed
        self.checkpoint_level = checkpoint_level
        self.use_self_global = use_self_global
        self.self_update_global = self_update_global
        self.use_cross_global = use_cross_global
        self.use_global_vector_ffn = use_global_vector_ffn
        self.use_first_self_attn = use_first_self_attn
        if block_self_attn_patterns is not None:
            if isinstance(block_self_attn_patterns, (tuple, list)):
                assert len(block_self_attn_patterns) == self.num_blocks
            else:
                block_self_attn_patterns = [block_self_attn_patterns for _ in range(self.num_blocks)]
            block_self_cuboid_size = []
            block_self_cuboid_strategy = []
            block_self_shift_size = []
            for idx, key in enumerate(block_self_attn_patterns):
                func = _get_pattern(CuboidSelfAttentionPatterns, key, "self-attention")
                cuboid_size, strategy, shift_size = func(mem_shapes[idx])
                block_self_cuboid_size.append(cuboid_size)
                block_self_cuboid_strategy.append(strategy)
                block_self_shift_size.append(shift_size)
        else:
            if not isinstance(block_self_cuboid_size[0][0], (list, tuple)):
                block_self_cuboid_size = [block_self_cuboid_size for _ in range(self.num_blocks)]
            else:
                assert len(block_self_cuboid_size) == self.num_blocks, (
                    f"Incorrect input format! Received block_self_cuboid_size={block_self_cuboid_size}"
                )

            if not isinstance(block_self_cuboid_strategy[0][0], (list, tuple)):
                block_self_cuboid_strategy = [block_self_cuboid_strategy for _ in range(self.num_blocks)]
            else:
                assert len(block_self_cuboid_strategy) == self.num_blocks, (
                    f"Incorrect input format! Received block_self_cuboid_strategy={block_self_cuboid_strategy}"
                )

            if not isinstance(block_self_shift_size[0][0], (list, tuple)):
                block_self_shift_size = [block_self_shift_size for _ in range(self.num_blocks)]
            else:
                assert len(block_self_shift_size) == self.num_blocks, (
                    f"Incorrect input format! Received block_self_shift_size={block_self_shift_size}"
                )
        self_blocks = []
        for i in range(self.num_blocks):
            if not self.use_first_self_attn and i == self.num_blocks - 1:
                # The top block has no additional self-attention layer.
                ele_depth = depth[i] - 1
            else:
                ele_depth = depth[i]
            stack_cuboid_blocks = [
                StackCuboidSelfAttentionBlock(
                    dim=self.mem_shapes[i][-1],
                    num_heads=num_heads,
                    block_cuboid_size=block_self_cuboid_size[i],
                    block_strategy=block_self_cuboid_strategy[i],
                    block_shift_size=block_self_shift_size[i],
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    ffn_drop=ffn_drop,
                    activation=ffn_activation,
                    gated_ffn=gated_ffn,
                    norm_layer=norm_layer,
                    use_inter_ffn=use_inter_ffn,
                    padding_type=padding_type,
                    use_global_vector=use_self_global,
                    use_global_vector_ffn=use_global_vector_ffn,
                    use_global_self_attn=use_global_self_attn,
                    separate_global_qkv=separate_global_qkv,
                    global_dim_ratio=global_dim_ratio,
                    checkpoint_level=checkpoint_level,
                    use_relative_pos=use_relative_pos,
                    use_final_proj=self_attn_use_final_proj,
                    attn_linear_init_mode=attn_linear_init_mode,
                    ffn_linear_init_mode=ffn_linear_init_mode,
                    norm_init_mode=norm_init_mode,
                )
                for _ in range(ele_depth)
            ]
            self_blocks.append(nn.ModuleList(stack_cuboid_blocks))
        self.self_blocks = nn.ModuleList(self_blocks)

        if block_cross_attn_patterns is not None:
            if isinstance(block_cross_attn_patterns, (tuple, list)):
                assert len(block_cross_attn_patterns) == self.num_blocks
            else:
                block_cross_attn_patterns = [block_cross_attn_patterns for _ in range(self.num_blocks)]

            block_cross_cuboid_hw = []
            block_cross_cuboid_strategy = []
            block_cross_shift_hw = []
            block_cross_n_temporal = []
            for idx, key in enumerate(block_cross_attn_patterns):
                if key == "last_frame_dst":
                    cuboid_hw = None
                    shift_hw = None
                    strategy = None
                    n_temporal = None
                else:
                    func = _get_pattern(CuboidCrossAttentionPatterns, key, "cross-attention")
                    cuboid_hw, shift_hw, strategy, n_temporal = func(mem_shapes[idx])
                block_cross_cuboid_hw.append(cuboid_hw)
                block_cross_cuboid_strategy.append(strategy)
                block_cross_shift_hw.append(shift_hw)
                block_cross_n_temporal.append(n_temporal)
        else:
            if not isinstance(block_cross_cuboid_hw[0][0], (list, tuple)):
                block_cross_cuboid_hw = [block_cross_cuboid_hw for _ in range(self.num_blocks)]
            else:
                assert len(block_cross_cuboid_hw) == self.num_blocks, (
                    f"Incorrect input format! Received block_cross_cuboid_hw={block_cross_cuboid_hw}"
                )

            if not isinstance(block_cross_cuboid_strategy[0][0], (list, tuple)):
                block_cross_cuboid_strategy = [block_cross_cuboid_strategy for _ in range(self.num_blocks)]
            else:
                assert len(block_cross_cuboid_strategy) == self.num_blocks, (
                    f"Incorrect input format! Received block_cross_cuboid_strategy={block_cross_cuboid_strategy}"
                )

            if not isinstance(block_cross_shift_hw[0][0], (list, tuple)):
                block_cross_shift_hw = [block_cross_shift_hw for _ in range(self.num_blocks)]
            else:
                assert len(block_cross_shift_hw) == self.num_blocks, (
                    f"Incorrect input format! Received block_cross_shift_hw={block_cross_shift_hw}"
                )
            if not isinstance(block_cross_n_temporal[0], (list, tuple)):
                block_cross_n_temporal = [block_cross_n_temporal for _ in range(self.num_blocks)]
            else:
                assert len(block_cross_n_temporal) == self.num_blocks, (
                    f"Incorrect input format! Received block_cross_n_temporal={block_cross_n_temporal}"
                )
        self.cross_blocks = nn.ModuleList()
        for i in range(self.cross_start, self.num_blocks):
            cross_block = nn.ModuleList(
                [
                    StackCuboidCrossAttentionBlock(
                        dim=self.mem_shapes[i][-1],
                        num_heads=num_heads,
                        block_cuboid_hw=block_cross_cuboid_hw[i],
                        block_strategy=block_cross_cuboid_strategy[i],
                        block_shift_hw=block_cross_shift_hw[i],
                        block_n_temporal=block_cross_n_temporal[i],
                        cross_last_n_frames=cross_last_n_frames,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        ffn_drop=ffn_drop,
                        gated_ffn=gated_ffn,
                        norm_layer=norm_layer,
                        use_inter_ffn=use_inter_ffn,
                        activation=ffn_activation,
                        max_temporal_relative=max_temporal_relative,
                        padding_type=padding_type,
                        use_global_vector=use_cross_global,
                        separate_global_qkv=separate_global_qkv,
                        global_dim_ratio=global_dim_ratio,
                        checkpoint_level=checkpoint_level,
                        use_relative_pos=use_relative_pos,
                        attn_linear_init_mode=attn_linear_init_mode,
                        ffn_linear_init_mode=ffn_linear_init_mode,
                        norm_init_mode=norm_init_mode,
                    )
                    for _ in range(depth[i])
                ]
            )
            self.cross_blocks.append(cross_block)

        if self.num_blocks > 1:
            if self.upsample_type == "upsample":
                self.upsample_layers = nn.ModuleList(
                    [
                        Upsample3DLayer(
                            dim=self.mem_shapes[i + 1][-1],
                            out_dim=self.mem_shapes[i][-1],
                            target_size=(target_temporal_length,) + tuple(self.mem_shapes[i][1:3]),
                            kernel_size=upsample_kernel_size,
                            temporal_upsample=False,
                            conv_init_mode=conv_init_mode,
                        )
                        for i in range(self.num_blocks - 1)
                    ]
                )
            else:
                raise NotImplementedError
            if self.hierarchical_pos_embed:
                self.hierarchical_pos_embed_l = nn.ModuleList(
                    [
                        PosEmbed(
                            embed_dim=self.mem_shapes[i][-1],
                            typ=pos_embed_type,
                            maxT=target_temporal_length,
                            maxH=self.mem_shapes[i][1],
                            maxW=self.mem_shapes[i][2],
                        )
                        for i in range(self.num_blocks - 1)
                    ]
                )

        self.reset_parameters()

    def reset_parameters(self):
        for ms in self.self_blocks:
            for m in ms:
                m.reset_parameters()
        for ms in self.cross_blocks:
            for m in ms:
                m.reset_parameters()
        if self.num_blocks > 1:
            for m in self.upsample_layers:
                m.reset_parameters()
        if self.hierarchical_pos_embed:
            for m in self.hierarchical_pos_embed_l:
                m.reset_parameters()

    def forward(self, x, mem_l, mem_global_vector_l=None):
        B, T_top, H_top, W_top, C = x.shape
        assert T_top == self.target_temporal_length
        assert (H_top, W_top) == (self.mem_shapes[-1][1], self.mem_shapes[-1][2])
        for i in range(self.num_blocks - 1, -1, -1):
            mem_global_vector = None if mem_global_vector_l is None else mem_global_vector_l[i]
            if not self.use_first_self_attn and i == self.num_blocks - 1:
                # The top block starts directly with cross-attention.
                if i >= self.cross_start:
                    x = self.cross_blocks[i - self.cross_start][0](x, mem_l[i], mem_global_vector)
                for idx in range(self.depth[i] - 1):
                    if self.use_self_global:
                        if self.self_update_global:
                            x, mem_global_vector = self.self_blocks[i][idx](x, mem_global_vector)
                        else:
                            x, _ = self.self_blocks[i][idx](x, mem_global_vector)
                    else:
                        x = self.self_blocks[i][idx](x)
                    if i >= self.cross_start:
                        x = self.cross_blocks[i - self.cross_start][idx + 1](x, mem_l[i], mem_global_vector)
            else:
                for idx in range(self.depth[i]):
                    if self.use_self_global:
                        if self.self_update_global:
                            x, mem_global_vector = self.self_blocks[i][idx](x, mem_global_vector)
                        else:
                            x, _ = self.self_blocks[i][idx](x, mem_global_vector)
                    else:
                        x = self.self_blocks[i][idx](x)
                    if i >= self.cross_start:
                        x = self.cross_blocks[i - self.cross_start][idx](x, mem_l[i], mem_global_vector)
            if i > 0:
                x = self.upsample_layers[i - 1](x)
                if self.hierarchical_pos_embed:
                    x = self.hierarchical_pos_embed_l[i - 1](x)
        return x


class InitialEncoder(nn.Module):
    """[K x (Conv3x3 -> GroupNorm(16) -> act)] -> PatchMerge, applied frame by frame."""

    def __init__(
        self,
        dim,
        out_dim,
        downsample_scale: Union[int, Sequence[int]],
        num_conv_layers=2,
        activation="leaky",
        padding_type="nearest",
        conv_init_mode="0",
        linear_init_mode="0",
        norm_init_mode="0",
    ):
        super(InitialEncoder, self).__init__()

        self.num_conv_layers = num_conv_layers
        self.conv_init_mode = conv_init_mode
        self.linear_init_mode = linear_init_mode
        self.norm_init_mode = norm_init_mode

        conv_block = []
        for i in range(num_conv_layers):
            if i == 0:
                conv_block.append(nn.Conv2d(kernel_size=(3, 3), padding=(1, 1), in_channels=dim, out_channels=out_dim))
                conv_block.append(nn.GroupNorm(16, out_dim))
                conv_block.append(get_activation(activation))
            else:
                conv_block.append(nn.Conv2d(kernel_size=(3, 3), padding=(1, 1), in_channels=out_dim, out_channels=out_dim))
                conv_block.append(nn.GroupNorm(16, out_dim))
                conv_block.append(get_activation(activation))

        self.conv_block = nn.Sequential(*conv_block)
        if isinstance(downsample_scale, int):
            patch_merge_downsample = (1, downsample_scale, downsample_scale)
        elif len(downsample_scale) == 2:
            patch_merge_downsample = (1, *downsample_scale)
        elif len(downsample_scale) == 3:
            patch_merge_downsample = tuple(downsample_scale)
        else:
            raise NotImplementedError(f"downsample_scale {downsample_scale} format not supported!")
        self.patch_merge = PatchMerging3D(
            dim=out_dim,
            out_dim=out_dim,
            padding_type=padding_type,
            downsample=patch_merge_downsample,
            linear_init_mode=linear_init_mode,
            norm_init_mode=norm_init_mode,
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            apply_initialization(m, conv_mode=self.conv_init_mode, linear_mode=self.linear_init_mode, norm_mode=self.norm_init_mode)

    def forward(self, x):
        B, T, H, W, C = x.shape
        if self.num_conv_layers > 0:
            x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
            x = self.conv_block(x).permute(0, 2, 3, 1)
            x = self.patch_merge(x.reshape(B, T, H, W, -1))
        else:
            x = self.patch_merge(x)
        return x


class FinalDecoder(nn.Module):
    """Upsample -> [K x (Conv3x3 -> GroupNorm(16) -> act)], applied frame by frame."""

    def __init__(
        self,
        target_thw,
        dim,
        num_conv_layers=2,
        activation="leaky",
        conv_init_mode="0",
        linear_init_mode="0",
        norm_init_mode="0",
    ):
        super(FinalDecoder, self).__init__()
        self.target_thw = target_thw
        self.dim = dim
        self.num_conv_layers = num_conv_layers
        self.conv_init_mode = conv_init_mode
        self.linear_init_mode = linear_init_mode
        self.norm_init_mode = norm_init_mode

        conv_block = []
        for i in range(num_conv_layers):
            conv_block.append(nn.Conv2d(kernel_size=(3, 3), padding=(1, 1), in_channels=dim, out_channels=dim))
            conv_block.append(nn.GroupNorm(16, dim))
            conv_block.append(get_activation(activation))
        self.conv_block = nn.Sequential(*conv_block)
        self.upsample = Upsample3DLayer(dim=dim, out_dim=dim, target_size=target_thw, kernel_size=3, conv_init_mode=conv_init_mode)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            apply_initialization(m, conv_mode=self.conv_init_mode, linear_mode=self.linear_init_mode, norm_mode=self.norm_init_mode)

    def forward(self, x):
        x = self.upsample(x)
        if self.num_conv_layers > 0:
            B, T, H, W, C = x.shape
            x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
            x = self.conv_block(x).permute(0, 2, 3, 1).reshape(B, T, H, W, -1)
        return x


class InitialStackPatchMergingEncoder(nn.Module):
    """[K x Conv2D] -> PatchMerge -> ... -> [K x Conv2D] -> PatchMerge (GroupNorm groups = dim // 4)."""

    def __init__(
        self,
        num_merge: int,
        in_dim,
        out_dim_list,
        downsample_scale_list,
        num_conv_per_merge_list=None,
        activation="leaky",
        padding_type="nearest",
        conv_init_mode="0",
        linear_init_mode="0",
        norm_init_mode="0",
    ):
        super(InitialStackPatchMergingEncoder, self).__init__()

        self.conv_init_mode = conv_init_mode
        self.linear_init_mode = linear_init_mode
        self.norm_init_mode = norm_init_mode

        self.num_merge = num_merge
        self.in_dim = in_dim
        self.out_dim_list = out_dim_list[:num_merge]
        self.downsample_scale_list = downsample_scale_list[:num_merge]
        self.num_conv_per_merge_list = num_conv_per_merge_list
        self.num_group_list = [max(1, out_dim // 4) for out_dim in self.out_dim_list]

        self.conv_block_list = nn.ModuleList()
        self.patch_merge_list = nn.ModuleList()
        for i in range(num_merge):
            if i == 0:
                in_dim = in_dim
            else:
                in_dim = self.out_dim_list[i - 1]
            out_dim = self.out_dim_list[i]
            downsample_scale = self.downsample_scale_list[i]

            conv_block = []
            for j in range(self.num_conv_per_merge_list[i]):
                if j == 0:
                    conv_in_dim = in_dim
                else:
                    conv_in_dim = out_dim
                conv_block.append(nn.Conv2d(kernel_size=(3, 3), padding=(1, 1), in_channels=conv_in_dim, out_channels=out_dim))
                conv_block.append(nn.GroupNorm(self.num_group_list[i], out_dim))
                conv_block.append(get_activation(activation))

            conv_block = nn.Sequential(*conv_block)
            self.conv_block_list.append(conv_block)
            patch_merge = PatchMerging3D(
                dim=out_dim,
                out_dim=out_dim,
                padding_type=padding_type,
                downsample=(1, downsample_scale, downsample_scale),
                linear_init_mode=linear_init_mode,
                norm_init_mode=norm_init_mode,
            )
            self.patch_merge_list.append(patch_merge)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            apply_initialization(m, conv_mode=self.conv_init_mode, linear_mode=self.linear_init_mode, norm_mode=self.norm_init_mode)

    def get_out_shape_list(self, input_shape):
        """Shapes (T, H, W, C) after each patch merging."""
        out_shape_list = []
        for patch_merge in self.patch_merge_list:
            input_shape = patch_merge.get_out_shape(input_shape)
            out_shape_list.append(input_shape)
        return out_shape_list

    def forward(self, x):
        for i, (conv_block, patch_merge) in enumerate(zip(self.conv_block_list, self.patch_merge_list)):
            B, T, H, W, C = x.shape
            if self.num_conv_per_merge_list[i] > 0:
                x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
                x = conv_block(x).permute(0, 2, 3, 1).reshape(B, T, H, W, -1)
            x = patch_merge(x)
        return x


class FinalStackUpsamplingDecoder(nn.Module):
    """Upsample -> [K x Conv2D] -> ... -> Upsample -> [K x Conv2D] (GroupNorm groups = dim // 4)."""

    def __init__(
        self,
        target_shape_list,
        in_dim,
        num_conv_per_up_list=None,
        activation="leaky",
        conv_init_mode="0",
        linear_init_mode="0",
        norm_init_mode="0",
    ):
        super(FinalStackUpsamplingDecoder, self).__init__()
        self.conv_init_mode = conv_init_mode
        self.linear_init_mode = linear_init_mode
        self.norm_init_mode = norm_init_mode

        self.target_shape_list = target_shape_list
        self.out_dim_list = [target_shape[-1] for target_shape in self.target_shape_list]
        self.num_upsample = len(target_shape_list)
        self.in_dim = in_dim
        self.num_conv_per_up_list = num_conv_per_up_list
        self.num_group_list = [max(1, out_dim // 4) for out_dim in self.out_dim_list]

        self.conv_block_list = nn.ModuleList()
        self.upsample_list = nn.ModuleList()
        for i in range(self.num_upsample):
            if i == 0:
                in_dim = in_dim
            else:
                in_dim = self.out_dim_list[i - 1]
            out_dim = self.out_dim_list[i]

            upsample = Upsample3DLayer(
                dim=in_dim, out_dim=in_dim, target_size=tuple(target_shape_list[i][:-1]), kernel_size=3, conv_init_mode=conv_init_mode
            )
            self.upsample_list.append(upsample)
            conv_block = []
            for j in range(num_conv_per_up_list[i]):
                if j == 0:
                    conv_in_dim = in_dim
                else:
                    conv_in_dim = out_dim
                conv_block.append(nn.Conv2d(kernel_size=(3, 3), padding=(1, 1), in_channels=conv_in_dim, out_channels=out_dim))
                conv_block.append(nn.GroupNorm(self.num_group_list[i], out_dim))
                conv_block.append(get_activation(activation))
            conv_block = nn.Sequential(*conv_block)
            self.conv_block_list.append(conv_block)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            apply_initialization(m, conv_mode=self.conv_init_mode, linear_mode=self.linear_init_mode, norm_mode=self.norm_init_mode)

    @staticmethod
    def get_init_params(enc_input_shape, enc_out_shape_list, large_channel=False):
        dec_target_shape_list = list(enc_out_shape_list[:-1])[::-1] + [tuple(enc_input_shape)]
        if large_channel:
            dec_target_shape_list_large_channel = []
            for i, enc_out_shape in enumerate(enc_out_shape_list[::-1]):
                dec_target_shape_large_channel = list(dec_target_shape_list[i])
                dec_target_shape_large_channel[-1] = enc_out_shape[-1]
                dec_target_shape_list_large_channel.append(tuple(dec_target_shape_large_channel))
            dec_target_shape_list = dec_target_shape_list_large_channel
        dec_in_dim = enc_out_shape_list[-1][-1]
        return dec_target_shape_list, dec_in_dim

    def forward(self, x):
        for i, (conv_block, upsample) in enumerate(zip(self.conv_block_list, self.upsample_list)):
            x = upsample(x)
            if self.num_conv_per_up_list[i] > 0:
                B, T, H, W, C = x.shape
                x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
                x = conv_block(x).permute(0, 2, 3, 1).reshape(B, T, H, W, -1)
        return x


class CuboidTransformerModel(nn.Module):
    """Earthformer (official ``CuboidTransformerModel``): a non-autoregressive hierarchical
    encoder-decoder built from cuboid attention.

    ::

        x --> downsample (optional) --> (+pos_embed) --> enc --> mem_l     initial_z (+pos_embed) --> FC
                                                          |          |
                                                          |----------|
                                                                |
        y <-- upsample (optional) <-- dec <---------------------

    Input ``(batch, T_in, H, W, C_in)`` (layout NTHWC as in the official code) with
    ``(T_in, H, W, C_in) == input_shape``; output ``(batch, T_out, H, W, C_out)`` with
    ``target_shape = (T_out, H, W, C_out)``.
    """

    def __init__(
        self,
        input_shape,
        target_shape,
        base_units=128,
        block_units=None,
        scale_alpha=1.0,
        num_heads=4,
        attn_drop=0.0,
        proj_drop=0.0,
        ffn_drop=0.0,
        # inter-attn downsample/upsample
        downsample=2,
        downsample_type="patch_merge",
        upsample_type="upsample",
        upsample_kernel_size=3,
        # encoder
        enc_depth=[4, 4, 4],
        enc_attn_patterns=None,
        enc_cuboid_size=[(4, 4, 4), (4, 4, 4)],
        enc_cuboid_strategy=[("l", "l", "l"), ("d", "d", "d")],
        enc_shift_size=[(0, 0, 0), (0, 0, 0)],
        enc_use_inter_ffn=True,
        # decoder
        dec_depth=[2, 2],
        dec_cross_start=0,
        dec_self_attn_patterns=None,
        dec_self_cuboid_size=[(4, 4, 4), (4, 4, 4)],
        dec_self_cuboid_strategy=[("l", "l", "l"), ("d", "d", "d")],
        dec_self_shift_size=[(1, 1, 1), (0, 0, 0)],
        dec_cross_attn_patterns=None,
        dec_cross_cuboid_hw=[(4, 4), (4, 4)],
        dec_cross_cuboid_strategy=[("l", "l", "l"), ("d", "l", "l")],
        dec_cross_shift_hw=[(0, 0), (0, 0)],
        dec_cross_n_temporal=[1, 2],
        dec_cross_last_n_frames=None,
        dec_use_inter_ffn=True,
        dec_hierarchical_pos_embed=False,
        # global vectors
        num_global_vectors=4,
        use_dec_self_global=True,
        dec_self_update_global=True,
        use_dec_cross_global=True,
        use_global_vector_ffn=True,
        use_global_self_attn=False,
        separate_global_qkv=False,
        global_dim_ratio=1,
        z_init_method="nearest_interp",
        # initial downsample and final upsample
        initial_downsample_type="conv",
        initial_downsample_activation="leaky",
        # initial_downsample_type == "conv"
        initial_downsample_scale=1,
        initial_downsample_conv_layers=2,
        final_upsample_conv_layers=2,
        # initial_downsample_type == "stack_conv"
        initial_downsample_stack_conv_num_layers=1,
        initial_downsample_stack_conv_dim_list=None,
        initial_downsample_stack_conv_downscale_list=[1],
        initial_downsample_stack_conv_num_conv_list=[2],
        ffn_activation="leaky",
        gated_ffn=False,
        norm_layer="layer_norm",
        padding_type="ignore",
        pos_embed_type="t+hw",
        checkpoint_level=True,
        use_relative_pos=True,
        self_attn_use_final_proj=True,
        dec_use_first_self_attn=False,
        # initialization
        attn_linear_init_mode="0",
        ffn_linear_init_mode="0",
        conv_init_mode="0",
        down_up_linear_init_mode="0",
        norm_init_mode="0",
    ):
        super(CuboidTransformerModel, self).__init__()
        self.attn_linear_init_mode = attn_linear_init_mode
        self.ffn_linear_init_mode = ffn_linear_init_mode
        self.conv_init_mode = conv_init_mode
        self.down_up_linear_init_mode = down_up_linear_init_mode
        self.norm_init_mode = norm_init_mode

        assert len(enc_depth) == len(dec_depth)
        self.base_units = base_units
        self.num_global_vectors = num_global_vectors
        if global_dim_ratio != 1:
            assert separate_global_qkv is True, "Setting global_dim_ratio != 1 requires separate_global_qkv == True."
        self.global_dim_ratio = global_dim_ratio
        self.z_init_method = z_init_method
        assert self.z_init_method in ["zeros", "nearest_interp", "last", "mean"]

        self.input_shape = tuple(input_shape)
        self.target_shape = tuple(target_shape)
        T_in, H_in, W_in, C_in = self.input_shape
        T_out, H_out, W_out, C_out = self.target_shape
        assert H_in == H_out and W_in == W_out

        if self.num_global_vectors > 0:
            self.init_global_vectors = nn.Parameter(torch.zeros((self.num_global_vectors, global_dim_ratio * base_units)))

        new_input_shape = self.get_initial_encoder_final_decoder(
            initial_downsample_scale=initial_downsample_scale,
            initial_downsample_type=initial_downsample_type,
            activation=initial_downsample_activation,
            initial_downsample_conv_layers=initial_downsample_conv_layers,
            final_upsample_conv_layers=final_upsample_conv_layers,
            padding_type=padding_type,
            initial_downsample_stack_conv_num_layers=initial_downsample_stack_conv_num_layers,
            initial_downsample_stack_conv_dim_list=initial_downsample_stack_conv_dim_list,
            initial_downsample_stack_conv_downscale_list=initial_downsample_stack_conv_downscale_list,
            initial_downsample_stack_conv_num_conv_list=initial_downsample_stack_conv_num_conv_list,
        )
        T_in, H_in, W_in, _ = new_input_shape

        self.encoder = CuboidTransformerEncoder(
            input_shape=(T_in, H_in, W_in, base_units),
            base_units=base_units,
            block_units=block_units,
            scale_alpha=scale_alpha,
            depth=enc_depth,
            downsample=downsample,
            downsample_type=downsample_type,
            block_attn_patterns=enc_attn_patterns,
            block_cuboid_size=enc_cuboid_size,
            block_strategy=enc_cuboid_strategy,
            block_shift_size=enc_shift_size,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            ffn_drop=ffn_drop,
            gated_ffn=gated_ffn,
            ffn_activation=ffn_activation,
            norm_layer=norm_layer,
            use_inter_ffn=enc_use_inter_ffn,
            padding_type=padding_type,
            use_global_vector=num_global_vectors > 0,
            use_global_vector_ffn=use_global_vector_ffn,
            use_global_self_attn=use_global_self_attn,
            separate_global_qkv=separate_global_qkv,
            global_dim_ratio=global_dim_ratio,
            checkpoint_level=checkpoint_level,
            use_relative_pos=use_relative_pos,
            self_attn_use_final_proj=self_attn_use_final_proj,
            attn_linear_init_mode=attn_linear_init_mode,
            ffn_linear_init_mode=ffn_linear_init_mode,
            conv_init_mode=conv_init_mode,
            down_linear_init_mode=down_up_linear_init_mode,
            norm_init_mode=norm_init_mode,
        )
        self.enc_pos_embed = PosEmbed(embed_dim=base_units, typ=pos_embed_type, maxH=H_in, maxW=W_in, maxT=T_in)
        mem_shapes = self.encoder.get_mem_shapes()

        self.z_proj = nn.Linear(mem_shapes[-1][-1], mem_shapes[-1][-1])
        self.dec_pos_embed = PosEmbed(
            embed_dim=mem_shapes[-1][-1], typ=pos_embed_type, maxT=T_out, maxH=mem_shapes[-1][1], maxW=mem_shapes[-1][2]
        )
        self.decoder = CuboidTransformerDecoder(
            target_temporal_length=T_out,
            mem_shapes=mem_shapes,
            cross_start=dec_cross_start,
            depth=dec_depth,
            upsample_type=upsample_type,
            block_self_attn_patterns=dec_self_attn_patterns,
            block_self_cuboid_size=dec_self_cuboid_size,
            block_self_shift_size=dec_self_shift_size,
            block_self_cuboid_strategy=dec_self_cuboid_strategy,
            block_cross_attn_patterns=dec_cross_attn_patterns,
            block_cross_cuboid_hw=dec_cross_cuboid_hw,
            block_cross_shift_hw=dec_cross_shift_hw,
            block_cross_cuboid_strategy=dec_cross_cuboid_strategy,
            block_cross_n_temporal=dec_cross_n_temporal,
            cross_last_n_frames=dec_cross_last_n_frames,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            ffn_drop=ffn_drop,
            upsample_kernel_size=upsample_kernel_size,
            ffn_activation=ffn_activation,
            gated_ffn=gated_ffn,
            norm_layer=norm_layer,
            use_inter_ffn=dec_use_inter_ffn,
            max_temporal_relative=T_in + T_out,
            padding_type=padding_type,
            hierarchical_pos_embed=dec_hierarchical_pos_embed,
            pos_embed_type=pos_embed_type,
            use_self_global=(num_global_vectors > 0) and use_dec_self_global,
            self_update_global=dec_self_update_global,
            use_cross_global=(num_global_vectors > 0) and use_dec_cross_global,
            use_global_vector_ffn=use_global_vector_ffn,
            use_global_self_attn=use_global_self_attn,
            separate_global_qkv=separate_global_qkv,
            global_dim_ratio=global_dim_ratio,
            checkpoint_level=checkpoint_level,
            use_relative_pos=use_relative_pos,
            self_attn_use_final_proj=self_attn_use_final_proj,
            use_first_self_attn=dec_use_first_self_attn,
            attn_linear_init_mode=attn_linear_init_mode,
            ffn_linear_init_mode=ffn_linear_init_mode,
            conv_init_mode=conv_init_mode,
            up_linear_init_mode=down_up_linear_init_mode,
            norm_init_mode=norm_init_mode,
        )
        self.reset_parameters()

    def get_initial_encoder_final_decoder(
        self,
        initial_downsample_type,
        activation,
        initial_downsample_scale,
        initial_downsample_conv_layers,
        final_upsample_conv_layers,
        padding_type,
        initial_downsample_stack_conv_num_layers,
        initial_downsample_stack_conv_dim_list,
        initial_downsample_stack_conv_downscale_list,
        initial_downsample_stack_conv_num_conv_list,
    ):
        T_in, H_in, W_in, C_in = self.input_shape
        T_out, H_out, W_out, C_out = self.target_shape
        self.initial_downsample_type = initial_downsample_type
        if self.initial_downsample_type == "conv":
            if isinstance(initial_downsample_scale, int):
                initial_downsample_scale = (1, initial_downsample_scale, initial_downsample_scale)
            elif len(initial_downsample_scale) == 2:
                initial_downsample_scale = (1, *initial_downsample_scale)
            elif len(initial_downsample_scale) == 3:
                initial_downsample_scale = tuple(initial_downsample_scale)
            else:
                raise NotImplementedError(f"initial_downsample_scale {initial_downsample_scale} format not supported!")
            self.initial_encoder = InitialEncoder(
                dim=C_in,
                out_dim=self.base_units,
                downsample_scale=initial_downsample_scale,
                num_conv_layers=initial_downsample_conv_layers,
                padding_type=padding_type,
                activation=activation,
                conv_init_mode=self.conv_init_mode,
                linear_init_mode=self.down_up_linear_init_mode,
                norm_init_mode=self.norm_init_mode,
            )
            self.final_decoder = FinalDecoder(
                dim=self.base_units,
                target_thw=(T_out, H_out, W_out),
                num_conv_layers=final_upsample_conv_layers,
                activation=activation,
                conv_init_mode=self.conv_init_mode,
                linear_init_mode=self.down_up_linear_init_mode,
                norm_init_mode=self.norm_init_mode,
            )
            new_input_shape = self.initial_encoder.patch_merge.get_out_shape(self.input_shape)
            self.dec_final_proj = nn.Linear(self.base_units, C_out)
        elif self.initial_downsample_type == "stack_conv":
            if initial_downsample_stack_conv_dim_list is None:
                initial_downsample_stack_conv_dim_list = [self.base_units] * initial_downsample_stack_conv_num_layers
            self.initial_encoder = InitialStackPatchMergingEncoder(
                num_merge=initial_downsample_stack_conv_num_layers,
                in_dim=C_in,
                out_dim_list=initial_downsample_stack_conv_dim_list,
                downsample_scale_list=initial_downsample_stack_conv_downscale_list,
                num_conv_per_merge_list=initial_downsample_stack_conv_num_conv_list,
                padding_type=padding_type,
                activation=activation,
                conv_init_mode=self.conv_init_mode,
                linear_init_mode=self.down_up_linear_init_mode,
                norm_init_mode=self.norm_init_mode,
            )
            # self.target_shape gives the correct T_out
            initial_encoder_out_shape_list = self.initial_encoder.get_out_shape_list(self.target_shape)
            dec_target_shape_list, dec_in_dim = FinalStackUpsamplingDecoder.get_init_params(
                enc_input_shape=self.target_shape, enc_out_shape_list=initial_encoder_out_shape_list, large_channel=True
            )
            self.final_decoder = FinalStackUpsamplingDecoder(
                target_shape_list=dec_target_shape_list,
                in_dim=dec_in_dim,
                num_conv_per_up_list=initial_downsample_stack_conv_num_conv_list[::-1],
                activation=activation,
                conv_init_mode=self.conv_init_mode,
                linear_init_mode=self.down_up_linear_init_mode,
                norm_init_mode=self.norm_init_mode,
            )
            self.dec_final_proj = nn.Linear(dec_target_shape_list[-1][-1], C_out)
            new_input_shape = self.initial_encoder.get_out_shape_list(self.input_shape)[-1]
        else:
            raise NotImplementedError
        self.input_shape_after_initial_downsample = new_input_shape
        return new_input_shape

    def reset_parameters(self):
        if self.num_global_vectors > 0:
            nn.init.trunc_normal_(self.init_global_vectors, std=0.02)
        if hasattr(self.initial_encoder, "reset_parameters"):
            self.initial_encoder.reset_parameters()
        else:
            apply_initialization(
                self.initial_encoder,
                conv_mode=self.conv_init_mode,
                linear_mode=self.down_up_linear_init_mode,
                norm_mode=self.norm_init_mode,
            )
        if hasattr(self.final_decoder, "reset_parameters"):
            self.final_decoder.reset_parameters()
        else:
            apply_initialization(
                self.final_decoder,
                conv_mode=self.conv_init_mode,
                linear_mode=self.down_up_linear_init_mode,
                norm_mode=self.norm_init_mode,
            )
        apply_initialization(self.dec_final_proj, linear_mode=self.down_up_linear_init_mode)
        self.encoder.reset_parameters()
        self.enc_pos_embed.reset_parameters()
        self.decoder.reset_parameters()
        self.dec_pos_embed.reset_parameters()
        apply_initialization(self.z_proj, linear_mode="0")

    def get_initial_z(self, final_mem, T_out):
        B = final_mem.shape[0]
        if self.z_init_method == "zeros":
            z_shape = (1, T_out) + final_mem.shape[2:]
            initial_z = torch.zeros(z_shape, dtype=final_mem.dtype, device=final_mem.device)
            initial_z = self.z_proj(self.dec_pos_embed(initial_z)).expand(B, -1, -1, -1, -1)
        elif self.z_init_method == "nearest_interp":
            initial_z = F.interpolate(
                final_mem.permute(0, 4, 1, 2, 3), size=(T_out, final_mem.shape[2], final_mem.shape[3])
            ).permute(0, 2, 3, 4, 1)
            initial_z = self.z_proj(initial_z)
        elif self.z_init_method == "last":
            initial_z = torch.broadcast_to(final_mem[:, -1:, :, :, :], (B, T_out) + final_mem.shape[2:])
            initial_z = self.z_proj(initial_z)
        elif self.z_init_method == "mean":
            initial_z = torch.broadcast_to(final_mem.mean(axis=1, keepdims=True), (B, T_out) + final_mem.shape[2:])
            initial_z = self.z_proj(initial_z)
        else:
            raise NotImplementedError
        return initial_z

    def forward(self, x, verbose=False):
        """``x``: ``(batch, T_in, H, W, C_in)``; returns ``(batch, T_out, H, W, C_out)``."""
        if x.ndim != 5 or tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(
                "CuboidTransformerModel expects input of shape (batch, T, H, W, C) = "
                f"(batch, {', '.join(map(str, self.input_shape))}), got {tuple(x.shape)}."
            )
        B, _, _, _, _ = x.shape
        T_out = self.target_shape[0]
        x = self.initial_encoder(x)
        x = self.enc_pos_embed(x)
        if self.num_global_vectors > 0:
            init_global_vectors = self.init_global_vectors.expand(B, self.num_global_vectors, self.global_dim_ratio * self.base_units)
            mem_l, mem_global_vector_l = self.encoder(x, init_global_vectors)
        else:
            mem_l = self.encoder(x)
        if verbose:
            for i, mem in enumerate(mem_l):
                print(f"mem[{i}].shape = {mem.shape}")
        initial_z = self.get_initial_z(final_mem=mem_l[-1], T_out=T_out)
        if self.num_global_vectors > 0:
            dec_out = self.decoder(initial_z, mem_l, mem_global_vector_l)
        else:
            dec_out = self.decoder(initial_z, mem_l)
        dec_out = self.final_decoder(dec_out)
        out = self.dec_final_proj(dec_out)
        return out


# --------------------------------------------------------------------------------------------------
# PyHazards additions: configuration presets, layout wrappers, checkpoints and builder
# --------------------------------------------------------------------------------------------------

# ``model`` sections of the official configuration files (scripts/cuboid_transformer/...), verbatim.
# ``block_units`` is commented out in every file and falls back to None.
_SEVIR_V1: Dict[str, Any] = {
    "input_shape": [13, 384, 384, 1],
    "target_shape": [12, 384, 384, 1],
    "base_units": 128,
    "block_units": None,
    "scale_alpha": 1.0,
    "enc_depth": [1, 1],
    "dec_depth": [1, 1],
    "enc_use_inter_ffn": True,
    "dec_use_inter_ffn": True,
    "dec_hierarchical_pos_embed": False,
    "downsample": 2,
    "downsample_type": "patch_merge",
    "upsample_type": "upsample",
    "num_global_vectors": 8,
    "use_dec_self_global": False,
    "dec_self_update_global": True,
    "use_dec_cross_global": False,
    "use_global_vector_ffn": False,
    "use_global_self_attn": True,
    "separate_global_qkv": True,
    "global_dim_ratio": 1,
    "self_pattern": "axial",
    "cross_self_pattern": "axial",
    "cross_pattern": "cross_1x1",
    "dec_cross_last_n_frames": None,
    "attn_drop": 0.1,
    "proj_drop": 0.1,
    "ffn_drop": 0.1,
    "num_heads": 4,
    "ffn_activation": "gelu",
    "gated_ffn": False,
    "norm_layer": "layer_norm",
    "padding_type": "zeros",
    "pos_embed_type": "t+h+w",
    "use_relative_pos": True,
    "self_attn_use_final_proj": True,
    "dec_use_first_self_attn": False,
    "z_init_method": "zeros",
    "checkpoint_level": 0,
    "initial_downsample_type": "stack_conv",
    "initial_downsample_activation": "leaky",
    "initial_downsample_stack_conv_num_layers": 3,
    "initial_downsample_stack_conv_dim_list": [16, 64, 128],
    "initial_downsample_stack_conv_downscale_list": [3, 2, 2],
    "initial_downsample_stack_conv_num_conv_list": [2, 2, 2],
    "attn_linear_init_mode": "0",
    "ffn_linear_init_mode": "0",
    "conv_init_mode": "0",
    "down_up_linear_init_mode": "0",
    "norm_init_mode": "0",
}

_SEVIR_LR: Dict[str, Any] = dict(
    _SEVIR_V1,
    input_shape=[7, 128, 128, 1],
    target_shape=[6, 128, 128, 1],
    base_units=64,
    num_global_vectors=0,
    use_global_self_attn=False,
    separate_global_qkv=False,
    initial_downsample_stack_conv_dim_list=[4, 16, 64],
    initial_downsample_stack_conv_downscale_list=[1, 2, 2],
)

_MOVING_MNIST: Dict[str, Any] = {
    "input_shape": [10, 64, 64, 1],
    "target_shape": [10, 64, 64, 1],
    "base_units": 64,
    "block_units": None,
    "scale_alpha": 1.0,
    "enc_depth": [4, 4],
    "dec_depth": [4, 4],
    "enc_use_inter_ffn": True,
    "dec_use_inter_ffn": True,
    "dec_hierarchical_pos_embed": False,
    "downsample": 2,
    "downsample_type": "patch_merge",
    "upsample_type": "upsample",
    "num_global_vectors": 0,
    "use_dec_self_global": False,
    "dec_self_update_global": True,
    "use_dec_cross_global": False,
    "use_global_vector_ffn": False,
    "use_global_self_attn": False,
    "separate_global_qkv": False,
    "global_dim_ratio": 1,
    "self_pattern": "axial",
    "cross_self_pattern": "axial",
    "cross_pattern": "cross_1x1",
    "dec_cross_last_n_frames": None,
    "attn_drop": 0.1,
    "proj_drop": 0.1,
    "ffn_drop": 0.1,
    "num_heads": 4,
    "ffn_activation": "gelu",
    "gated_ffn": False,
    "norm_layer": "layer_norm",
    "padding_type": "zeros",
    "pos_embed_type": "t+hw",
    "use_relative_pos": True,
    "self_attn_use_final_proj": True,
    "dec_use_first_self_attn": False,
    "z_init_method": "zeros",
    "initial_downsample_type": "conv",
    "initial_downsample_activation": "leaky",
    "initial_downsample_scale": 2,
    "initial_downsample_conv_layers": 2,
    "final_upsample_conv_layers": 1,
    "checkpoint_level": 0,
    "attn_linear_init_mode": "0",
    "ffn_linear_init_mode": "0",
    "conv_init_mode": "0",
    "down_up_linear_init_mode": "0",
    "norm_init_mode": "0",
}

_ENSO_V1: Dict[str, Any] = dict(
    _MOVING_MNIST,
    input_shape=[12, 24, 48, 1],
    target_shape=[14, 24, 48, 1],
    enc_depth=[1, 1],
    dec_depth=[1, 1],
    pos_embed_type="t+h+w",
    initial_downsample_scale=[1, 1, 2],
)

EARTHFORMER_CONFIGS: Dict[str, Dict[str, Any]] = {
    # scripts/cuboid_transformer/sevir/earthformer_sevir_v1.yaml (= cfg_sevir.yaml): 8,659,677 params,
    # the configuration of the released SEVIR checkpoint.
    "sevir": _SEVIR_V1,
    # scripts/cuboid_transformer/sevir/cfg_sevirlr.yaml (SEVIR-LR, 128x128): 1,505,069 params.
    "sevir_lr": _SEVIR_LR,
    # scripts/cuboid_transformer/moving_mnist/cfg.yaml (nbody/cfg.yaml is identical): 6,702,109 params.
    "moving_mnist": _MOVING_MNIST,
    "nbody": _MOVING_MNIST,
    # scripts/cuboid_transformer/enso/earthformer_enso_v1.yaml (= cfg.yaml): 1,394,325 params, the
    # configuration of the released ICAR-ENSO checkpoint.
    "enso": _ENSO_V1,
}

# YAML pattern keys and the CuboidTransformerModel arguments the training scripts expand them into.
_PATTERN_KEYS = {
    "self_pattern": "enc_attn_patterns",
    "cross_self_pattern": "dec_self_attn_patterns",
    "cross_pattern": "dec_cross_attn_patterns",
}
_MODEL_ARGS = frozenset(name for name in inspect.signature(CuboidTransformerModel.__init__).parameters if name != "self")


def earthformer_config(config: str = "sevir", **overrides: Any) -> Dict[str, Any]:
    """Keyword arguments of :class:`CuboidTransformerModel` for an official configuration.

    ``config`` names a preset in :data:`EARTHFORMER_CONFIGS`; ``overrides`` replace entries of
    its ``model`` section (YAML keys, including ``self_pattern`` / ``cross_self_pattern`` /
    ``cross_pattern``) or set any other ``CuboidTransformerModel`` argument. Patterns are expanded
    to one per hierarchy exactly as the official training scripts do.
    """
    if config not in EARTHFORMER_CONFIGS:
        raise ValueError(f"Unknown Earthformer config {config!r}; expected one of {sorted(EARTHFORMER_CONFIGS)}.")
    unknown = sorted(set(overrides) - _MODEL_ARGS - set(_PATTERN_KEYS))
    if unknown:
        raise ValueError(f"Unknown Earthformer arguments: {unknown}.")
    cfg = dict(EARTHFORMER_CONFIGS[config])
    cfg.update(overrides)
    num_blocks = len(cfg["enc_depth"])
    kwargs = {key: value for key, value in cfg.items() if key not in _PATTERN_KEYS}
    for yaml_key, arg in _PATTERN_KEYS.items():
        if arg not in overrides:  # an explicit enc_attn_patterns etc. wins over the YAML key
            pattern = cfg[yaml_key]
            kwargs[arg] = [pattern] * num_blocks if isinstance(pattern, str) else list(pattern)
    return kwargs


class Earthformer(CuboidTransformerModel):
    """:class:`CuboidTransformerModel` with the PyHazards frame layout.

    Input ``(batch, T_in, C_in, H, W)``, output ``(batch, T_out, C_out, H, W)``. Parameters are those
    of ``CuboidTransformerModel`` (no prefix), so official state dicts load with ``strict=True``.
    """

    def _check_frames(self, x: torch.Tensor) -> None:
        T_in, H, W, C_in = self.input_shape
        if x.ndim != 5 or tuple(x.shape[1:]) != (T_in, C_in, H, W):
            raise ValueError(
                f"{type(self).__name__} expects input of shape (batch, time, channels, height, width) = "
                f"(batch, {T_in}, {C_in}, {H}, {W}), got {tuple(x.shape)}."
            )

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        self._check_frames(x)
        out = super().forward(x.permute(0, 1, 3, 4, 2), verbose=verbose)
        return out.permute(0, 1, 4, 2, 3)


class EarthformerSegmenter(Earthformer):
    """PyHazards next-fire-mask adaptation: one predicted frame returned as logits.

    The network is :class:`CuboidTransformerModel` with ``target_shape = (1, H, W, out_channels)``;
    ``forward`` maps ``(batch, T_in, C_in, H, W)`` to ``(batch, out_channels, H, W)``.
    """

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        return super().forward(x, verbose=verbose)[:, 0]


# Official checkpoints (README "Pretrained Weights"): state dicts of CuboidTransformerModel.
# The S3 bucket answered HTTP 403 on 2026-10-07; the Internet Archive kept byte-identical copies.
EARTHFORMER_CHECKPOINTS: Dict[str, Dict[str, Any]] = {
    "sevir": {
        "config": "sevir",
        "urls": (
            "https://earthformer.s3.amazonaws.com/pretrained_checkpoints/earthformer_sevir.pt",
            "https://web.archive.org/web/20250517094035id_/https://earthformer.s3.amazonaws.com/pretrained_checkpoints/earthformer_sevir.pt",
        ),
        "sha256": "2795b23a32cc03ebd77a6c6b57ebbe133070ff9cae1b365d82bb363fd6967215",
    },
    "icarenso2021": {
        "config": "enso",
        "urls": (
            "https://earthformer.s3.amazonaws.com/pretrained_checkpoints/earthformer_icarenso2021.pt",
            "https://web.archive.org/web/20250517094035id_/https://earthformer.s3.amazonaws.com/pretrained_checkpoints/earthformer_icarenso2021.pt",
        ),
        "sha256": "3428356328ec02291461f737684ea597a3126af86a09fe173b75ed54ffe1511f",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def earthformer_checkpoint_path(name: str) -> Path:
    """Local path of an official checkpoint, downloaded into the torch hub cache on first use."""
    if name not in EARTHFORMER_CHECKPOINTS:
        raise ValueError(f"Unknown Earthformer checkpoint {name!r}; expected one of {sorted(EARTHFORMER_CHECKPOINTS)}.")
    spec = EARTHFORMER_CHECKPOINTS[name]
    path = Path(torch.hub.get_dir()) / "checkpoints" / f"earthformer_{name}.pt"
    if path.exists() and _sha256(path) == spec["sha256"]:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in spec["urls"]:
        try:
            torch.hub.download_url_to_file(url, str(path), hash_prefix=spec["sha256"], progress=False)
            return path
        except (urllib.error.URLError, RuntimeError) as error:  # HTTP 403 or hash mismatch
            errors.append(f"{url}: {error}")
    raise RuntimeError("Could not download the Earthformer checkpoint:\n" + "\n".join(errors))


def load_earthformer_state_dict(source: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """State dict of an official checkpoint name (``"sevir"``, ``"icarenso2021"``) or a local file."""
    path = earthformer_checkpoint_path(str(source)) if str(source) in EARTHFORMER_CHECKPOINTS else Path(source)
    return dict(torch.load(path, map_location="cpu", weights_only=True))


def earthformer_builder(
    task: str,
    config: str = "sevir",
    in_channels: Optional[int] = None,
    out_channels: Optional[int] = None,
    history: Optional[int] = None,
    horizon: Optional[int] = None,
    img_size: Optional[Union[int, Sequence[int]]] = None,
    pretrained: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> nn.Module:
    """Earthformer at an official configuration (``config``: sevir, sevir_lr, moving_mnist, nbody, enso).

    ``task="forecasting"``: frames to frames, ``(batch, history, in_channels, H, W)`` ->
    ``(batch, horizon, out_channels, H, W)``. ``task="segmentation"``: the PyHazards next-fire-mask
    adaptation, ``(batch, history, in_channels, H, W)`` -> ``(batch, out_channels, H, W)`` logits
    with one predicted frame and ``out_channels=1`` by default. ``history``, ``horizon``,
    ``in_channels``, ``out_channels`` and ``img_size`` (int or (H, W)) override the preset's
    ``input_shape`` / ``target_shape``; other keyword arguments override preset entries
    (see :func:`earthformer_config`). ``pretrained`` is ``"sevir"``, ``"icarenso2021"`` or a path to
    a ``CuboidTransformerModel`` state dict, loaded with ``strict=True``.
    """
    kwargs.pop("name", None)
    task = task.lower()
    if task not in ("forecasting", "segmentation"):
        raise ValueError(f"earthformer supports task='forecasting' or 'segmentation', got {task!r}.")
    if config not in EARTHFORMER_CONFIGS:
        raise ValueError(f"Unknown Earthformer config {config!r}; expected one of {sorted(EARTHFORMER_CONFIGS)}.")
    for key in ("input_shape", "target_shape"):
        if key in kwargs:
            raise ValueError(f"Set history, horizon, in_channels, out_channels and img_size instead of {key}.")
    preset = EARTHFORMER_CONFIGS[config]
    T_in, H, W, C_in = preset["input_shape"]
    T_out, _, _, C_out = preset["target_shape"]
    if task == "segmentation":
        if horizon not in (None, 1):
            raise ValueError(f"task='segmentation' predicts one frame; got horizon={horizon}.")
        T_out, C_out = 1, 1
    if img_size is not None:
        H, W = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
    T_in = T_in if history is None else history
    T_out = T_out if horizon is None else horizon
    C_in = C_in if in_channels is None else in_channels
    C_out = C_out if out_channels is None else out_channels
    for label, value in (("history", T_in), ("horizon", T_out), ("in_channels", C_in), ("out_channels", C_out), ("height", H), ("width", W)):
        if int(value) <= 0:
            raise ValueError(f"{label} must be positive, got {value}.")

    model_kwargs = earthformer_config(config, input_shape=[T_in, H, W, C_in], target_shape=[T_out, H, W, C_out], **kwargs)
    model_cls = EarthformerSegmenter if task == "segmentation" else Earthformer
    model = model_cls(**model_kwargs)
    if pretrained is not None:
        name = str(pretrained)
        if name in EARTHFORMER_CHECKPOINTS:
            expected = EARTHFORMER_CHECKPOINTS[name]["config"]
            preset_shapes = (EARTHFORMER_CONFIGS[expected]["input_shape"], EARTHFORMER_CONFIGS[expected]["target_shape"])
            if task != "forecasting" or config != expected or (model.input_shape, model.target_shape) != tuple(map(tuple, preset_shapes)):
                raise ValueError(
                    f"The {name!r} checkpoint needs task='forecasting', config={expected!r} and the preset's "
                    "input and target shapes."
                )
        model.load_state_dict(load_earthformer_state_dict(pretrained), strict=True)
    return model


__all__ = [
    "EARTHFORMER_CHECKPOINTS",
    "EARTHFORMER_CONFIGS",
    "CuboidCrossAttentionLayer",
    "CuboidSelfAttentionLayer",
    "CuboidTransformerDecoder",
    "CuboidTransformerEncoder",
    "CuboidTransformerModel",
    "Earthformer",
    "EarthformerSegmenter",
    "earthformer_builder",
    "earthformer_checkpoint_path",
    "earthformer_config",
    "load_earthformer_state_dict",
]
