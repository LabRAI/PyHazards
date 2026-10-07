"""Prithvi-EO-2.0 masked-autoencoder ViT and the TerraTorch segmentation recipe, in plain PyTorch.

Ported from the reference code, keeping its module and parameter names so the official checkpoints
load with ``strict=True``:

- ``prithvi_mae.py`` shipped in the Hugging Face repository ``ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL``
  (revision 63adbd39; Apache License 2.0, Copyright (c) IBM Corp. 2024): ``PatchEmbed``,
  ``TemporalEncoder``, ``LocationEncoder``, ``PrithviViT``, ``MAEDecoder``, ``PrithviMAE`` and the
  sin/cos position-embedding helpers. The timm ``Block``, ``Attention``, ``Mlp`` and ``DropPath`` it
  imports (timm 1.0.15, Apache License 2.0, Copyright (c) Ross Wightman) are re-implemented here with
  the same attribute names and initialisation order.
- TerraTorch 0.99.8 (``terratorch/models``; Apache License 2.0, Copyright contributors to the
  Terratorch project): necks ``SelectIndices``, ``ReshapeTokensToImage``,
  ``LearnedInterpolateToPyramidal``, the ``UNetDecoder`` wrapper, ``SegmentationHead`` and the
  ``PixelWiseModel`` forward pass (padding, bilinear rescale, crop).
- segmentation_models_pytorch 0.4.0 (``decoders/unet/decoder.py``, ``base/modules.py``,
  ``base/initialization.py``; MIT License, Copyright (c) 2019 Pavel Iakubovskii): ``UnetDecoder``,
  ``DecoderBlock``, ``Conv2dReLU`` and ``initialize_decoder``.

TerraTorch 0.99.8 with smp 0.4.0 is the stack the official Prithvi-EO-2.0-300M-BurnScars checkpoint was
trained with (and the release its demo pins). Later releases (smp >= 0.5) resize the last U-Net decoder
block to the skip resolution instead of upsampling it by two, which changes this model's outputs; this
port keeps the training-time behaviour.

Pretrained weights are not bundled. :func:`load_pretrained_state_dict` downloads a pinned file from
Hugging Face over plain HTTPS on first use, verifies its sha256 and caches it under
``torch.hub.get_dir()/checkpoints`` (or ``cache_dir``).
"""

from __future__ import annotations

import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# HLS bands the Prithvi-EO models were pretrained on, in input order (HLS B02, B03, B04, B05/B8A,
# B06/B11, B07/B12), and the pretraining normalisation (HLS digital numbers, reflectance x 10000).
PRITHVI_BANDS = ("BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2")
PRITHVI_EO_V2_MEAN = (1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0)
PRITHVI_EO_V2_STD = (2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0)

# Encoder configurations of the TerraTorch backbone registry (``prithvi_cfgs``) and the SelectIndices
# used by the official Prithvi-EO-2.0 fine-tuning configs for each size.
PRITHVI_EO_V2_CONFIGS = {
    "300m": dict(embed_dim=1024, depth=24, num_heads=16, patch_size=(1, 16, 16), select_indices=(5, 11, 17, 23)),
    "600m": dict(embed_dim=1280, depth=32, num_heads=16, patch_size=(1, 14, 14), select_indices=(7, 15, 23, 31)),
}


# --------------------------------------------------------------------------------------------------
# Position embeddings (prithvi_mae.py)
# --------------------------------------------------------------------------------------------------


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """1D sin/cos embedding of ``pos`` (M,) -> (M, embed_dim), computed in float64 like the reference."""
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_3d_sincos_pos_embed(embed_dim: int, grid_size: Sequence[int], add_cls_token: bool = False) -> np.ndarray:
    """3D sin/cos embedding over a (T, H, W) token grid: 6/16 width, 6/16 height, 4/16 time channels."""
    if embed_dim % 16 != 0:
        raise ValueError(f"embed_dim must be divisible by 16, got {embed_dim}")
    t_size, h_size, w_size = grid_size
    w_embed_dim = embed_dim // 16 * 6
    h_embed_dim = embed_dim // 16 * 6
    t_embed_dim = embed_dim // 16 * 4
    w_pos_embed = get_1d_sincos_pos_embed_from_grid(w_embed_dim, np.arange(w_size))
    h_pos_embed = get_1d_sincos_pos_embed_from_grid(h_embed_dim, np.arange(h_size))
    t_pos_embed = get_1d_sincos_pos_embed_from_grid(t_embed_dim, np.arange(t_size))
    w_pos_embed = np.tile(w_pos_embed, (t_size * h_size, 1))
    h_pos_embed = np.tile(np.repeat(h_pos_embed, w_size, axis=0), (t_size, 1))
    t_pos_embed = np.repeat(t_pos_embed, h_size * w_size, axis=0)
    pos_embed = np.concatenate((w_pos_embed, h_pos_embed, t_pos_embed), axis=1)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def _get_1d_sincos_embed_from_grid_torch(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """Torch version of :func:`get_1d_sincos_pos_embed_from_grid` for float ``pos``."""
    omega = torch.arange(embed_dim // 2, dtype=pos.dtype).to(pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def _interpolate_pos_encoding(
    pos_embed: torch.Tensor,
    grid_size: Sequence[int],
    patch_size: Sequence[int],
    shape: Sequence[int],
    embed_dim: int,
) -> torch.Tensor:
    """Recompute the sin/cos table for a new frame count and bicubically resize it to a new grid."""
    t, h, w = shape
    t_patches = t // patch_size[0]
    h_patches = h // patch_size[1]
    w_patches = w // patch_size[2]
    if [t_patches, h_patches, w_patches] == list(grid_size):
        return pos_embed
    if t_patches != grid_size[0]:
        new_grid_size = (t_patches, *grid_size[1:])
        new_pos_embed = get_3d_sincos_pos_embed(pos_embed.shape[-1], new_grid_size, add_cls_token=True)
        new_pos_embed = torch.from_numpy(new_pos_embed).float().unsqueeze(0).to(pos_embed.device)
    else:
        new_grid_size = grid_size
        new_pos_embed = pos_embed
    class_pos_embed, patch_pos_embed = new_pos_embed[:, :1], new_pos_embed[:, 1:]
    patch_pos_embed = patch_pos_embed.reshape(*new_grid_size, embed_dim).permute(0, 3, 1, 2)
    patch_pos_embed = F.interpolate(patch_pos_embed, size=(h_patches, w_patches), mode="bicubic", align_corners=True)
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    return torch.cat((class_pos_embed, patch_pos_embed), dim=1)


def _to_2tuple(value: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(value, (tuple, list)):
        return tuple(value)  # type: ignore[return-value]
    return (value, value)


def _as_float(coords: torch.Tensor) -> torch.Tensor:
    return coords if coords.is_floating_point() else coords.float()


# --------------------------------------------------------------------------------------------------
# Embeddings and transformer block (prithvi_mae.py + timm 1.0.15)
# --------------------------------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """3D patch embedding: a Conv3d whose kernel and stride equal the (t, h, w) patch size."""

    def __init__(
        self,
        input_size: Tuple[int, int, int] = (1, 224, 224),
        patch_size: Tuple[int, int, int] = (1, 16, 16),
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[type] = None,
        flatten: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.grid_size = [s // p for s, p in zip(self.input_size, self.patch_size)]
        if self.grid_size < [1, 1, 1]:
            raise ValueError("Patch size is bigger than input size.")
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, T, H, W = x.shape
        if T / self.patch_size[0] % 1 or H / self.patch_size[1] % 1 or W / self.patch_size[2] % 1:
            warnings.warn(
                f"Input {tuple(x.shape[-3:])} is not divisible by patch size {self.patch_size}. "
                "The border will be ignored, add backbone_padding for pixel-wise tasks."
            )
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


class TemporalEncoder(nn.Module):
    """Sin/cos encoding of (year, day of year), half of the channels each, times a scale."""

    def __init__(self, embed_dim: int, trainable_scale: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.year_embed_dim = embed_dim // 2
        self.julian_day_embed_dim = embed_dim - self.year_embed_dim
        if trainable_scale:
            self.scale = nn.Parameter(torch.full((1,), 0.1))
        else:
            self.register_buffer("scale", torch.ones(1))

    def forward(self, temporal_coords: torch.Tensor, tokens_per_frame: Optional[int] = None) -> torch.Tensor:
        shape = temporal_coords.shape[:2] + (-1,)
        year = _get_1d_sincos_embed_from_grid_torch(self.year_embed_dim, temporal_coords[:, :, 0].flatten()).reshape(
            shape
        )
        julian_day = _get_1d_sincos_embed_from_grid_torch(
            self.julian_day_embed_dim, temporal_coords[:, :, 1].flatten()
        ).reshape(shape)
        embedding = self.scale * torch.cat([year, julian_day], dim=-1)
        if tokens_per_frame is not None:
            embedding = torch.repeat_interleave(embedding, tokens_per_frame, dim=1)
        return embedding


class LocationEncoder(nn.Module):
    """Sin/cos encoding of (latitude, longitude), half of the channels each, times a scale."""

    def __init__(self, embed_dim: int, trainable_scale: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.lat_embed_dim = embed_dim // 2
        self.lon_embed_dim = embed_dim - self.lat_embed_dim
        if trainable_scale:
            self.scale = nn.Parameter(torch.full((1,), 0.1))
        else:
            self.register_buffer("scale", torch.ones(1))

    def forward(self, location_coords: torch.Tensor) -> torch.Tensor:
        shape = location_coords.shape[:1] + (1, -1)
        lat = _get_1d_sincos_embed_from_grid_torch(self.lat_embed_dim, location_coords[:, 0].flatten()).reshape(shape)
        lon = _get_1d_sincos_embed_from_grid_torch(self.lon_embed_dim, location_coords[:, 1].flatten()).reshape(shape)
        return self.scale * torch.cat([lat, lon], dim=-1)


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    """timm multi-head self-attention with a fused qkv projection (scaled_dot_product_attention path)."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.norm = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.norm(self.drop1(self.act(self.fc1(x))))))


class Block(nn.Module):
    """Pre-norm transformer block (timm ``vision_transformer.Block`` without layer scale)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_path: float = 0.0,
        norm_layer: type = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.ls1 = nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.ls2 = nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# --------------------------------------------------------------------------------------------------
# Encoder, MAE decoder and MAE (prithvi_mae.py)
# --------------------------------------------------------------------------------------------------


class PrithviViT(nn.Module):
    """Prithvi ViT encoder: 3D patch embedding, fixed 3D sin/cos positions, optional time/location codes.

    Input is ``(B, C, T, H, W)``; ``(B, C, H, W)`` is accepted when ``num_frames == 1``. Optional
    ``temporal_coords`` are ``(B, T, 2)`` (year, day of year) and ``location_coords`` ``(B, 2)``
    (latitude, longitude); they are used only when ``coords_encoding`` enables them.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int, int]] = (1, 16, 16),
        num_frames: int = 1,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: type = nn.LayerNorm,
        coords_encoding: Optional[Sequence[str]] = None,
        coords_scale_learn: bool = False,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.img_size = _to_2tuple(img_size)
        if isinstance(patch_size, int):
            patch_size = (1, patch_size, patch_size)
        patch_size = tuple(patch_size)
        self.patch_embed = PatchEmbed(
            input_size=(num_frames,) + self.img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.out_channels = [embed_dim * self.patch_embed.grid_size[0]] * depth

        coords_encoding = list(coords_encoding or [])
        self.temporal_encoding = "time" in coords_encoding
        self.location_encoding = "location" in coords_encoding
        if self.temporal_encoding:
            if patch_size[0] != 1:
                raise ValueError(f"With temporal encoding, patch_size[0] must be 1, received {patch_size[0]}")
            self.temporal_embed_enc = TemporalEncoder(embed_dim, coords_scale_learn)
        if self.location_encoding:
            self.location_embed_enc = LocationEncoder(embed_dim, coords_scale_learn)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer("pos_embed", torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_path=drop_path)
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size, add_cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.normal_(self.cls_token, std=0.02)
        self.apply(_init_weights)

    def _validate(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and self.patch_embed.input_size[0] == 1:
            x = x.unsqueeze(2)
        if x.ndim != 5:
            layout = "(B, C, T, H, W)" + (" or (B, C, H, W)" if self.num_frames == 1 else "")
            raise ValueError(f"PrithviViT expects input of shape {layout}, got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_chans:
            raise ValueError(
                f"PrithviViT expects {self.in_chans} channels in dim 1 of (B, C, T, H, W), got shape {tuple(x.shape)}"
            )
        p = self.patch_embed.patch_size
        if x.shape[2] < p[0] or x.shape[3] < p[1] or x.shape[4] < p[2]:
            raise ValueError(f"Input shape {tuple(x.shape)} is smaller than the patch size {p}")
        return x

    def _add_coords(
        self,
        x: torch.Tensor,
        temporal_coords: Optional[torch.Tensor],
        location_coords: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch, num_tokens = x.shape[0], x.shape[1]
        if self.temporal_encoding and temporal_coords is not None:
            temporal_coords = _as_float(temporal_coords)
            num_tokens_per_frame = num_tokens // self.num_frames
            if (
                temporal_coords.ndim != 3
                or temporal_coords.shape[0] != batch
                or temporal_coords.shape[2] != 2
                or temporal_coords.shape[1] * num_tokens_per_frame != num_tokens
            ):
                raise ValueError(
                    "temporal_coords must have shape (B, T, 2) with (year, day of year) per frame and "
                    f"T = num_frames = {self.num_frames}; got shape {tuple(temporal_coords.shape)}"
                )
            x = x + self.temporal_embed_enc(temporal_coords, num_tokens_per_frame)
        if self.location_encoding and location_coords is not None:
            location_coords = _as_float(location_coords)
            if location_coords.ndim != 2 or location_coords.shape != (batch, 2):
                raise ValueError(
                    "location_coords must have shape (B, 2) with (latitude, longitude); "
                    f"got shape {tuple(location_coords.shape)}"
                )
            x = x + self.location_embed_enc(location_coords)
        return x

    def random_masking(self, sequence: torch.Tensor, mask_ratio: float, noise: Optional[torch.Tensor] = None):
        """Per-sample random masking by argsort of uniform noise (MAE)."""
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - mask_ratio))
        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)
        ids_shuffle = torch.argsort(noise, dim=1).to(sequence.device)
        ids_restore = torch.argsort(ids_shuffle, dim=1).to(sequence.device)
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return sequence_unmasked, mask, ids_restore

    def interpolate_pos_encoding(self, sample_shape: Sequence[int]) -> torch.Tensor:
        return _interpolate_pos_encoding(
            pos_embed=self.pos_embed,
            grid_size=self.patch_embed.grid_size,
            patch_size=self.patch_embed.patch_size,
            shape=sample_shape,
            embed_dim=self.embed_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        temporal_coords: Optional[torch.Tensor] = None,
        location_coords: Optional[torch.Tensor] = None,
        mask_ratio: float = 0.75,
    ):
        """MAE encoder pass: returns (latent of the kept tokens with cls token, mask, ids_restore)."""
        x = self._validate(x)
        sample_shape = x.shape[-3:]
        x = self.patch_embed(x)
        pos_embed = self.interpolate_pos_encoding(sample_shape)
        x = x + pos_embed[:, 1:, :]
        x = self._add_coords(x, temporal_coords, location_coords)
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_token = self.cls_token + pos_embed[:, :1, :]
        x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_features(
        self,
        x: torch.Tensor,
        temporal_coords: Optional[torch.Tensor] = None,
        location_coords: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """All tokens after every block, ``(B, 1 + T*h*w, D)`` each; only the last one is layer-normed."""
        x = self._validate(x)
        sample_shape = x.shape[-3:]
        x = self.patch_embed(x)
        pos_embed = self.interpolate_pos_encoding(sample_shape)
        x = x + pos_embed[:, 1:, :]
        x = self._add_coords(x, temporal_coords, location_coords)
        cls_token = self.cls_token + pos_embed[:, :1, :]
        x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        out = []
        for block in self.blocks:
            x = block(x)
            out.append(x.clone())
        out[-1] = self.norm(x)
        return out


class MAEDecoder(nn.Module):
    """Transformer decoder of the Prithvi MAE (reconstructs masked patches)."""

    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int, int]] = (1, 16, 16),
        grid_size: Sequence[int] = (3, 14, 14),
        in_chans: int = 3,
        encoder_embed_dim: int = 1024,
        decoder_embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: type = nn.LayerNorm,
        coords_encoding: Optional[Sequence[str]] = None,
        coords_scale_learn: bool = False,
    ):
        super().__init__()
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed_dim = decoder_embed_dim
        self.grid_size = list(grid_size)
        if isinstance(patch_size, int):
            patch_size = (1, patch_size, patch_size)
        self.patch_size = tuple(patch_size)
        self.num_frames = self.grid_size[0] * self.patch_size[0]
        num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        coords_encoding = list(coords_encoding or [])
        self.temporal_encoding = "time" in coords_encoding
        self.location_encoding = "location" in coords_encoding
        if self.temporal_encoding:
            self.temporal_embed_dec = TemporalEncoder(decoder_embed_dim, coords_scale_learn)
        if self.location_encoding:
            self.location_embed_dec = LocationEncoder(decoder_embed_dim, coords_scale_learn)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.register_buffer("decoder_pos_embed", torch.zeros(1, num_patches + 1, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList(
            [Block(decoder_embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(depth)]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, self.patch_size[0] * self.patch_size[1] * self.patch_size[2] * in_chans, bias=True
        )
        self.initialize_weights()

    def initialize_weights(self) -> None:
        decoder_pos_embed = get_3d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], self.grid_size, add_cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(_init_weights)

    def interpolate_pos_encoding(self, sample_shape: Sequence[int]) -> torch.Tensor:
        return _interpolate_pos_encoding(
            pos_embed=self.decoder_pos_embed,
            grid_size=self.grid_size,
            patch_size=self.patch_size,
            shape=sample_shape,
            embed_dim=self.decoder_embed_dim,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        ids_restore: torch.Tensor,
        temporal_coords: Optional[torch.Tensor] = None,
        location_coords: Optional[torch.Tensor] = None,
        input_size: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        x = self.decoder_embed(hidden_states)
        cls_token = x[:, :1, :]
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]).to(x.device))
        decoder_pos_embed = self.interpolate_pos_encoding(input_size[-3:])
        cls_token = cls_token + decoder_pos_embed[:, :1, :]
        x = x + decoder_pos_embed[:, 1:, :]
        if self.temporal_encoding and temporal_coords is not None:
            num_tokens_per_frame = x.shape[1] // self.num_frames
            x = x + self.temporal_embed_dec(_as_float(temporal_coords), num_tokens_per_frame)
        if self.location_encoding and location_coords is not None:
            x = x + self.location_embed_dec(_as_float(location_coords))
        x = torch.cat([cls_token, x], dim=1)
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        pred = self.decoder_pred(x)
        return pred[:, 1:, :]


class PrithviMAE(nn.Module):
    """Prithvi masked autoencoder (encoder + decoder); the released pretraining checkpoints load strictly."""

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int, int]] = (1, 16, 16),
        num_frames: int = 4,
        in_chans: int = 6,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: type = nn.LayerNorm,
        norm_pix_loss: bool = False,
        coords_encoding: Optional[Sequence[str]] = None,
        coords_scale_learn: bool = False,
        drop_path: float = 0.0,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        self.encoder = PrithviViT(
            img_size=img_size,
            num_frames=num_frames,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            coords_encoding=coords_encoding,
            coords_scale_learn=coords_scale_learn,
            drop_path=drop_path,
        )
        self.decoder = MAEDecoder(
            patch_size=patch_size,
            grid_size=self.encoder.patch_embed.grid_size,
            in_chans=in_chans,
            encoder_embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            coords_encoding=coords_encoding,
            coords_scale_learn=coords_scale_learn,
        )
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.out_channels = self.encoder.out_channels

    def patchify(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(B, C, T, H, W) -> (B, num_patches, pt*ph*pw*C)."""
        s, p, q = self.encoder.patch_embed.patch_size
        B, C, T, H, W = pixel_values.shape
        x = pixel_values.reshape(B, C, T // s, s, H // p, p, W // q, q)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)
        return x.reshape(B, (T // s) * (H // p) * (W // q), s * p * q * C)

    def unpatchify(self, patchified_pixel_values: torch.Tensor, image_size: Optional[Tuple[int, int]] = None):
        """(B, num_patches, pt*ph*pw*C) -> (B, C, T, H, W)."""
        s, p, q = self.encoder.patch_embed.patch_size
        image_size = _to_2tuple(image_size) if image_size is not None else self.encoder.img_size
        h, w = image_size[0] // p, image_size[1] // q
        C = self.encoder.in_chans
        B, L, _ = patchified_pixel_values.shape
        t = L // (h * w)
        x = patchified_pixel_values.reshape(B, t, h, w, s, p, q, C)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return x.reshape(B, C, t * s, h * p, w * q)

    def forward_loss(self, pixel_values: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target = self.patchify(pixel_values)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        loss = ((pred - target) ** 2).mean(dim=-1)
        return (loss * mask).sum() / mask.sum()

    def forward(
        self,
        pixel_values: torch.Tensor,
        temporal_coords: Optional[torch.Tensor] = None,
        location_coords: Optional[torch.Tensor] = None,
        mask_ratio: Optional[float] = None,
    ):
        """Returns (reconstruction loss on masked patches, predicted patches, mask)."""
        if pixel_values.ndim == 4 and self.encoder.patch_embed.input_size[0] == 1:
            pixel_values = pixel_values.unsqueeze(2)
        mask_ratio = mask_ratio or self.mask_ratio
        latent, mask, ids_restore = self.encoder(pixel_values, temporal_coords, location_coords, mask_ratio)
        pred = self.decoder(latent, ids_restore, temporal_coords, location_coords, input_size=pixel_values.shape)
        loss = self.forward_loss(pixel_values, pred, mask)
        return loss, pred, mask

    def forward_features(
        self,
        x: torch.Tensor,
        temporal_coords: Optional[torch.Tensor] = None,
        location_coords: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        return self.encoder.forward_features(x, temporal_coords, location_coords)


# --------------------------------------------------------------------------------------------------
# TerraTorch necks, smp U-Net decoder, segmentation head and pixel-wise model
# --------------------------------------------------------------------------------------------------


class SelectIndices(nn.Module):
    """Keep the encoder outputs at ``indices``."""

    def __init__(self, indices: Sequence[int]):
        super().__init__()
        self.indices = list(indices)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        return [features[i] for i in self.indices]


class ReshapeTokensToImage(nn.Module):
    """Tokens ``(B, 1 + t*h*w, E)`` -> feature maps ``(B, t*E, h, w)`` (time folded into channels)."""

    def __init__(self, remove_cls_token: bool = True, effective_time_dim: int = 1):
        super().__init__()
        self.remove_cls_token = remove_cls_token
        self.effective_time_dim = effective_time_dim

    def forward(self, features: List[torch.Tensor], grid_size: Tuple[int, int]) -> List[torch.Tensor]:
        h, w = grid_size
        t = self.effective_time_dim
        out = []
        for x in features:
            x_no_token = x[:, 1:, :] if self.remove_cls_token else x
            batch, _, embed = x_no_token.shape
            x_no_token = x_no_token.reshape(batch, t, h, w, embed).permute(0, 1, 4, 2, 3)
            out.append(x_no_token.reshape(batch, t * embed, h, w))
        return out


class LearnedInterpolateToPyramidal(nn.Module):
    """Turn four same-resolution maps into a 4x / 2x / 1x / 0.5x pyramid with learned up-convolutions."""

    def __init__(self, channel_list: Sequence[int]):
        super().__init__()
        if len(channel_list) != 4:
            raise ValueError(f"LearnedInterpolateToPyramidal needs exactly 4 feature maps, got {len(channel_list)}")
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(channel_list[0], channel_list[0] // 2, 2, 2),
            nn.BatchNorm2d(channel_list[0] // 2),
            nn.GELU(),
            nn.ConvTranspose2d(channel_list[0] // 2, channel_list[0] // 4, 2, 2),
        )
        self.fpn2 = nn.Sequential(nn.ConvTranspose2d(channel_list[1], channel_list[1] // 2, 2, 2))
        self.fpn3 = nn.Sequential(nn.Identity())
        self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
        self.embedding_dim = [channel_list[0] // 4, channel_list[1] // 2, channel_list[2], channel_list[3]]

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.fpn1(features[0]), self.fpn2(features[1]), self.fpn3(features[2]), self.fpn4(features[3])]


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, stride: int = 1,
                 use_batchnorm: bool = True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         bias=not use_batchnorm)
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        super().__init__(conv, bn, relu)


class SmpAttention(nn.Module):
    """smp ``base.modules.Attention`` with ``name=None``: an identity kept for parameter-name parity."""

    def __init__(self):
        super().__init__()
        self.attention = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(x)


class DecoderBlock(nn.Module):
    """smp 0.4.0 U-Net block: nearest x2 upsampling, concat skip, two conv-BN-ReLU."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, use_batchnorm: bool = True):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1,
                                use_batchnorm=use_batchnorm)
        self.attention1 = SmpAttention()
        self.conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.attention2 = SmpAttention()

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.attention2(x)


class UnetDecoder(nn.Module):
    """smp 0.4.0 ``UnetDecoder`` (no center block)."""

    def __init__(self, encoder_channels: Sequence[int], decoder_channels: Sequence[int], n_blocks: int = 5,
                 use_batchnorm: bool = True):
        super().__init__()
        if n_blocks != len(decoder_channels):
            raise ValueError(f"Model depth is {n_blocks}, but decoder_channels has {len(decoder_channels)} blocks.")
        encoder_channels = list(encoder_channels[1:])[::-1]
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        self.center = nn.Identity()
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(in_ch, skip_ch, out_ch, use_batchnorm=use_batchnorm)
                for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, decoder_channels)
            ]
        )

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        features = features[1:][::-1]
        head, skips = features[0], features[1:]
        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        return x


def initialize_decoder(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class UNetDecoder(nn.Module):
    """TerraTorch ``UNetDecoder``: the smp decoder fed with the first map duplicated as its dropped skip."""

    def __init__(self, embed_dim: Sequence[int], channels: Sequence[int], use_batchnorm: bool = True):
        super().__init__()
        if len(embed_dim) != len(channels):
            raise ValueError("channels should have the same length as embed_dim")
        self.decoder = UnetDecoder(
            encoder_channels=[embed_dim[0], *embed_dim],
            decoder_channels=channels,
            n_blocks=len(channels),
            use_batchnorm=use_batchnorm,
        )
        initialize_decoder(self.decoder)
        self.out_channels = channels[-1]

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x = [x[0].clone(), *x]
        return self.decoder(*x)


class SegmentationHead(nn.Module):
    """TerraTorch segmentation head: optional dropout, then a 1x1 convolution to class logits."""

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.Identity(),
            nn.Identity() if dropout == 0 else nn.Dropout(dropout),
            nn.Conv2d(in_channels=in_channels, out_channels=num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class PrithviSegmentation(nn.Module):
    """TerraTorch ``EncoderDecoderFactory`` segmentation model on a Prithvi encoder.

    ``encoder`` (PrithviViT) -> ``SelectIndices`` -> ``ReshapeTokensToImage`` ->
    ``LearnedInterpolateToPyramidal`` -> ``UNetDecoder`` -> ``SegmentationHead``, as in the official
    Prithvi-EO-2.0 burn-scar fine-tune. The input is reflect-padded at the bottom/right to a multiple of
    twice the patch size, the logits are bilinearly resized to the padded size and cropped back, so the
    output is ``(B, num_classes, H, W)``.

    Submodules are registered in the reference order (``encoder``, ``decoder``, ``head``, ``neck``) but
    created in the reference initialisation order (encoder, neck, decoder, head), so that the same seed
    gives the same weights as TerraTorch and the Lightning checkpoint (minus its ``model.`` prefix)
    loads with ``strict=True``.
    """

    def __init__(
        self,
        encoder: PrithviViT,
        select_indices: Sequence[int],
        decoder_channels: Sequence[int] = (512, 256, 128, 64),
        num_classes: int = 2,
        head_dropout: float = 0.0,
        rescale: bool = True,
    ):
        super().__init__()
        if len(select_indices) != 4:
            raise ValueError(f"select_indices needs 4 encoder blocks, got {list(select_indices)}")
        depth = len(encoder.blocks)
        if any(not -depth <= i < depth for i in select_indices):
            raise ValueError(f"select_indices {list(select_indices)} out of range for {depth} encoder blocks")
        channel_list = [encoder.out_channels[i] for i in select_indices]
        time_dim = encoder.patch_embed.grid_size[0]
        neck = nn.ModuleList(
            [
                SelectIndices(select_indices),
                ReshapeTokensToImage(effective_time_dim=time_dim),
                LearnedInterpolateToPyramidal(channel_list),
            ]
        )
        pyramid_channels = [channel_list[0] // 4, channel_list[1] // 2, channel_list[2], channel_list[3]]
        decoder = UNetDecoder(pyramid_channels, list(decoder_channels))
        self.encoder = encoder
        self.decoder = decoder
        self.head = SegmentationHead(decoder.out_channels, num_classes, dropout=head_dropout)
        self.neck = neck
        self.num_classes = num_classes
        self.rescale = rescale

    def forward(
        self,
        x: torch.Tensor,
        temporal_coords: Optional[torch.Tensor] = None,
        location_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_frames = self.encoder.num_frames
        if x.ndim == 4 and num_frames == 1:
            x = x.unsqueeze(2)
        if x.ndim != 5 or x.shape[2] != num_frames:
            raise ValueError(
                f"Prithvi segmentation expects input of shape (B, C, T={num_frames}, H, W)"
                + (" or (B, C, H, W)" if num_frames == 1 else "")
                + f", got shape {tuple(x.shape)}"
            )
        height, width = x.shape[-2:]
        _, patch_h, patch_w = self.encoder.patch_embed.patch_size
        pad_h = (2 * patch_h - height % (2 * patch_h)) % (2 * patch_h)
        pad_w = (2 * patch_w - width % (2 * patch_w)) % (2 * patch_w)
        if pad_h >= height or pad_w >= width:
            raise ValueError(
                f"Input shape {tuple(x.shape)} is too small: H and W must exceed the reflect padding to a "
                f"multiple of {2 * patch_h}x{2 * patch_w}"
            )
        if pad_h or pad_w:
            x = torch.stack([F.pad(img, (0, pad_w, 0, pad_h), mode="reflect") for img in x])
        padded_size = x.shape[-2:]
        grid_size = (padded_size[0] // patch_h, padded_size[1] // patch_w)

        features = self.encoder.forward_features(x, temporal_coords, location_coords)
        features = self.neck[0](features)
        features = self.neck[1](features, grid_size)
        features = self.neck[2](features)
        mask = self.head(self.decoder([f.clone() for f in features]))
        if self.rescale and mask.shape[-2:] != padded_size:
            mask = F.interpolate(mask, size=padded_size, mode="bilinear")
        return mask[..., :height, :width]


# --------------------------------------------------------------------------------------------------
# Pretrained weights (downloaded on demand, never bundled)
# --------------------------------------------------------------------------------------------------


class PretrainedWeights(NamedTuple):
    """A checkpoint file in a Hugging Face model repository, pinned by revision and sha256."""

    repo_id: str
    revision: str
    filename: str
    sha256: str
    num_bytes: int

    @property
    def url(self) -> str:
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        return f"{endpoint}/{self.repo_id}/resolve/{self.revision}/{self.filename}"


PRITHVI_WEIGHTS: Dict[str, PretrainedWeights] = {
    "prithvi_eo_v2_300_burnscars": PretrainedWeights(
        repo_id="ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars",
        revision="a3f2c410e45b8ac7417976614528a872f024d831",
        filename="Prithvi_EO_V2_300M_BurnScars.pt",
        sha256="0c5f9334be9a75c9006387ab8f3dc05a55ea7fb5ef7956717316be57c62954d3",
        num_bytes=1297798380,
    ),
    "prithvi_eo_v2_300_tl": PretrainedWeights(
        repo_id="ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL",
        revision="63adbd39c271da4c42f447e69b1a7c91a338cdc9",
        filename="Prithvi_EO_V2_300M_TL.pt",
        sha256="3629cedfbb350faafcb0dac902ae0d3c927e25ce8d9e0024aa1276ec66956ddb",
        num_bytes=1326660716,
    ),
    "prithvi_eo_v2_600_tl": PretrainedWeights(
        repo_id="ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL",
        revision="3d72adc6dfc4cc3862bf4f41da8c83db267193c9",
        filename="Prithvi_EO_V2_600M_TL.pt",
        sha256="7b92c53b0204a76bb775bd8930f045e05776251caa8c83f7367ed0b75b594702",
        num_bytes=2638217218,
    ),
}


def download_weights(spec: PretrainedWeights, cache_dir: Optional[Union[str, Path]] = None, progress: bool = True) -> Path:
    """Download ``spec`` over HTTPS into the cache (once) and return the local path.

    The file is checked against the pinned sha256 while downloading. ``cache_dir`` defaults to
    ``torch.hub.get_dir()/checkpoints``; ``HF_ENDPOINT`` selects a Hugging Face mirror.
    """
    root = Path(cache_dir) if cache_dir is not None else Path(torch.hub.get_dir()) / "checkpoints"
    target = root / spec.repo_id.replace("/", "--") / spec.revision / spec.filename
    if target.exists() and target.stat().st_size == spec.num_bytes:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.hub.download_url_to_file(spec.url, str(target), hash_prefix=spec.sha256, progress=progress)
    return target


def load_pretrained_state_dict(
    name: str,
    weights_path: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> "OrderedDict[str, torch.Tensor]":
    """State dict of a pinned Prithvi checkpoint, from ``weights_path`` or downloaded on demand.

    Lightning checkpoints (fine-tuned models) are unwrapped to their ``state_dict``.
    """
    if name not in PRITHVI_WEIGHTS:
        raise ValueError(f"Unknown Prithvi weights {name!r}; available: {sorted(PRITHVI_WEIGHTS)}")
    path = Path(weights_path) if weights_path is not None else download_weights(PRITHVI_WEIGHTS[name], cache_dir)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    return OrderedDict(state)


def finetuned_state_dict_to_model(state: Dict[str, torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
    """TerraTorch task checkpoint -> :class:`PrithviSegmentation` keys (drop the ``model.`` prefix)."""
    return OrderedDict((key[len("model."):], value) for key, value in state.items() if key.startswith("model."))


def mae_state_dict_to_encoder(state: Dict[str, torch.Tensor], encoder: PrithviViT) -> "OrderedDict[str, torch.Tensor]":
    """Pretraining (PrithviMAE) checkpoint -> :class:`PrithviViT` keys, as TerraTorch's filter does.

    Keeps ``encoder.*`` without the prefix and drops the MAE decoder. ``pos_embed`` is a fixed sin/cos
    table that depends on ``num_frames``, so the encoder's own table replaces the checkpoint's.
    """
    out = OrderedDict()
    for key, value in state.items():
        if not key.startswith("encoder."):
            continue
        key = key[len("encoder."):]
        if (not encoder.temporal_encoding and "temporal_embed" in key) or (
            not encoder.location_encoding and "location_embed" in key
        ):
            continue
        out[key] = encoder.pos_embed.detach().clone() if key == "pos_embed" else value
    return out


__all__ = [
    "PRITHVI_BANDS",
    "PRITHVI_EO_V2_CONFIGS",
    "PRITHVI_EO_V2_MEAN",
    "PRITHVI_EO_V2_STD",
    "PRITHVI_WEIGHTS",
    "LearnedInterpolateToPyramidal",
    "MAEDecoder",
    "PretrainedWeights",
    "PrithviMAE",
    "PrithviSegmentation",
    "PrithviViT",
    "ReshapeTokensToImage",
    "SegmentationHead",
    "SelectIndices",
    "UNetDecoder",
    "download_weights",
    "finetuned_state_dict_to_model",
    "get_3d_sincos_pos_embed",
    "load_pretrained_state_dict",
    "mae_state_dict_to_encoder",
]
