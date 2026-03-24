from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EOSequencePatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        return self.proj(x)


class PrithviEOBackbone(nn.Module):
    """Lightweight temporal-location-aware EO backbone inspired by Prithvi-EO-2.0-TL."""

    def __init__(
        self,
        image_size: int = 32,
        in_channels: int = 6,
        patch_size: int = 4,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        time_dim: int = 1,
        location_dim: int = 2,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size={image_size} must be divisible by patch_size={patch_size}")
        self.image_size = int(image_size)
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.time_dim = int(time_dim)
        self.location_dim = int(location_dim)
        self.grid_size = self.image_size // self.patch_size

        self.patch_embed = EOSequencePatchEmbed(
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
        )
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.grid_size * self.grid_size, self.embed_dim)
        )
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)

        self.time_proj = nn.Linear(self.time_dim, self.embed_dim)
        self.location_proj = nn.Linear(self.location_dim, self.embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=int(num_heads),
            dim_feedforward=int(self.embed_dim * mlp_ratio),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(depth))
        self.norm = nn.LayerNorm(self.embed_dim)

    def _unpack_inputs(self, inputs: torch.Tensor | Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if isinstance(inputs, dict):
            x = inputs.get("x")
            time_metadata = inputs.get("time_metadata")
            location_metadata = inputs.get("location_metadata")
        else:
            x = inputs
            time_metadata = None
            location_metadata = None

        if not isinstance(x, torch.Tensor):
            raise ValueError("PrithviEOBackbone expects a tensor input or a dict containing key 'x'.")
        if x.ndim != 5:
            raise ValueError(
                "PrithviEOBackbone expects input shape (B, T, C, H, W), "
                f"got {tuple(x.shape)}."
            )
        if x.size(2) != self.in_channels:
            raise ValueError(
                f"PrithviEOBackbone expected in_channels={self.in_channels}, got {x.size(2)}."
            )
        if x.size(-1) != self.image_size or x.size(-2) != self.image_size:
            raise ValueError(
                f"PrithviEOBackbone expected spatial size {self.image_size}x{self.image_size}, "
                f"got {tuple(x.shape[-2:])}."
            )
        return x, time_metadata, location_metadata

    def _build_time_metadata(self, batch: int, timesteps: int, device: torch.device, meta: torch.Tensor | None) -> torch.Tensor:
        if meta is None:
            base = torch.linspace(0.0, 1.0, timesteps, device=device).view(1, timesteps, 1)
            return base.expand(batch, -1, -1)
        if meta.ndim == 2:
            meta = meta.unsqueeze(-1)
        if meta.ndim != 3:
            raise ValueError(f"time_metadata must have shape (B,T) or (B,T,D), got {tuple(meta.shape)}")
        if meta.size(0) != batch or meta.size(1) != timesteps:
            raise ValueError(
                f"time_metadata expected batch/timestep=({batch},{timesteps}), got ({meta.size(0)},{meta.size(1)})"
            )
        if meta.size(-1) == self.time_dim:
            return meta.to(device=device, dtype=torch.float32)
        if meta.size(-1) > self.time_dim:
            return meta[..., : self.time_dim].to(device=device, dtype=torch.float32)
        pad = torch.zeros(batch, timesteps, self.time_dim - meta.size(-1), device=device)
        return torch.cat([meta.to(device=device, dtype=torch.float32), pad], dim=-1)

    def _build_location_metadata(self, batch: int, device: torch.device, meta: torch.Tensor | None) -> torch.Tensor:
        if meta is None:
            return torch.zeros(batch, self.location_dim, device=device)
        if meta.ndim != 2:
            raise ValueError(f"location_metadata must have shape (B,D), got {tuple(meta.shape)}")
        if meta.size(0) != batch:
            raise ValueError(f"location_metadata expected batch={batch}, got {meta.size(0)}")
        if meta.size(-1) == self.location_dim:
            return meta.to(device=device, dtype=torch.float32)
        if meta.size(-1) > self.location_dim:
            return meta[..., : self.location_dim].to(device=device, dtype=torch.float32)
        pad = torch.zeros(batch, self.location_dim - meta.size(-1), device=device)
        return torch.cat([meta.to(device=device, dtype=torch.float32), pad], dim=-1)

    def forward(self, inputs: torch.Tensor | Dict[str, Any]) -> torch.Tensor:
        x, time_metadata, location_metadata = self._unpack_inputs(inputs)
        batch, timesteps, _, height, width = x.shape
        device = x.device

        feat = self.patch_embed(x)
        _, _, _, h_tokens, w_tokens = feat.shape
        tokens = feat.permute(0, 2, 3, 4, 1).reshape(batch, timesteps * h_tokens * w_tokens, self.embed_dim)

        spatial_pos = self.spatial_pos_embed.unsqueeze(1).expand(-1, timesteps, -1, -1)
        spatial_pos = spatial_pos.reshape(1, timesteps * h_tokens * w_tokens, self.embed_dim)
        tokens = tokens + spatial_pos

        time_meta = self._build_time_metadata(batch, timesteps, device, time_metadata)
        time_tokens = self.time_proj(time_meta).unsqueeze(2).expand(-1, -1, h_tokens * w_tokens, -1)
        time_tokens = time_tokens.reshape(batch, timesteps * h_tokens * w_tokens, self.embed_dim)
        tokens = tokens + time_tokens

        location_meta = self._build_location_metadata(batch, device, location_metadata)
        tokens = tokens + self.location_proj(location_meta).unsqueeze(1)

        encoded = self.norm(self.encoder(tokens))
        encoded = encoded.reshape(batch, timesteps, h_tokens, w_tokens, self.embed_dim).mean(dim=1)
        return encoded.permute(0, 3, 1, 2).contiguous()


class PrithviEO2TL(nn.Module):
    """Temporal-location-aware EO segmentation model inspired by Prithvi-EO-2.0-TL."""

    def __init__(
        self,
        image_size: int = 32,
        in_channels: int = 6,
        out_dim: int = 1,
        patch_size: int = 4,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        time_dim: int = 1,
        location_dim: int = 2,
        decoder_channels: int = 64,
    ):
        super().__init__()
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")
        self.image_size = int(image_size)
        self.backbone = PrithviEOBackbone(
            image_size=image_size,
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            time_dim=time_dim,
            location_dim=location_dim,
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, decoder_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Conv2d(decoder_channels, out_dim, kernel_size=1)

    def forward(self, inputs: torch.Tensor | Dict[str, Any]) -> torch.Tensor:
        if isinstance(inputs, dict):
            x = inputs["x"]
        else:
            x = inputs
        features = self.backbone(inputs)
        logits = self.head(self.decoder(features))
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

def prithvi_eo_2_tl_builder(
    task: str,
    image_size: int = 32,
    in_channels: int = 6,
    out_dim: int = 1,
    patch_size: int = 4,
    embed_dim: int = 128,
    depth: int = 4,
    num_heads: int = 4,
    mlp_ratio: float = 4.0,
    dropout: float = 0.1,
    time_dim: int = 1,
    location_dim: int = 2,
    decoder_channels: int = 64,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() != "segmentation":
        raise ValueError(f"prithvi_eo_2_tl is segmentation-only, got task={task!r}.")
    return PrithviEO2TL(
        image_size=image_size,
        in_channels=in_channels,
        out_dim=out_dim,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        time_dim=time_dim,
        location_dim=location_dim,
        decoder_channels=decoder_channels,
    )


__all__ = ["PrithviEOBackbone", "PrithviEO2TL", "prithvi_eo_2_tl_builder"]
