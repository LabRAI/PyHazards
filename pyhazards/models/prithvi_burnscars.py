from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .prithvi_eo_2_tl import PrithviEOBackbone


class PrithviBurnScars(nn.Module):
    """Burn-scar segmentation model built on a Prithvi-EO-style temporal backbone."""

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
        self.in_channels = int(in_channels)
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
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, decoder_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim + decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoder_channels, decoder_channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Conv2d(decoder_channels // 2, out_dim, kernel_size=1)

    def _extract_x(self, inputs: torch.Tensor | Dict[str, Any]) -> torch.Tensor:
        if isinstance(inputs, dict):
            x = inputs.get("x")
        else:
            x = inputs
        if not isinstance(x, torch.Tensor):
            raise ValueError("PrithviBurnScars expects a tensor input or a dict containing key 'x'.")
        if x.ndim != 5:
            raise ValueError(f"PrithviBurnScars expects input shape (B,T,C,H,W), got {tuple(x.shape)}")
        if x.size(2) != self.in_channels:
            raise ValueError(f"PrithviBurnScars expected in_channels={self.in_channels}, got {x.size(2)}")
        return x

    def forward(self, inputs: torch.Tensor | Dict[str, Any]) -> torch.Tensor:
        x = self._extract_x(inputs)
        features = self.backbone(inputs)
        skip = self.skip(x.mean(dim=1))
        up = F.interpolate(features, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat([up, skip], dim=1)
        logits = self.head(self.decoder(fused))
        return logits



def prithvi_burnscars_builder(
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
        raise ValueError(f"prithvi_burnscars is segmentation-only, got task={task!r}.")
    return PrithviBurnScars(
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


__all__ = ["PrithviBurnScars", "prithvi_burnscars_builder"]
