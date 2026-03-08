from __future__ import annotations

import torch
import torch.nn as nn


class UrbanFloodCast(nn.Module):
    """U-Net style urban inundation baseline."""

    def __init__(
        self,
        in_channels: int = 3,
        history: int = 4,
        base_channels: int = 32,
        out_channels: int = 1,
    ):
        super().__init__()
        self.history = int(history)
        merged_channels = in_channels * history
        self.encoder = nn.Sequential(
            nn.Conv2d(merged_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("UrbanFloodCast expects inputs shaped (batch, history, channels, height, width).")
        if x.size(1) != self.history:
            raise ValueError(f"UrbanFloodCast expected history={self.history}, got {x.size(1)}.")
        bsz, history, channels, height, width = x.shape
        merged = x.reshape(bsz, history * channels, height, width)
        features = self.encoder(merged)
        return self.decoder(features)


def urbanfloodcast_builder(
    task: str,
    in_channels: int = 3,
    history: int = 4,
    base_channels: int = 32,
    out_channels: int = 1,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() not in {"regression", "segmentation"}:
        raise ValueError("UrbanFloodCast only supports regression or segmentation-style inundation outputs.")
    return UrbanFloodCast(
        in_channels=in_channels,
        history=history,
        base_channels=base_channels,
        out_channels=out_channels,
    )


__all__ = ["UrbanFloodCast", "urbanfloodcast_builder"]
