from __future__ import annotations

import torch
import torch.nn as nn


class FloodCast(nn.Module):
    """Compact spatiotemporal inundation baseline."""

    def __init__(
        self,
        in_channels: int = 3,
        history: int = 4,
        hidden_dim: int = 32,
        out_channels: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.history = int(history)
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
        )
        self.head = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("FloodCast expects inputs shaped (batch, history, channels, height, width).")
        if x.size(1) != self.history:
            raise ValueError(f"FloodCast expected history={self.history}, got {x.size(1)}.")
        encoded = self.encoder(x.permute(0, 2, 1, 3, 4))
        fused = encoded.mean(dim=2)
        return self.head(fused)


def floodcast_builder(
    task: str,
    in_channels: int = 3,
    history: int = 4,
    hidden_dim: int = 32,
    out_channels: int = 1,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() not in {"regression", "segmentation"}:
        raise ValueError("FloodCast only supports regression or segmentation-style inundation outputs.")
    return FloodCast(
        in_channels=in_channels,
        history=history,
        hidden_dim=hidden_dim,
        out_channels=out_channels,
        dropout=dropout,
    )


__all__ = ["FloodCast", "floodcast_builder"]
