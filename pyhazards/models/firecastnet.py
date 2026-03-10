from __future__ import annotations

import torch
import torch.nn as nn


class FireCastNet(nn.Module):
    """Compact encoder-decoder wildfire spread network."""

    def __init__(
        self,
        in_channels: int = 12,
        hidden_dim: int = 32,
        out_channels: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.in_channels = int(in_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "FireCastNet expects input shape (batch, channels, height, width), "
                f"got {tuple(x.shape)}."
            )
        if x.size(1) != self.in_channels:
            raise ValueError(f"FireCastNet expected in_channels={self.in_channels}, got {x.size(1)}.")
        encoded = self.encoder(x)
        return self.decoder(encoded)


def firecastnet_builder(
    task: str,
    in_channels: int = 12,
    hidden_dim: int = 32,
    out_channels: int = 1,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() not in {"segmentation", "regression"}:
        raise ValueError(f"firecastnet supports task='segmentation' or 'regression', got {task!r}.")
    return FireCastNet(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        out_channels=out_channels,
        dropout=dropout,
    )


__all__ = ["FireCastNet", "firecastnet_builder"]
