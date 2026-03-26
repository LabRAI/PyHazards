from __future__ import annotations

import torch
import torch.nn as nn


class TSSatFire(nn.Module):
    """Spatio-temporal wildfire prediction model inspired by TS-SatFire."""

    def __init__(
        self,
        history: int = 5,
        in_channels: int = 8,
        hidden_dim: int = 32,
        out_channels: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        if history <= 0:
            raise ValueError(f"history must be positive, got {history}")
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.history = int(history)
        self.in_channels = int(in_channels)
        self.temporal_encoder = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=(3, 3, 3), padding=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(3, 3, 3), padding=1),
            nn.GELU(),
        )
        self.time_attention = nn.Conv3d(hidden_dim, 1, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                "TSSatFire expects input shape (batch, history, channels, height, width), "
                f"got {tuple(x.shape)}."
            )
        if x.size(1) != self.history:
            raise ValueError(f"TSSatFire expected history={self.history}, got {x.size(1)}.")
        if x.size(2) != self.in_channels:
            raise ValueError(f"TSSatFire expected in_channels={self.in_channels}, got {x.size(2)}.")

        feat = self.temporal_encoder(x.permute(0, 2, 1, 3, 4))
        attn = torch.softmax(self.time_attention(feat), dim=2)
        pooled = torch.sum(attn * feat, dim=2)
        return self.decoder(pooled)


def ts_satfire_builder(
    task: str,
    history: int = 5,
    in_channels: int = 8,
    hidden_dim: int = 32,
    out_channels: int = 1,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() not in {"segmentation", "regression"}:
        raise ValueError(f"ts_satfire supports task='segmentation' or 'regression', got {task!r}.")
    return TSSatFire(
        history=history,
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        out_channels=out_channels,
        dropout=dropout,
    )


__all__ = ["TSSatFire", "ts_satfire_builder"]
