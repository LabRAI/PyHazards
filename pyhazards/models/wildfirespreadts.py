from __future__ import annotations

import torch
import torch.nn as nn


class WildfireSpreadTS(nn.Module):
    """Temporal convolution baseline for wildfire spread masks."""

    def __init__(
        self,
        history: int = 4,
        in_channels: int = 6,
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
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=(3, 3, 3), padding=1),
            nn.GELU(),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(3, 3, 3), padding=1),
            nn.GELU(),
        )
        self.head = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                "WildfireSpreadTS expects input shape (batch, history, channels, height, width), "
                f"got {tuple(x.shape)}."
            )
        if x.size(1) != self.history:
            raise ValueError(f"WildfireSpreadTS expected history={self.history}, got {x.size(1)}.")
        if x.size(2) != self.in_channels:
            raise ValueError(f"WildfireSpreadTS expected in_channels={self.in_channels}, got {x.size(2)}.")
        encoded = self.encoder(x.permute(0, 2, 1, 3, 4))
        return self.head(torch.mean(encoded, dim=2))


def wildfirespreadts_builder(
    task: str,
    history: int = 4,
    in_channels: int = 6,
    hidden_dim: int = 32,
    out_channels: int = 1,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() not in {"segmentation", "regression"}:
        raise ValueError(
            "wildfirespreadts supports task='segmentation' or 'regression', "
            f"got {task!r}."
        )
    return WildfireSpreadTS(
        history=history,
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        out_channels=out_channels,
        dropout=dropout,
    )


__all__ = ["WildfireSpreadTS", "wildfirespreadts_builder"]
