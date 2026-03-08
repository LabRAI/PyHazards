from __future__ import annotations

import torch
import torch.nn as nn


class SAFNet(nn.Module):
    """Spatiotemporal intensity-focused storm baseline."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        horizon: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.temporal = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.track_head = nn.Linear(hidden_dim, 2 * self.horizon)
        self.intensity_head = nn.Linear(hidden_dim, self.horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("SAFNet expects inputs shaped (batch, history, features).")
        encoded = self.temporal(x.transpose(1, 2)).mean(dim=-1)
        encoded = self.dropout(encoded)
        track = self.track_head(encoded).view(x.size(0), self.horizon, 2)
        intensity = self.intensity_head(encoded).view(x.size(0), self.horizon, 1)
        return torch.cat([track, intensity], dim=-1)


def saf_net_builder(
    task: str,
    input_dim: int = 8,
    hidden_dim: int = 64,
    horizon: int = 5,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() != "regression":
        raise ValueError("SAFNet only supports regression for track/intensity forecasting.")
    return SAFNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        horizon=horizon,
        dropout=dropout,
    )


__all__ = ["SAFNet", "saf_net_builder"]
