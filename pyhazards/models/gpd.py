from __future__ import annotations

import torch
import torch.nn as nn


class GPD(nn.Module):
    """Simple CNN baseline for generalized phase detection style picking."""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("GPD expects inputs shaped (batch, channels, length).")
        return self.head(self.features(x))


def gpd_builder(
    task: str,
    in_channels: int = 3,
    hidden_dim: int = 32,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() != "regression":
        raise ValueError("GPD only supports regression-style phase picking outputs.")
    return GPD(in_channels=in_channels, hidden_dim=hidden_dim, dropout=dropout)


__all__ = ["GPD", "gpd_builder"]
