from __future__ import annotations

import torch
import torch.nn as nn


class PhaseNet(nn.Module):
    """Lightweight phase-picking network for synthetic waveform smoke runs."""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("PhaseNet expects inputs shaped (batch, channels, length).")
        return self.head(self.encoder(x))


def phasenet_builder(
    task: str,
    in_channels: int = 3,
    hidden_dim: int = 32,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() != "regression":
        raise ValueError("PhaseNet only supports regression-style phase picking outputs.")
    return PhaseNet(in_channels=in_channels, hidden_dim=hidden_dim)


__all__ = ["PhaseNet", "phasenet_builder"]
