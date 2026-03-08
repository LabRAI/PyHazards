from __future__ import annotations

import torch
import torch.nn as nn


class EQTransformer(nn.Module):
    """Compact sequence model for joint earthquake phase picking."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 48,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.ReLU(),
        )
        self.temporal = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.Linear(2 * hidden_dim, 1)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("EQTransformer expects inputs shaped (batch, channels, length).")
        encoded = self.encoder(x).transpose(1, 2)
        temporal, _ = self.temporal(encoded)
        weights = torch.softmax(self.attention(temporal), dim=1)
        pooled = torch.sum(weights * temporal, dim=1)
        return self.head(pooled)


def eqtransformer_builder(
    task: str,
    in_channels: int = 3,
    hidden_dim: int = 48,
    num_layers: int = 2,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() != "regression":
        raise ValueError("EQTransformer only supports regression-style phase picking outputs.")
    return EQTransformer(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )


__all__ = ["EQTransformer", "eqtransformer_builder"]
