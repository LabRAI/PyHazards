from __future__ import annotations

import torch
import torch.nn as nn


class Hurricast(nn.Module):
    """Compact storm-track and intensity baseline for Wave 2 vertical slices."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        horizon: int = 5,
        output_dim: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.output_dim = int(output_dim)
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.horizon * self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Hurricast expects inputs shaped (batch, history, features).")
        encoded, _ = self.encoder(x)
        last = encoded[:, -1, :]
        preds = self.head(last)
        return preds.view(x.size(0), self.horizon, self.output_dim)


def hurricast_builder(
    task: str,
    input_dim: int = 8,
    hidden_dim: int = 64,
    num_layers: int = 2,
    horizon: int = 5,
    output_dim: int = 3,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() != "regression":
        raise ValueError("Hurricast only supports regression for track/intensity forecasting.")
    return Hurricast(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        horizon=horizon,
        output_dim=output_dim,
        dropout=dropout,
    )


__all__ = ["Hurricast", "hurricast_builder"]
