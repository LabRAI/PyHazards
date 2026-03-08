from __future__ import annotations

import torch
import torch.nn as nn


class TropiCycloneNet(nn.Module):
    """GRU + attention baseline for all-basin tropical cyclone forecasting."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        horizon: int = 5,
        output_dim: int = 3,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.output_dim = int(output_dim)
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attention = nn.Linear(2 * hidden_dim, 1)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.horizon * self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("TropiCycloneNet expects inputs shaped (batch, history, features).")
        encoded, _ = self.encoder(x)
        weights = torch.softmax(self.attention(encoded), dim=1)
        pooled = torch.sum(weights * encoded, dim=1)
        preds = self.head(pooled)
        return preds.view(x.size(0), self.horizon, self.output_dim)


def tropicyclonenet_builder(
    task: str,
    input_dim: int = 8,
    hidden_dim: int = 64,
    horizon: int = 5,
    output_dim: int = 3,
    num_layers: int = 2,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() != "regression":
        raise ValueError("TropiCycloneNet only supports regression for track/intensity forecasting.")
    return TropiCycloneNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        horizon=horizon,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout,
    )


__all__ = ["TropiCycloneNet", "tropicyclonenet_builder"]
