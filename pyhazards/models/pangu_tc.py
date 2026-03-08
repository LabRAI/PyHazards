from __future__ import annotations

import torch
import torch.nn as nn


class PanguTC(nn.Module):
    """Experimental wrapper-style Pangu-Weather storm adapter."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 96,
        horizon: int = 5,
        output_dim: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.output_dim = int(output_dim)
        self.temporal = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.horizon * self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("PanguTC expects inputs shaped (batch, history, features).")
        encoded = self.temporal(x.transpose(1, 2)).mean(dim=-1)
        preds = self.head(encoded)
        return preds.view(x.size(0), self.horizon, self.output_dim)


def pangu_tc_builder(
    task: str,
    input_dim: int = 8,
    hidden_dim: int = 96,
    horizon: int = 5,
    output_dim: int = 3,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() != "regression":
        raise ValueError("PanguTC only supports regression for track/intensity forecasting.")
    return PanguTC(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        horizon=horizon,
        output_dim=output_dim,
        dropout=dropout,
    )


__all__ = ["PanguTC", "pangu_tc_builder"]
