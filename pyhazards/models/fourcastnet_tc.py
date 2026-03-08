from __future__ import annotations

import torch
import torch.nn as nn


class FourCastNetTC(nn.Module):
    """Experimental wrapper-style FourCastNet storm adapter."""

    def __init__(
        self,
        input_dim: int = 8,
        history: int = 6,
        hidden_dim: int = 96,
        horizon: int = 5,
        output_dim: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.history = int(history)
        self.horizon = int(horizon)
        self.output_dim = int(output_dim)
        self.net = nn.Sequential(
            nn.Linear(self.history * input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.horizon * self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("FourCastNetTC expects inputs shaped (batch, history, features).")
        if x.size(1) != self.history:
            raise ValueError(f"FourCastNetTC expected history={self.history}, got {x.size(1)}.")
        preds = self.net(x.reshape(x.size(0), -1))
        return preds.view(x.size(0), self.horizon, self.output_dim)


def fourcastnet_tc_builder(
    task: str,
    input_dim: int = 8,
    history: int = 6,
    hidden_dim: int = 96,
    horizon: int = 5,
    output_dim: int = 3,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() != "regression":
        raise ValueError("FourCastNetTC only supports regression for track/intensity forecasting.")
    return FourCastNetTC(
        input_dim=input_dim,
        history=history,
        hidden_dim=hidden_dim,
        horizon=horizon,
        output_dim=output_dim,
        dropout=dropout,
    )


__all__ = ["FourCastNetTC", "fourcastnet_tc_builder"]
