from __future__ import annotations

import torch
import torch.nn as nn


class WildfireForecasting(nn.Module):
    """Sequence forecaster for weekly wildfire size-group activity."""

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 64,
        output_dim: int = 5,
        lookback: int = 12,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if lookback <= 0:
            raise ValueError(f"lookback must be positive, got {lookback}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.lookback = int(lookback)
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "WildfireForecasting expects input shape (batch, lookback, features), "
                f"got {tuple(x.shape)}."
            )
        if x.size(1) != self.lookback:
            raise ValueError(
                f"WildfireForecasting expected lookback={self.lookback}, got sequence length {x.size(1)}."
            )
        encoded, _ = self.encoder(x)
        weights = torch.softmax(self.attention(encoded), dim=1)
        pooled = torch.sum(weights * encoded, dim=1)
        return self.head(pooled)


def wildfire_forecasting_builder(
    task: str,
    input_dim: int = 7,
    hidden_dim: int = 64,
    output_dim: int = 5,
    lookback: int = 12,
    num_layers: int = 2,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() not in {"forecasting", "regression"}:
        raise ValueError(
            "wildfire_forecasting supports task='forecasting' or 'regression', "
            f"got {task!r}."
        )
    return WildfireForecasting(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        lookback=lookback,
        num_layers=num_layers,
        dropout=dropout,
    )


__all__ = ["WildfireForecasting", "wildfire_forecasting_builder"]
