from __future__ import annotations

import torch
import torch.nn as nn


class ASUFM(nn.Module):
    """Temporal convolution baseline for wildfire activity forecasting."""

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 64,
        output_dim: int = 5,
        lookback: int = 12,
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
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.lookback = int(lookback)
        self.temporal = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "ASUFM expects input shape (batch, lookback, features), "
                f"got {tuple(x.shape)}."
            )
        if x.size(1) != self.lookback:
            raise ValueError(f"ASUFM expected lookback={self.lookback}, got sequence length {x.size(1)}.")
        encoded = self.temporal(x.transpose(1, 2))
        gated = encoded * self.gate(encoded)
        pooled = torch.mean(gated, dim=-1)
        return self.head(pooled)


def asufm_builder(
    task: str,
    input_dim: int = 7,
    hidden_dim: int = 64,
    output_dim: int = 5,
    lookback: int = 12,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() not in {"forecasting", "regression"}:
        raise ValueError(f"asufm supports task='forecasting' or 'regression', got {task!r}.")
    return ASUFM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        lookback=lookback,
        dropout=dropout,
    )


__all__ = ["ASUFM", "asufm_builder"]
