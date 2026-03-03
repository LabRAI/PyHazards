from __future__ import annotations

import torch
import torch.nn as nn


class WildfireFPALSTM(nn.Module):
    """Sequence model for next-week FPA-FOD count forecasting."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 5,
        num_layers: int = 1,
        dropout: float = 0.2,
        lookback: int = 50,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if lookback <= 0:
            raise ValueError(f"lookback must be positive, got {lookback}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.lookback = lookback
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, T, D), got {tuple(x.shape)}")
        if x.size(1) != self.lookback:
            raise ValueError(f"Expected lookback={self.lookback}, got sequence length {x.size(1)}")
        _, (hidden, _) = self.encoder(x)
        return self.head(hidden[-1])


def wildfire_fpa_lstm_builder(
    task: str,
    input_dim: int,
    hidden_dim: int = 64,
    output_dim: int = 5,
    num_layers: int = 1,
    dropout: float = 0.2,
    lookback: int = 50,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() not in {"regression", "forecasting"}:
        raise ValueError(
            f"wildfire_fpa_lstm supports task='regression' or 'forecasting', got {task!r}"
        )
    return WildfireFPALSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout,
        lookback=lookback,
    )


__all__ = ["WildfireFPALSTM", "wildfire_fpa_lstm_builder"]
