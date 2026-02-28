# pyhazards/models/wildfire_fpa_lstm.py
from __future__ import annotations

import torch
import torch.nn as nn

from .registry import register_model


class WildfireFPALSTM(nn.Module):
    """
    LSTM forecaster producing raw next-week counts (regression).

    forward:
      x: (B, T, D)  T should equal lookback (default 50)
      returns: (B, output_dim=5)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 5,
        num_layers: int = 1,
        dropout: float = 0.2,
        lookback: int = 50,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be > 0, got {output_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0,1), got {dropout}")
        if lookback <= 0:
            raise ValueError(f"lookback must be > 0, got {lookback}")

        self.lookback = lookback

        # PyTorch LSTM dropout is applied only when num_layers > 1
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"LSTM expects x as (B, T, D). Got {tuple(x.shape)}")
        B, T, D = x.shape
        if T != self.lookback:
            raise ValueError(f"Expected lookback T={self.lookback}, got T={T}")

        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]  # (B, hidden_dim)
        return self.head(h_last)


def build_wildfire_fpa_lstm(
    *,
    task: str,
    input_dim: int,
    hidden_dim: int = 64,
    output_dim: int = 5,
    num_layers: int = 1,
    dropout: float = 0.2,
    lookback: int = 50,
    **kwargs,
) -> nn.Module:
    if kwargs:
        raise TypeError(f"Unexpected kwargs for wildfire_fpa_lstm: {sorted(kwargs.keys())}")

    t = str(task).lower()
    if t not in ("regression", "forecasting"):
        raise ValueError(f"wildfire_fpa_lstm supports task='regression'/'forecasting', got {task!r}")

    return WildfireFPALSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout,
        lookback=lookback,
    )


register_model(
    name="wildfire_fpa_lstm",
    builder=build_wildfire_fpa_lstm,
    defaults={
        "hidden_dim": 64,
        "output_dim": 5,
        "num_layers": 1,
        "dropout": 0.2,
        "lookback": 50,
    },
)
