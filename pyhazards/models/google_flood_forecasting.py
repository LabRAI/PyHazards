from __future__ import annotations

import torch
import torch.nn as nn


class GoogleFloodForecasting(nn.Module):
    """Sequence baseline for streamflow-style flood forecasting."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        out_dim: int = 1,
        history: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")
        if history <= 0:
            raise ValueError(f"history must be positive, got {history}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.history = int(history)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=2,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, batch) -> torch.Tensor:
        if not isinstance(batch, dict) or "x" not in batch:
            raise ValueError("GoogleFloodForecasting expects a mapping input with key 'x'.")
        x = batch["x"]
        if x.ndim != 4:
            raise ValueError(
                "GoogleFloodForecasting expects input shape (batch, history, nodes, features), "
                f"got {tuple(x.shape)}."
            )
        if x.size(1) != self.history:
            raise ValueError(f"GoogleFloodForecasting expected history={self.history}, got {x.size(1)}.")
        encoded = self.proj(x)
        temporal = encoded.permute(0, 2, 1, 3).reshape(-1, self.history, encoded.size(-1))
        hidden = self.temporal(temporal)[:, -1]
        preds = self.head(hidden)
        return preds.view(x.size(0), x.size(2), -1)


def google_flood_forecasting_builder(
    task: str,
    input_dim: int = 2,
    hidden_dim: int = 64,
    out_dim: int = 1,
    history: int = 4,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() != "regression":
        raise ValueError(
            "google_flood_forecasting only supports task='regression', "
            f"got {task!r}."
        )
    return GoogleFloodForecasting(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        history=history,
        dropout=dropout,
    )


__all__ = ["GoogleFloodForecasting", "google_flood_forecasting_builder"]
