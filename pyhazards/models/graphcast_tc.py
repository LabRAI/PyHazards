from __future__ import annotations

import torch
import torch.nn as nn


class GraphCastTC(nn.Module):
    """Experimental wrapper-style GraphCast storm adapter."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 96,
        horizon: int = 5,
        output_dim: int = 3,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.output_dim = int(output_dim)
        self.proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=2 * hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_dim, self.horizon * self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("GraphCastTC expects inputs shaped (batch, history, features).")
        encoded = self.encoder(self.proj(x))
        preds = self.head(encoded.mean(dim=1))
        return preds.view(x.size(0), self.horizon, self.output_dim)


def graphcast_tc_builder(
    task: str,
    input_dim: int = 8,
    hidden_dim: int = 96,
    horizon: int = 5,
    output_dim: int = 3,
    num_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() != "regression":
        raise ValueError("GraphCastTC only supports regression for track/intensity forecasting.")
    return GraphCastTC(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        horizon=horizon,
        output_dim=output_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    )


__all__ = ["GraphCastTC", "graphcast_tc_builder"]
