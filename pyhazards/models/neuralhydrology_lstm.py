from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _streamflow_inputs(batch: Any) -> torch.Tensor:
    x = batch["x"] if isinstance(batch, dict) else batch
    if x.ndim != 4:
        raise ValueError("NeuralHydrology-style models expect inputs shaped (batch, history, nodes, features).")
    return x


class NeuralHydrologyLSTM(nn.Module):
    """Adapter-style LSTM streamflow baseline."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        out_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.out_dim = int(out_dim)
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, self.out_dim)

    def forward(self, batch: Any) -> torch.Tensor:
        x = _streamflow_inputs(batch)
        bsz, history, nodes, features = x.shape
        series = x.permute(0, 2, 1, 3).reshape(bsz * nodes, history, features)
        encoded, _ = self.encoder(series)
        preds = self.head(encoded[:, -1])
        return preds.view(bsz, nodes, self.out_dim)


def neuralhydrology_lstm_builder(
    task: str,
    input_dim: int = 2,
    hidden_dim: int = 64,
    num_layers: int = 2,
    out_dim: int = 1,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() != "regression":
        raise ValueError("NeuralHydrologyLSTM only supports regression for streamflow forecasting.")
    return NeuralHydrologyLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        out_dim=out_dim,
        dropout=dropout,
    )


__all__ = ["NeuralHydrologyLSTM", "neuralhydrology_lstm_builder"]
