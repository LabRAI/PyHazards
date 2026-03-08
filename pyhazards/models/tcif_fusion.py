from __future__ import annotations

import torch
import torch.nn as nn


class TCIFFusion(nn.Module):
    """Knowledge-guided fusion baseline for tropical cyclone forecasting."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        horizon: int = 5,
        output_dim: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.output_dim = int(output_dim)
        left_dim = max(1, input_dim // 2)
        right_dim = input_dim - left_dim
        self.left_dim = left_dim
        self.left_encoder = nn.GRU(left_dim, hidden_dim, batch_first=True)
        self.right_encoder = nn.GRU(max(1, right_dim), hidden_dim, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, self.horizon * self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("TCIFFusion expects inputs shaped (batch, history, features).")
        left = x[:, :, : self.left_dim]
        right = x[:, :, self.left_dim :]
        if right.size(-1) == 0:
            right = x[:, :, :1]
        _, left_hidden = self.left_encoder(left)
        _, right_hidden = self.right_encoder(right)
        fused = torch.cat([left_hidden[-1], right_hidden[-1]], dim=-1)
        preds = self.fusion(fused)
        return preds.view(x.size(0), self.horizon, self.output_dim)


def tcif_fusion_builder(
    task: str,
    input_dim: int = 8,
    hidden_dim: int = 64,
    horizon: int = 5,
    output_dim: int = 3,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() != "regression":
        raise ValueError("TCIFFusion only supports regression for track/intensity forecasting.")
    return TCIFFusion(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        horizon=horizon,
        output_dim=output_dim,
        dropout=dropout,
    )


__all__ = ["TCIFFusion", "tcif_fusion_builder"]
