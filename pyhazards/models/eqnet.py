from __future__ import annotations

import torch
import torch.nn as nn


class EQNet(nn.Module):
    """Transformer-style earthquake phase-picking baseline."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 48,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=5, padding=2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=2 * hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("EQNet expects inputs shaped (batch, channels, length).")
        seq = self.proj(x).transpose(1, 2)
        encoded = self.encoder(seq)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


def eqnet_builder(
    task: str,
    in_channels: int = 3,
    hidden_dim: int = 48,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() != "regression":
        raise ValueError("EQNet only supports regression-style phase picking outputs.")
    return EQNet(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )


__all__ = ["EQNet", "eqnet_builder"]
