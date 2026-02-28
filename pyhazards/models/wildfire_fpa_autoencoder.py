# pyhazards/models/wildfire_fpa_autoencoder.py
from __future__ import annotations

import torch
import torch.nn as nn

from .registry import register_model


class WildfireFPALSTMAutoencoder(nn.Module):
    """
    LSTM autoencoder for sequences.

    forward:
      x: (B, T, D)
      returns reconstruction x_hat: (B, T, D)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
        lookback: int = 50,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be > 0, got {latent_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0,1), got {dropout}")
        if lookback <= 0:
            raise ValueError(f"lookback must be > 0, got {lookback}")

        self.lookback = lookback
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.to_latent = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.to_recon = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"AE expects x as (B, T, D). Got {tuple(x.shape)}")
        B, T, D = x.shape
        if T != self.lookback:
            raise ValueError(f"Expected lookback T={self.lookback}, got T={T}")

        _, (h_n, _) = self.encoder(x)
        h_last = h_n[-1]  # (B, hidden_dim)
        z = self.to_latent(h_last)  # (B, latent_dim)

        z_seq = z.unsqueeze(1).expand(B, T, z.shape[-1])  # (B, T, latent_dim)
        dec_out, _ = self.decoder(z_seq)
        return self.to_recon(dec_out)

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        x_hat = self.forward(x)
        err = (x_hat - x) ** 2
        if reduction == "none":
            return err
        if reduction == "mean":
            return err.mean(dim=(1, 2))
        if reduction == "sum":
            return err.sum(dim=(1, 2))
        raise ValueError(f"Unknown reduction: {reduction!r}")


def build_wildfire_fpa_ae(
    *,
    task: str,
    input_dim: int,
    hidden_dim: int = 64,
    latent_dim: int = 32,
    num_layers: int = 1,
    dropout: float = 0.2,
    lookback: int = 50,
    **kwargs,
) -> nn.Module:
    if kwargs:
        raise TypeError(f"Unexpected kwargs for wildfire_fpa_ae: {sorted(kwargs.keys())}")

    t = str(task).lower()
    if t not in ("reconstruction", "autoencoder", "regression", "forecasting"):
        raise ValueError(
            "wildfire_fpa_ae supports task in "
            "{'reconstruction','autoencoder','regression','forecasting'}, "
            f"got {task!r}"
        )

    return WildfireFPALSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        dropout=dropout,
        lookback=lookback,
    )


register_model(
    name="wildfire_fpa_ae",
    builder=build_wildfire_fpa_ae,
    defaults={
        "hidden_dim": 64,
        "latent_dim": 32,
        "num_layers": 1,
        "dropout": 0.2,
        "lookback": 50,
    },
)
