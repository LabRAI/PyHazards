from __future__ import annotations

import torch
import torch.nn as nn


class WildfireFPAAutoencoder(nn.Module):
    """Autoencoder block used in the FPA-FOD forecasting stack."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
        lookback: int = 50,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
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
        self.to_latent = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.to_reconstruction = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, T, D), got {tuple(x.shape)}")
        _, sequence_length, _ = x.shape
        if sequence_length != self.lookback:
            raise ValueError(f"Expected lookback={self.lookback}, got sequence length {sequence_length}")
        _, (hidden, _) = self.encoder(x)
        return self.to_latent(hidden[-1])

    def decode(self, latent: torch.Tensor, sequence_length: int | None = None) -> torch.Tensor:
        if latent.ndim != 2:
            raise ValueError(f"Expected latent shape (B, Z), got {tuple(latent.shape)}")
        sequence_length = sequence_length or self.lookback
        batch_size = latent.size(0)
        repeated = latent.unsqueeze(1).expand(batch_size, sequence_length, latent.size(-1))
        decoded, _ = self.decoder(repeated)
        return self.to_reconstruction(decoded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        return self.decode(latent, sequence_length=x.size(1))

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        residual = (self.forward(x) - x) ** 2
        if reduction == "none":
            return residual
        if reduction == "mean":
            return residual.mean(dim=(1, 2))
        if reduction == "sum":
            return residual.sum(dim=(1, 2))
        raise ValueError(f"Unknown reduction: {reduction!r}")


def wildfire_fpa_autoencoder_builder(
    task: str,
    input_dim: int,
    hidden_dim: int = 64,
    latent_dim: int = 32,
    num_layers: int = 1,
    dropout: float = 0.2,
    lookback: int = 50,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() not in {"reconstruction", "autoencoder", "regression", "forecasting"}:
        raise ValueError(
            "wildfire_fpa_autoencoder supports task in "
            "{'reconstruction', 'autoencoder', 'regression', 'forecasting'}."
        )
    return WildfireFPAAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        dropout=dropout,
        lookback=lookback,
    )

__all__ = [
    "WildfireFPAAutoencoder",
    "wildfire_fpa_autoencoder_builder",
]
