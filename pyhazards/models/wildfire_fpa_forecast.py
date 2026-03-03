from __future__ import annotations

import torch
import torch.nn as nn

from .wildfire_fpa_autoencoder import WildfireFPAAutoencoder


class WildfireFPAForecast(nn.Module):
    """Forecast model that combines an LSTM temporal encoder with an autoencoder latent summary."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 5,
        latent_dim: int = 32,
        num_layers: int = 1,
        ae_hidden_dim: int | None = None,
        ae_num_layers: int | None = None,
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
        self.temporal = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.autoencoder = WildfireFPAAutoencoder(
            input_dim=input_dim,
            hidden_dim=ae_hidden_dim or hidden_dim,
            latent_dim=latent_dim,
            num_layers=ae_num_layers or num_layers,
            dropout=dropout,
            lookback=lookback,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim + latent_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, T, D), got {tuple(x.shape)}")
        if x.size(1) != self.lookback:
            raise ValueError(f"Expected lookback={self.lookback}, got sequence length {x.size(1)}")

        _, (hidden, _) = self.temporal(x)
        latent = self.autoencoder.encode(x)
        return self.head(torch.cat([hidden[-1], latent], dim=-1))

    def forward_with_reconstruction(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        preds = self.forward(x)
        recon = self.autoencoder(x)
        return preds, recon


def wildfire_fpa_forecast_builder(
    task: str,
    input_dim: int,
    hidden_dim: int = 64,
    output_dim: int = 5,
    latent_dim: int = 32,
    num_layers: int = 1,
    ae_hidden_dim: int | None = None,
    ae_num_layers: int | None = None,
    dropout: float = 0.2,
    lookback: int = 50,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() not in {"regression", "forecasting"}:
        raise ValueError(
            f"wildfire_fpa_forecast supports task='regression' or 'forecasting', got {task!r}"
        )
    return WildfireFPAForecast(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        ae_hidden_dim=ae_hidden_dim,
        ae_num_layers=ae_num_layers,
        dropout=dropout,
        lookback=lookback,
    )


__all__ = ["WildfireFPAForecast", "wildfire_fpa_forecast_builder"]
