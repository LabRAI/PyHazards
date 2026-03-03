from __future__ import annotations

from typing import Callable, Union

import torch
import torch.nn as nn

from .wildfire_fpa_dnn import WildfireFPADNN
from .wildfire_fpa_forecast import WildfireFPAForecast


class WildfireFPA(nn.Module):
    """Paper-facing wrapper for the two-stage FPA-FOD wildfire framework."""

    def __init__(self, stage: str, component: nn.Module):
        super().__init__()
        normalized_stage = stage.lower()
        if normalized_stage not in {"classification", "forecasting", "regression"}:
            raise ValueError(f"Unsupported wildfire_fpa stage: {stage!r}")

        self.stage = "forecasting" if normalized_stage == "regression" else normalized_stage
        self.component = component

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.component(x)

    def forward_with_reconstruction(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self.component, "forward_with_reconstruction"):
            raise AttributeError(
                "forward_with_reconstruction is only available for the forecasting stage "
                "of wildfire_fpa."
            )
        return self.component.forward_with_reconstruction(x)


def wildfire_fpa_builder(
    task: str,
    in_dim: int | None = None,
    input_dim: int | None = None,
    out_dim: int | None = None,
    output_dim: int | None = None,
    depth: int = 2,
    hidden_dim: int = 64,
    activation: Union[str, Callable[[], nn.Module]] = "relu",
    dropout: float | None = None,
    latent_dim: int = 32,
    num_layers: int = 1,
    ae_hidden_dim: int | None = None,
    ae_num_layers: int | None = None,
    lookback: int = 50,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    normalized_task = task.lower()

    if normalized_task == "classification":
        feature_dim = in_dim if in_dim is not None else input_dim
        if feature_dim is None:
            raise TypeError("wildfire_fpa classification requires in_dim (or input_dim).")

        component = WildfireFPADNN(
            in_dim=feature_dim,
            out_dim=out_dim if out_dim is not None else (output_dim if output_dim is not None else 5),
            depth=depth,
            hidden_dim=hidden_dim,
            activation=activation,
            dropout=0.0 if dropout is None else dropout,
        )
        return WildfireFPA(stage="classification", component=component)

    if normalized_task in {"forecasting", "regression"}:
        sequence_dim = input_dim if input_dim is not None else in_dim
        if sequence_dim is None:
            raise TypeError("wildfire_fpa forecasting requires input_dim (or in_dim).")

        component = WildfireFPAForecast(
            input_dim=sequence_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim if output_dim is not None else (out_dim if out_dim is not None else 5),
            latent_dim=latent_dim,
            num_layers=num_layers,
            ae_hidden_dim=ae_hidden_dim,
            ae_num_layers=ae_num_layers,
            dropout=0.2 if dropout is None else dropout,
            lookback=lookback,
        )
        return WildfireFPA(stage=normalized_task, component=component)

    raise ValueError(
        "wildfire_fpa supports task='classification' for the DNN stage and "
        "task in {'forecasting', 'regression'} for the LSTM + autoencoder stage."
    )


__all__ = ["WildfireFPA", "wildfire_fpa_builder"]
