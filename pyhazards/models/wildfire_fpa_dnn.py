from __future__ import annotations

from typing import Callable, Union

import torch
import torch.nn as nn


def _activation_from_name(name: Union[str, Callable[[], nn.Module]]) -> nn.Module:
    if callable(name):
        return name()
    key = str(name).strip().lower()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    if key == "tanh":
        return nn.Tanh()
    if key in {"silu", "swish"}:
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name!r}")


class WildfireFPADNN(nn.Module):
    """DNN classifier for incident-level FPA-FOD features."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 5,
        depth: int = 2,
        hidden_dim: int = 64,
        activation: Union[str, Callable[[], nn.Module]] = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        layers = []
        current_dim = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(_activation_from_name(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected input shape (B, D), got {tuple(x.shape)}")
        return self.net(x)


def wildfire_fpa_dnn_builder(
    task: str,
    in_dim: int,
    out_dim: int = 5,
    depth: int = 2,
    hidden_dim: int = 64,
    activation: Union[str, Callable[[], nn.Module]] = "relu",
    dropout: float = 0.0,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() != "classification":
        raise ValueError(f"wildfire_fpa_dnn supports task='classification', got {task!r}")
    return WildfireFPADNN(
        in_dim=in_dim,
        out_dim=out_dim,
        depth=depth,
        hidden_dim=hidden_dim,
        activation=activation,
        dropout=dropout,
    )


__all__ = ["WildfireFPADNN", "wildfire_fpa_dnn_builder"]
