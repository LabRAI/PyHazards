# pyhazards/models/wildfire_fpa_mlp.py
from __future__ import annotations

from typing import Callable, Union

import torch
import torch.nn as nn

from .registry import register_model


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
    if key in ("silu", "swish"):
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name!r}")


class WildfireFPAMLP(nn.Module):
    """
    DNN/MLP for 5-way classification.

    forward:
      x: (B, in_dim)
      returns logits: (B, out_dim)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 5,
        depth: int = 2,
        hidden_dim: int = 64,
        activation: Union[str, Callable[[], nn.Module]] = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim must be > 0, got {in_dim}")
        if out_dim <= 0:
            raise ValueError(f"out_dim must be > 0, got {out_dim}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0,1), got {dropout}")

        layers = []
        d_in = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d_in, hidden_dim))
            layers.append(_activation_from_name(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            d_in = hidden_dim

        layers.append(nn.Linear(d_in, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"MLP expects x as (B, D). Got {tuple(x.shape)}")
        return self.net(x)


def build_wildfire_fpa_mlp(
    *,
    task: str,
    in_dim: int,
    out_dim: int = 5,
    depth: int = 2,
    hidden_dim: int = 64,
    activation: Union[str, Callable[[], nn.Module]] = "relu",
    dropout: float = 0.0,
    **kwargs,
) -> nn.Module:
    # strict: catch typos
    if kwargs:
        raise TypeError(f"Unexpected kwargs for wildfire_fpa_mlp: {sorted(kwargs.keys())}")

    if str(task).lower() != "classification":
        raise ValueError(f"wildfire_fpa_mlp supports only task='classification', got {task!r}")

    return WildfireFPAMLP(
        in_dim=in_dim,
        out_dim=out_dim,
        depth=depth,
        hidden_dim=hidden_dim,
        activation=activation,
        dropout=dropout,
    )


# ✅ register at import time (NO decorator)
register_model(
    name="wildfire_fpa_mlp",
    builder=build_wildfire_fpa_mlp,
    defaults={
        "out_dim": 5,
        "depth": 2,
        "hidden_dim": 64,
        "activation": "relu",
        "dropout": 0.0,
    },
)
