"""LSTM and ConvLSTM next-day wildfire danger classifiers of Kondylatos et al. (2022).

Kondylatos, Prapas, Ronco, Papoutsis, Camps-Valls, Piles, Fernandez-Torres & Carvalhais,
"Wildfire Danger Prediction and Understanding With Deep Learning", Geophysical Research Letters
49(17), e2022GL099368 (https://doi.org/10.1029/2022GL099368).

Ported from Orion-AI-Lab/wildfire_forecasting, ``wildfire_forecasting/models/modules/fire_modules.py``
(``SimpleLSTM``, ``SimpleConvLSTM``), MIT License, Copyright (c) 2022 iprapas. Commit
2b18bcf194284d56bd4e1774f7d5315db94cba34; the model code is unchanged since the v0.1-alpha release
that the paper cites (Zenodo 10.5281/zenodo.6524771). The recurrence of the repository's
``models/modules/convlstm.py`` (from ndrplz/ConvLSTM_pytorch) is provided by
:class:`pyhazards.models.convlstm.ConvLSTM`, which computes the same gates with the same parameter
names; tests/oracle checks the two against each other.

Both models classify one 1 km grid cell as burned / not burned on day ``t`` from the ten previous
days. Each day has 25 features, concatenated as dynamic, static, land cover: 10 dynamic variables
(NDVI, day and night LST, ERA5-Land max 2 m dew point, max 2 m temperature, max surface pressure,
total precipitation, soil moisture index, max wind speed, min relative humidity), 5 static variables
(elevation, slope, distance to roads, distance to waterways, population density) repeated over time,
and the 10 Corine Land Cover class fractions. Within each group the reference dataset orders the
variables as listed in its ``variable_dict.json``.

- ``SimpleLSTM`` reads the cell's ``(batch, 10, 25)`` time series: LayerNorm(25) -> LSTM(25, 64) ->
  last step -> Linear 64-64-32-2 with ReLU and dropout 0.5 -> log-softmax (29,652 parameters).
- ``SimpleConvLSTM`` reads the ``(batch, 10, 25, 25, 25)`` block of 25 x 25 cells centred on the cell:
  LayerNorm over the features of every pixel -> ConvLSTM(25, 32, 3x3) -> last hidden state ->
  Conv2d 3x3 + ReLU + 2x2 max-pool -> Linear 4608-64-32-2 with dropout 0.5 -> log-softmax
  (372,212 parameters).

Both return **log-probabilities** over (no fire, fire), exactly as the reference does; the reference
trains them with ``nn.NLLLoss`` and takes ``exp(output)[:, 1]`` as the fire-danger probability.
Parameter names and initialisation order match the reference, so its state dicts load with
``strict=True`` and the same seed gives the same initial weights.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .convlstm import ConvLSTM

WILDFIRE_FORECASTING_VARIANTS = ("lstm", "convlstm")

# Hidden sizes of the paper runs (configs/experiment/lstm_temporal_cls.yaml and
# clstm_spatiotemporal_cls.yaml, which the reference README names as the paper's hyperparameters).
PAPER_HIDDEN_SIZE = {"lstm": 64, "convlstm": 32}


def _check_common(input_dim: int, hidden_size: int, lstm_layers: int, dropout: float) -> None:
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}")
    if hidden_size < 2:
        raise ValueError(f"hidden_size must be at least 2, got {hidden_size}")
    if lstm_layers <= 0:
        raise ValueError(f"lstm_layers must be positive, got {lstm_layers}")
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}")


class SimpleLSTM(nn.Module):
    """Per-cell LSTM danger classifier (``SimpleLSTM`` in the reference ``fire_modules.py``).

    Input ``(batch, time, input_dim)``; output ``(batch, 2)`` log-probabilities.
    """

    def __init__(self, input_dim: int = 25, hidden_size: int = 64, lstm_layers: int = 1, dropout: float = 0.5):
        super().__init__()
        _check_common(input_dim, hidden_size, lstm_layers, dropout)
        self.input_dim = int(input_dim)
        self.ln1 = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=lstm_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.drop1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size // 2, 2)
        # The reference registers the same layers again inside ``fc_nn``; its state dict therefore
        # holds each fully connected tensor under two keys (``fc1.*`` and ``fc_nn.0.*``, ...).
        self.fc_nn = nn.Sequential(self.fc1, self.relu, self.drop1, self.fc2, self.relu, self.drop2, self.fc3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "SimpleLSTM expects input shape (batch, time, features), e.g. (batch, 10, 25), "
                f"got {tuple(x.shape)}."
            )
        if x.size(-1) != self.input_dim:
            raise ValueError(f"SimpleLSTM expected {self.input_dim} features per day, got shape {tuple(x.shape)}.")
        x = self.ln1(x)
        lstm_out, _ = self.lstm(x)
        x = self.fc_nn(lstm_out[:, -1, :])
        return F.log_softmax(x, dim=1)


class SimpleConvLSTM(nn.Module):
    """Patch ConvLSTM danger classifier (``SimpleConvLSTM`` in the reference ``fire_modules.py``).

    Input ``(batch, time, input_dim, patch_size, patch_size)``; output ``(batch, 2)`` log-probabilities.
    The fully connected head is sized for ``patch_size x patch_size`` patches (25 in the paper).
    """

    def __init__(
        self,
        input_dim: int = 25,
        hidden_size: int = 32,
        lstm_layers: int = 1,
        dropout: float = 0.5,
        patch_size: int = 25,
    ):
        super().__init__()
        _check_common(input_dim, hidden_size, lstm_layers, dropout)
        if patch_size < 2:
            raise ValueError(f"patch_size must be at least 2, got {patch_size}")
        kernel_size = 3
        self.input_dim = int(input_dim)
        self.patch_size = int(patch_size)
        self.ln1 = nn.LayerNorm(input_dim)
        # The reference uses its own ConvLSTM (from ndrplz/ConvLSTM_pytorch) with batch_first=True,
        # bias=True, return_all_layers=False and dilation=1. pyhazards.models.convlstm.ConvLSTM has the
        # same gates, zero initial state and parameter names (``cell_list.{i}.conv``).
        self.convlstm = ConvLSTM(input_dim, hidden_size, kernel_size=(kernel_size, kernel_size), num_layers=lstm_layers)
        self.conv1 = nn.Conv2d(
            hidden_size, hidden_size, kernel_size=(kernel_size, kernel_size), stride=(1, 1), padding=(1, 1)
        )
        self.fc1 = nn.Linear((patch_size // 2) * (patch_size // 2) * hidden_size, 2 * hidden_size)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(2 * hidden_size, hidden_size)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                "SimpleConvLSTM expects input shape (batch, time, features, height, width), "
                f"e.g. (batch, 10, 25, 25, 25), got {tuple(x.shape)}."
            )
        if x.size(2) != self.input_dim:
            raise ValueError(
                f"SimpleConvLSTM expected {self.input_dim} features per day, got shape {tuple(x.shape)}."
            )
        pooled = self.patch_size // 2
        if (x.size(3) // 2, x.size(4) // 2) != (pooled, pooled):
            raise ValueError(
                f"SimpleConvLSTM's head is sized for {self.patch_size}x{self.patch_size} patches, "
                f"got shape {tuple(x.shape)}."
            )
        # LayerNorm over the features of every pixel: (b, t, c, h, w) -> (b, t, h, w, c) -> back.
        x = self.ln1(x.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        _, last_states = self.convlstm(x)
        x = last_states[-1][0]  # hidden state of the last layer at the last step
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.drop1(self.fc1(x)))
        x = F.relu(self.drop2(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def wildfire_forecasting_builder(
    task: str,
    variant: str = "lstm",
    input_dim: int = 25,
    hidden_size: Optional[int] = None,
    lstm_layers: int = 1,
    dropout: float = 0.5,
    patch_size: int = 25,
    **kwargs,
) -> nn.Module:
    """Build the Kondylatos et al. (2022) LSTM (``variant="lstm"``, default) or ConvLSTM.

    ``hidden_size`` defaults to the paper value of the chosen variant (64 for the LSTM, 32 for the
    ConvLSTM). ``patch_size`` only applies to the ConvLSTM. Both models return ``(batch, 2)``
    log-probabilities.
    """
    _ = kwargs
    if task.lower() != "classification":
        raise ValueError(f"wildfire_forecasting supports task='classification', got {task!r}.")
    if variant not in WILDFIRE_FORECASTING_VARIANTS:
        raise ValueError(f"variant must be one of {WILDFIRE_FORECASTING_VARIANTS}, got {variant!r}.")
    if hidden_size is None:
        hidden_size = PAPER_HIDDEN_SIZE[variant]
    if variant == "lstm":
        return SimpleLSTM(input_dim=input_dim, hidden_size=hidden_size, lstm_layers=lstm_layers, dropout=dropout)
    return SimpleConvLSTM(
        input_dim=input_dim,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        dropout=dropout,
        patch_size=patch_size,
    )


__all__ = [
    "PAPER_HIDDEN_SIZE",
    "SimpleConvLSTM",
    "SimpleLSTM",
    "WILDFIRE_FORECASTING_VARIANTS",
    "wildfire_forecasting_builder",
]
