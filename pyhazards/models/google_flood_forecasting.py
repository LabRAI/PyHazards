"""Google flood-forecasting LSTM: hindcast / forecast LSTMs with masked-mean input embeddings and a CMAL head.

Port of ``googlehydrology/modelzoo/mean_embedding_forecast_lstm.py`` (``MeanEmbeddingForecastLSTM``),
``fc.py`` (``FC``), ``head.py`` (``CMAL``, ``Regression``) and ``utils/lstm_utils.py`` (``lstm_init``) of
google-research/flood-forecasting at commit cdda28dda4cc6f1c5e3aaafcc69c6602c9e7cda1
(https://github.com/google-research/flood-forecasting, Apache License 2.0, Copyright 2025 Google LLC;
googlehydrology is a fork of NeuralHydrology, BSD-3-Clause). Changes: plain constructor arguments instead
of the googlehydrology ``Config``; the data-assimilation hooks (``assimilation_overrides``,
``return_embeddings``) and the hot-start state files (``save_state`` / ``load_state_from_disk``) are not
ported; inputs may also be given as tensors whose columns follow the configured feature order.

This is the model of Google's released FloodHub weights (Gauch et al., "How to deal w___ missing input
data", HESS 29:6221-6235, 2025), the successor of the hindcast/forecast LSTM of Nearing et al. (Nature
627:559-563, 2024). Static attributes are embedded by an FC network; every dynamic input group (one
weather product, e.g. ``hres``, ``graphcast``, ``imerg``, ``cpc``) is embedded together with the static
embedding by its own FC network, and the group embeddings are averaged with a NaN-aware mean, so a
missing product is skipped. A hindcast LSTM runs over the averaged hindcast embedding; a forecast LSTM
runs over the averaged forecast embedding plus the hindcast LSTM's outputs. The CMAL head gives, for
every time step, a mixture of ``n_distributions`` asymmetric Laplace distributions (``mu``, ``b``,
``tau``, ``pi``). Steps from the first one with no valid input on are NaN, as in the reference.

``pretrained=True`` downloads the released ``model_epoch110.pt`` (13.6 MB, pinned commit and sha256;
covered by the repository's Apache-2.0 licence) into the torch hub cache and loads it with
``strict=True`` after stripping the ``_orig_mod.`` prefix that ``torch.compile`` added. The release
expects inputs normalised with its ``scaler.zarr`` (not read by PyHazards).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

LSTM_IH_XAVIER = "lstm-ih-xavier"
LSTM_HH_ORTHOGONAL = "lstm-hh-orthogonal"
FC_XAVIER = "fc-xavier"

FLOODHUB_STATIC_ATTRIBUTES = [
    "p_mean", "pet_mean_ERA5_LAND", "aridity_ERA5_LAND", "frac_snow", "moisture_index_ERA5_LAND",
    "seasonality_ERA5_LAND", "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur",
    "aet_mm_syr", "ari_ix_sav", "crp_pc_sse", "ele_mt_sav", "ero_kh_sav", "for_pc_sse", "gdp_ud_ssu",
    "gla_pc_sse", "glc_pc_s01", "glc_pc_s02", "glc_pc_s03", "glc_pc_s04", "glc_pc_s06", "glc_pc_s07",
    "glc_pc_s08", "glc_pc_s09", "glc_pc_s10", "glc_pc_s11", "glc_pc_s12", "glc_pc_s13", "glc_pc_s14",
    "glc_pc_s15", "glc_pc_s16", "glc_pc_s17", "glc_pc_s18", "glc_pc_s19", "glc_pc_s20", "glc_pc_s21",
    "glc_pc_s22", "hft_ix_s09", "hft_ix_s93", "inu_pc_slt", "inu_pc_smn", "inu_pc_smx", "ire_pc_sse",
    "kar_pc_sse", "lka_pc_sse", "nli_ix_sav", "pac_pc_sse", "pet_mm_syr", "pnv_pc_s01", "pnv_pc_s02",
    "pnv_pc_s03", "pnv_pc_s04", "pnv_pc_s05", "pnv_pc_s06", "pnv_pc_s07", "pnv_pc_s08", "pnv_pc_s09",
    "pnv_pc_s10", "pnv_pc_s11", "pnv_pc_s12", "pnv_pc_s13", "pnv_pc_s14", "pnv_pc_s15", "ppd_pk_sav",
    "pre_mm_syr", "prm_pc_sse", "rdd_mk_sav", "snw_pc_syr", "swc_pc_syr", "tmp_dc_syr", "urb_pc_sse",
    "wet_pc_s01", "wet_pc_s02", "wet_pc_s03", "wet_pc_s04", "wet_pc_s05", "wet_pc_s06", "wet_pc_s07",
    "wet_pc_s08", "wet_pc_s09", "wet_pc_sg1", "wet_pc_sg2",
]
_HRES = [
    "hres_surface_net_solar_radiation",
    "hres_surface_net_thermal_radiation",
    "hres_surface_pressure",
    "hres_temperature_2m",
    "hres_total_precipitation",
]
_GRAPHCAST = ["graphcast_temperature_2m", "graphcast_total_precipitation"]

# pretrained-models/google-floodhub-settings-110-epochs/config.yml at the pinned commit.
FLOODHUB_CONFIG: Dict[str, Any] = {
    "static_attributes": FLOODHUB_STATIC_ATTRIBUTES,
    "hindcast_inputs": {"hres": _HRES, "graphcast": _GRAPHCAST, "imerg": ["imerg_precipitation"], "cpc": ["cpc_precipitation"]},
    "forecast_inputs": {"hres": _HRES, "graphcast": _GRAPHCAST},
    "seq_length": 365,
    "lead_time": 7,
    "hidden_size": 512,
    "statics_embedding": {"hiddens": [100, 100, 20], "activation": ["tanh", "tanh", "linear"], "dropout": 0.0},
    "hindcast_embedding": {"hiddens": [100, 20], "activation": ["tanh", "linear"], "dropout": 0.0},
    "forecast_embedding": {"hiddens": [20, 20, 20, 20], "activation": ["tanh", "tanh", "tanh", "linear"], "dropout": 0.0},
    "head": "cmal",
    "n_distributions": 3,
    "n_targets": 1,
    "output_dropout": 0.4,
    "initial_forget_bias": 3.0,
    "weight_init_opts": [LSTM_IH_XAVIER, LSTM_HH_ORTHOGONAL, FC_XAVIER],
}

FLOODHUB_WEIGHTS = {
    "url": (
        "https://raw.githubusercontent.com/google-research/flood-forecasting/"
        "cdda28dda4cc6f1c5e3aaafcc69c6602c9e7cda1/pretrained-models/google-floodhub-settings-110-epochs/"
        "model_epoch110.pt"
    ),
    "sha256": "90280f0687a5a95006580164c243be6f8551580f772be26247c8b0ef3d53fdba",
    "filename": "google_floodhub_settings_110_epochs_model_epoch110.pt",
}


def _unique(features: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(features))


def _embedding_spec(spec: Mapping[str, Any], label: str) -> Dict[str, Any]:
    if spec is None:
        raise ValueError(f"{label} embedding specification is required.")
    hiddens = [int(h) for h in spec.get("hiddens", [])]
    if not hiddens:
        raise ValueError(f"{label} embedding needs at least one entry in 'hiddens'.")
    activation = spec.get("activation", "tanh")
    activation = list(activation) if isinstance(activation, (list, tuple)) else [activation] * len(hiddens)
    if len(activation) != len(hiddens):
        raise ValueError(f"{label} embedding: hiddens and activation layers must match.")
    if spec.get("type", "fc").lower() != "fc":
        raise ValueError(f"{label} embedding type {spec.get('type')!r} not supported (only 'fc').")
    return {"hiddens": hiddens, "activation": activation, "dropout": float(spec.get("dropout", 0.0))}


class FC(nn.Module):
    """googlehydrology ``FC``: Linear / activation / Dropout blocks and a linear output layer.

    Initialisation: uniform(-sqrt(3 / n_in), sqrt(3 / n_in)) weights, or, with ``xavier_init``,
    ``xavier_uniform_`` with gain sqrt(3 / n_in) (the reference passes that bound as the gain); zero bias.
    """

    def __init__(self, input_size: int, hidden_sizes: Sequence[int], activation: Union[str, Sequence[str]] = "tanh",
                 dropout: float = 0.0, xavier_init: bool = False):
        super().__init__()
        self._xavier_init = xavier_init
        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must at least have one entry to create a fully-connected net.")
        self.output_size = hidden_sizes[-1]
        hidden_sizes = list(hidden_sizes[:-1])
        if isinstance(activation, str):
            activations = [self._get_activation(activation)] * len(hidden_sizes)
        else:
            activations = [self._get_activation(e) for e in activation]
        layers: List[nn.Module] = []
        if hidden_sizes:
            for i, hidden_size in enumerate(hidden_sizes):
                layers.append(nn.Linear(input_size if i == 0 else hidden_sizes[i - 1], hidden_size))
                layers.append(activations[i])
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(hidden_sizes[-1], self.output_size))
        else:
            layers.append(nn.Linear(input_size, self.output_size))
        self.net = nn.Sequential(*layers)
        self._reset_parameters()

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        activations = {"tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "relu": nn.ReLU, "linear": nn.Identity}
        if name.lower() not in activations:
            raise ValueError(f"{name} currently not supported as activation in this class")
        return activations[name.lower()]()

    def _reset_parameters(self) -> None:
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                n_in = layer.weight.shape[1]
                gain = np.sqrt(3 / n_in)
                if self._xavier_init:
                    nn.init.xavier_uniform_(layer.weight, gain)
                else:
                    nn.init.uniform_(layer.weight, -gain, gain)
                nn.init.constant_(layer.bias, val=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CMAL(nn.Module):
    """Countable mixture of asymmetric Laplacians (Klotz et al., HESS 2022): ``mu``, ``b``, ``tau``, ``pi``."""

    def __init__(self, n_in: int, n_out: int, n_hidden: int = 100, n_distributions: Optional[int] = None):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self.n_distributions = n_distributions
        self._softplus = torch.nn.Softplus(2)
        self._eps = 1e-5

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = torch.relu(self.fc1(x))
        h = self.fc2(h)
        m_latent, b_latent, t_latent, p_latent = h.chunk(4, dim=-1)
        m = m_latent
        b = self._softplus(b_latent) + self._eps
        t = (1 - self._eps) * torch.sigmoid(t_latent) + self._eps
        p = (1 - self._eps) * torch.softmax(p_latent, dim=-1) + self._eps
        return {"mu": m, "b": b, "tau": t, "pi": p}

    def point_prediction(self, outputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Mean of the predicted mixture, ``(batch, time, n_targets)``."""
        mu, b, tau, pi = (outputs[k] for k in ("mu", "b", "tau", "pi"))
        n_distributions = self.n_distributions or mu.shape[-1]
        shape = (*mu.shape[:-1], -1, n_distributions)
        mu, b, tau, pi = (e.reshape(shape) for e in (mu, b, tau, pi))
        pi = pi / pi.sum(dim=-1, keepdim=True)
        tau = torch.clamp(tau, min=1e-6, max=1.0 - 1e-6)
        means = mu + b * (1 - 2 * tau) / (tau * (1 - tau))
        return torch.sum(pi * means, dim=-1)


class RegressionHead(nn.Module):
    """googlehydrology ``Regression`` head (single linear layer, optional ReLU / Softplus)."""

    def __init__(self, n_in: int, n_out: int, activation: str = "linear"):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(n_in, n_out)]
        if activation != "linear":
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "softplus":
                layers.append(nn.Softplus())
            else:
                raise ValueError(f"output_activation must be 'linear', 'relu' or 'softplus', got {activation!r}.")
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"y_hat": self.net(x)}

    def point_prediction(self, outputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return outputs["y_hat"]


def lstm_init(lstms: Iterable[nn.LSTM], forget_bias: Optional[float], weight_opts: Iterable[str]) -> None:
    """Forget-gate bias of ``bias_hh``, Xavier ``weight_ih`` and orthogonal ``weight_hh`` (as configured)."""
    weight_opts = set(weight_opts)
    with torch.no_grad():
        for lstm in lstms:
            for name, param in lstm.named_parameters():
                if forget_bias is not None and name.startswith("bias_hh_"):
                    param.data[slice(lstm.hidden_size, 2 * lstm.hidden_size)] = forget_bias
                elif "weight_ih" in name and LSTM_IH_XAVIER in weight_opts:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name and LSTM_HH_ORTHOGONAL in weight_opts:
                    nn.init.orthogonal_(param)


class GoogleFloodForecasting(nn.Module):
    """``MeanEmbeddingForecastLSTM`` of googlehydrology (see the module docstring).

    Inputs: ``x_s`` (batch, n_static); ``x_d_hindcast`` and ``x_d_forecast``, each either a dict of
    per-feature tensors (batch, time, 1), as googlehydrology passes them, or one tensor (batch, time,
    n_features) with the columns in the order of :attr:`hindcast_features` / :attr:`forecast_features`.
    Hindcast series have at most ``seq_length + lead_time`` steps (they are NaN-padded to that length) and
    forecast series exactly ``seq_length + lead_time``. When every group is shared between hindcast and
    forecast and ``lead_time == 0``, ``{"x_d": ..., "x_s": ...}`` feeds both (PyHazards streamflow layout).
    Output: the head's dict, e.g. ``mu``, ``b``, ``tau``, ``pi`` of shape (batch, seq_length + lead_time,
    n_targets * n_distributions).
    """

    def __init__(
        self,
        static_attributes: Union[int, Sequence[str]],
        hindcast_inputs: Mapping[str, Sequence[str]],
        forecast_inputs: Mapping[str, Sequence[str]],
        seq_length: int = 365,
        lead_time: int = 7,
        hidden_size: int = 512,
        statics_embedding: Optional[Mapping[str, Any]] = None,
        hindcast_embedding: Optional[Mapping[str, Any]] = None,
        forecast_embedding: Optional[Mapping[str, Any]] = None,
        head: str = "cmal",
        n_distributions: int = 3,
        n_targets: int = 1,
        output_dropout: float = 0.4,
        initial_forget_bias: Optional[float] = 3.0,
        weight_init_opts: Sequence[str] = (LSTM_IH_XAVIER, LSTM_HH_ORTHOGONAL, FC_XAVIER),
        output_activation: str = "linear",
    ):
        super().__init__()
        if isinstance(static_attributes, int):
            static_attributes = [f"static_{i}" for i in range(static_attributes)]
        self.static_attributes = list(static_attributes)
        if not self.static_attributes:
            raise ValueError("Cannot create embedding layer with input size 0: static attributes are required.")
        self.hindcast_inputs_grouped = {str(k): _unique(v) for k, v in hindcast_inputs.items()}
        self.forecast_inputs_grouped = {str(k): _unique(v) for k, v in forecast_inputs.items()}
        if not self.hindcast_inputs_grouped or not self.forecast_inputs_grouped:
            raise ValueError("hindcast_inputs and forecast_inputs each need at least one feature group.")
        self.shared_groups = [g for g in self.hindcast_inputs_grouped if g in self.forecast_inputs_grouped]
        for group in self.shared_groups:
            if self.hindcast_inputs_grouped[group] != self.forecast_inputs_grouped[group]:
                raise ValueError(f"Same features must be defined in forecast and hindcast for group={group!r}.")
        self.hindcast_features = _unique(f for g in self.hindcast_inputs_grouped.values() for f in g)
        self.forecast_features = _unique(f for g in self.forecast_inputs_grouped.values() for f in g)
        if int(seq_length) <= 0 or int(lead_time) < 0 or int(hidden_size) <= 0:
            raise ValueError("seq_length and hidden_size must be positive and lead_time >= 0.")
        self.seq_length = int(seq_length)
        self.lead_time = int(lead_time)
        self.hidden_size = int(hidden_size)
        self.weight_init_opts = list(weight_init_opts)
        unknown = set(self.weight_init_opts) - {LSTM_IH_XAVIER, LSTM_HH_ORTHOGONAL, FC_XAVIER}
        if unknown:
            raise ValueError(f"unsupported weight_init_opts: {sorted(unknown)}")
        statics = _embedding_spec(statics_embedding or FLOODHUB_CONFIG["statics_embedding"], "statics")
        hindcast = _embedding_spec(hindcast_embedding or FLOODHUB_CONFIG["hindcast_embedding"], "hindcast")
        forecast = _embedding_spec(forecast_embedding or FLOODHUB_CONFIG["forecast_embedding"], "forecast")
        self.head_type = head.lower()
        if self.head_type not in ("cmal", "regression"):
            raise ValueError(f"head must be 'cmal' or 'regression', got {head!r}.")
        output_size = int(n_targets) * (4 * int(n_distributions) if self.head_type == "cmal" else 1)

        # Same creation order as the reference (initialisation draws from the RNG in this order).
        self.static_embedding_fc = self._create_fc(statics, len(self.static_attributes))
        static_size = self.static_embedding_fc.output_size
        self.hindcast_embeddings_fc = nn.ModuleDict(
            {
                name: self._create_fc(hindcast, len(features) + static_size)
                for name, features in self.hindcast_inputs_grouped.items()
                if name not in self.shared_groups
            }
        )
        self.forecast_embeddings_fc = nn.ModuleDict(
            {
                name: self._create_fc(forecast, len(features) + static_size)
                for name, features in self.forecast_inputs_grouped.items()
                if name not in self.shared_groups
            }
        )
        self.shared_embeddings_fc = nn.ModuleDict(
            {
                name: self._create_fc(forecast, len(self.forecast_inputs_grouped[name]) + static_size)
                for name in self.shared_groups
            }
        )
        if hindcast["hiddens"][-1] != forecast["hiddens"][-1] and self.shared_groups:
            raise ValueError("Shared groups feed the hindcast mean, so hindcast and forecast embeddings must have the same size.")
        self.hindcast_lstm = nn.LSTM(
            input_size=static_size + hindcast["hiddens"][-1], hidden_size=self.hidden_size, batch_first=True
        )
        self.forecast_lstm = nn.LSTM(
            input_size=static_size + forecast["hiddens"][-1] + self.hidden_size,
            hidden_size=self.hidden_size,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=output_dropout)
        if self.head_type == "cmal":
            self.head = CMAL(n_in=self.hidden_size, n_out=output_size, n_hidden=100, n_distributions=int(n_distributions))
        else:
            self.head = RegressionHead(n_in=self.hidden_size, n_out=output_size, activation=output_activation)
        lstm_init([self.hindcast_lstm, self.forecast_lstm], initial_forget_bias, self.weight_init_opts)

    def _create_fc(self, spec: Mapping[str, Any], input_size: int) -> FC:
        if input_size <= 0:
            raise ValueError("Cannot create embedding layer with input size 0")
        return FC(
            input_size=input_size,
            hidden_sizes=spec["hiddens"],
            activation=spec["activation"],
            dropout=spec["dropout"],
            xavier_init=FC_XAVIER in self.weight_init_opts,
        )

    # -- inputs ---------------------------------------------------------------------------------------
    def _group_tensors(self, data: Any, features: List[str], grouped: Dict[str, List[str]], label: str):
        if isinstance(data, Mapping):
            missing = [f for g in grouped.values() for f in g if f not in data]
            if missing:
                raise ValueError(f"x_d_{label} is missing features {missing}.")
            return {name: torch.cat([data[f] for f in feats], dim=-1) for name, feats in grouped.items()}
        if not isinstance(data, torch.Tensor) or data.ndim != 3 or data.shape[-1] != len(features):
            shape = tuple(data.shape) if isinstance(data, torch.Tensor) else type(data).__name__
            raise ValueError(
                f"x_d_{label} must be a dict of (batch, time, 1) tensors or a tensor of shape "
                f"(batch, time, {len(features)}) with columns {features}; got {shape}."
            )
        index = {f: i for i, f in enumerate(features)}
        return {name: data[..., [index[f] for f in feats]] for name, feats in grouped.items()}

    def _forward_data(self, data: Mapping[str, Any]):
        if not isinstance(data, Mapping) or "x_s" not in data:
            raise ValueError("GoogleFloodForecasting expects a mapping with 'x_s' and the dynamic inputs.")
        if "x_d" in data and "x_d_hindcast" not in data and "x_d_forecast" not in data:
            if self.lead_time != 0 or self.hindcast_inputs_grouped != self.forecast_inputs_grouped:
                raise ValueError(
                    "'x_d' (streamflow layout) needs lead_time=0 and identical hindcast and forecast groups; "
                    "pass x_d_hindcast and x_d_forecast instead."
                )
            hindcast_data = forecast_data = data["x_d"]
        else:
            if "x_d_hindcast" not in data or "x_d_forecast" not in data:
                raise ValueError("GoogleFloodForecasting expects 'x_d_hindcast' and 'x_d_forecast'.")
            hindcast_data, forecast_data = data["x_d_hindcast"], data["x_d_forecast"]
        x_s = data["x_s"]
        if not isinstance(x_s, torch.Tensor) or x_s.ndim != 2 or x_s.shape[1] != len(self.static_attributes):
            shape = tuple(x_s.shape) if isinstance(x_s, torch.Tensor) else type(x_s).__name__
            raise ValueError(f"x_s must have shape (batch, {len(self.static_attributes)}), got {shape}.")
        hindcast = self._group_tensors(hindcast_data, self.hindcast_features, self.hindcast_inputs_grouped, "hindcast")
        forecast = self._group_tensors(forecast_data, self.forecast_features, self.forecast_inputs_grouped, "forecast")
        total = self.seq_length + self.lead_time
        for name, tensor in hindcast.items():
            if name not in self.shared_groups and (tensor.shape[0] != x_s.shape[0] or tensor.shape[1] > total):
                raise ValueError(f"Hindcast group {name!r} must have shape (batch, <= {total}, ...), got {tuple(tensor.shape)}.")
        for name, tensor in forecast.items():
            if tensor.shape[0] != x_s.shape[0] or tensor.shape[1] != total:
                raise ValueError(
                    f"Forecast group {name!r} must have shape (batch, seq_length + lead_time = {total}, ...), "
                    f"got {tuple(tensor.shape)}."
                )
        return x_s, hindcast, forecast

    # -- forward (reference logic) ---------------------------------------------------------------------
    def forward(self, data: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        x_s, hindcast_features, forecast_features = self._forward_data(data)
        static_embedding = self.static_embedding_fc(x_s)
        mean_hindcast_embedding, mean_forecast_embedding = self._calc_mean_embeddings(
            hindcast_features, forecast_features, static_embedding
        )
        hindcast_missing = self._missing_steps(mean_hindcast_embedding)
        forecast_missing = hindcast_missing | self._missing_steps(mean_forecast_embedding)
        hindcast_state = self._calc_lstm(self.hindcast_lstm, mean_hindcast_embedding, static_embedding)
        forecast_state = self._calc_lstm(
            self.forecast_lstm, mean_forecast_embedding, static_embedding, other_inputs=hindcast_state
        )
        return {
            key: value.masked_fill(forecast_missing, float("nan"))
            for key, value in self.head(self.dropout(forecast_state)).items()
        }

    def point_prediction(self, outputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Deterministic prediction ``(batch, time, n_targets)`` (the CMAL mixture mean)."""
        return self.head.point_prediction(outputs)

    def _calc_mean_embeddings(self, hindcast_features, forecast_features, static_embedding):
        def embed(networks: nn.ModuleDict, features: Dict[str, torch.Tensor], append_nan: bool) -> List[torch.Tensor]:
            return [
                self._calc_dynamic_embedding(fc, features[name], static_embedding, append_nan)
                for name, fc in networks.items()
            ]

        hindcast = embed(self.hindcast_embeddings_fc, hindcast_features, append_nan=True)
        forecast = embed(self.forecast_embeddings_fc, forecast_features, append_nan=False)
        shared = embed(self.shared_embeddings_fc, forecast_features, append_nan=False)  # shared use forecast data
        return self._masked_mean(hindcast + shared), self._masked_mean(forecast + shared)

    @staticmethod
    def _append_static_embedding(embedding: torch.Tensor, static_embedding: torch.Tensor) -> torch.Tensor:
        repeated = static_embedding.unsqueeze(1).repeat(1, embedding.shape[1], 1)
        return torch.cat([embedding, repeated], dim=-1)

    def _add_nan_padding(self, embedding: torch.Tensor) -> torch.Tensor:
        padding = torch.full(
            (embedding.shape[0], self.seq_length + self.lead_time - embedding.shape[1], embedding.shape[2]),
            np.nan,
            device=embedding.device,
        )
        return torch.cat([embedding, padding], dim=1)

    @staticmethod
    def _masked_mean(tensors: List[torch.Tensor]) -> torch.Tensor:
        merged = torch.cat([e.unsqueeze(-1) for e in tensors], dim=-1)
        return torch.nanmean(merged, dim=-1)

    def _calc_dynamic_embedding(self, network: nn.Module, dynamic_data: torch.Tensor,
                                static_embedding: torch.Tensor, append_nan: bool) -> torch.Tensor:
        # Steps with missing inputs are zeroed before the network and set back to NaN afterwards.
        nan_mask = torch.isnan(dynamic_data).any(dim=-1, keepdim=True)
        dynamic_data = dynamic_data.masked_fill(nan_mask, 0.0)
        output = network(self._append_static_embedding(dynamic_data, static_embedding))
        output = output.masked_fill(nan_mask, float("nan"))
        if append_nan:
            output = self._add_nan_padding(output)
        return output

    @staticmethod
    def _missing_steps(masked_mean_embedding: torch.Tensor) -> torch.Tensor:
        """Steps with no valid input and every step after the first such step, (batch, time, 1)."""
        missing = torch.isnan(masked_mean_embedding).any(dim=-1, keepdim=True)
        return torch.cummax(missing.to(torch.int8), dim=1).values.bool()

    def _calc_lstm(self, lstm: nn.LSTM, masked_mean_embeddings: torch.Tensor, static_embedding: torch.Tensor,
                   other_inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        if other_inputs is not None:
            masked_mean_embeddings = torch.cat([masked_mean_embeddings, other_inputs], dim=-1)
        lstm_inputs = self._append_static_embedding(masked_mean_embeddings, static_embedding)
        lstm_inputs = lstm_inputs.nan_to_num(nan=0.0)  # missing steps are zero-filled; outputs masked later
        output, _ = lstm(input=lstm_inputs)
        return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def floodhub_checkpoint_path(cache_dir: Optional[Union[str, Path]] = None) -> Path:
    """Local copy of the released FloodHub weights, downloaded (and sha256-checked) on first use."""
    root = Path(cache_dir) if cache_dir is not None else Path(torch.hub.get_dir()) / "checkpoints"
    path = root / FLOODHUB_WEIGHTS["filename"]
    if path.exists() and _sha256(path) == FLOODHUB_WEIGHTS["sha256"]:
        return path
    root.mkdir(parents=True, exist_ok=True)
    torch.hub.download_url_to_file(FLOODHUB_WEIGHTS["url"], str(path), hash_prefix=FLOODHUB_WEIGHTS["sha256"], progress=False)
    return path


def load_floodhub_state_dict(path: Optional[Union[str, Path]] = None) -> Dict[str, torch.Tensor]:
    """State dict of the released model (``_orig_mod.`` prefix of the compiled model removed)."""
    path = floodhub_checkpoint_path() if path is None else Path(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    prefix = "_orig_mod."
    return {(key[len(prefix):] if key.startswith(prefix) else key): value for key, value in state.items()}


def _streamflow_config(n_dynamic: int, n_static: int) -> Dict[str, Any]:
    features = [f"x_d_{i}" for i in range(int(n_dynamic))]
    return {
        **FLOODHUB_CONFIG,
        "static_attributes": int(n_static),
        "hindcast_inputs": {"x_d": features},
        "forecast_inputs": {"x_d": features},
        "lead_time": 0,
    }


def google_flood_forecasting_builder(
    task: str,
    config: Optional[str] = "floodhub",
    pretrained: bool = False,
    weights_path: Optional[Union[str, Path]] = None,
    n_dynamic: int = 5,
    n_static: int = 27,
    **overrides: Any,
) -> nn.Module:
    """Build the model.

    ``config="floodhub"``: the released FloodHub model (84 static attributes; hindcast groups hres,
    graphcast, imerg, cpc; forecast groups hres, graphcast; 365 + 7 days; hidden size 512; CMAL with 3
    components; 3,402,832 parameters). ``config="streamflow"``: the same architecture with one shared
    group of ``n_dynamic`` daily inputs, ``n_static`` attributes and ``lead_time=0``, so it reads the
    PyHazards streamflow layout ``{"x_d", "x_s"}`` (a PyHazards adaptation, not a Google configuration).
    ``config=None``: everything from ``overrides``. Keyword ``overrides`` replace configuration entries
    (``hidden_size``, ``seq_length``, embedding specs, ...). ``pretrained=True`` (or ``weights_path``)
    loads the released weights with ``strict=True``.
    """
    overrides.pop("name", None)
    if task.lower() not in ("regression", "streamflow"):
        raise ValueError(f"google_flood_forecasting supports task='regression' (streamflow), got {task!r}.")
    if config == "floodhub":
        cfg = dict(FLOODHUB_CONFIG)
    elif config == "streamflow":
        cfg = _streamflow_config(n_dynamic, n_static)
    elif config is None:
        cfg = {}
    else:
        raise ValueError(f"config must be 'floodhub', 'streamflow' or None, got {config!r}.")
    unknown = set(overrides) - set(FLOODHUB_CONFIG) - {"output_activation"}
    if unknown:
        raise ValueError(f"Unexpected arguments for google_flood_forecasting: {sorted(unknown)}")
    cfg.update(overrides)
    model = GoogleFloodForecasting(**cfg)
    if pretrained or weights_path is not None:
        model.load_state_dict(load_floodhub_state_dict(weights_path), strict=True)
    return model


__all__ = [
    "CMAL",
    "FC",
    "FLOODHUB_CONFIG",
    "FLOODHUB_STATIC_ATTRIBUTES",
    "FLOODHUB_WEIGHTS",
    "GoogleFloodForecasting",
    "floodhub_checkpoint_path",
    "google_flood_forecasting_builder",
    "load_floodhub_state_dict",
    "lstm_init",
]
