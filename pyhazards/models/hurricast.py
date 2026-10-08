"""Hurricast (HUML): multimodal 24-hour tropical cyclone intensity and track forecasts.

Paper: L. Boussioux, C. Zeng, T. Guenais and D. Bertsimas, "Hurricane Forecasting: A Novel Multimodal
Machine Learning Framework", Weather and Forecasting 37(6), 817-831 (2022), doi:10.1175/WAF-D-21-0091.1
(preprint arXiv 2011.06125).

The official repository (github.com/leobix/hurricast) has no LICENSE file (its README claims MIT and
links a LICENSE that does not exist), so this module is written from the paper (Section 3 and Appendix
A1) and the configuration values of the released code; the official code is used only as a test
oracle (tests/oracle/test_hurricast_oracle.py) and is never copied. Module and parameter names follow
the official ``ExperimentalHurricast`` so that its state dicts load with ``strict=True`` into
:class:`HurricastEncoderDecoder` (``Hurricast.network``), and parameters are created in the official
order, so the same seed gives the same initial weights.

Hurricast forecasts one 24-hour lead time in three steps:

1. **Feature extraction.** For 8 three-hourly time steps (t-21 h ... t), the nine ERA5 maps (u, v and
   geopotential z at 225, 500 and 700 hPa, 25 x 25 degrees at 1 degree around the storm) of every step
   go through a CNN encoder (three 3 x 3 convolutions with BatchNorm and ReLU, two 2 x 2 max poolings,
   then dense layers) to a 128-d embedding. The 14 statistical features of the step are concatenated in
   front of it and the 8-step sequence goes through a decoder: a Transformer (2 layers, 2 heads, model
   width 142, feed-forward 128, sinusoidal positional encoding, mean pooling over time, tanh) or a
   recurrent decoder. A linear layer predicts the standardised 24-hour intensity (``c = 1``) or the
   latitude / longitude displacement (``c = 2``); the network is trained with MSE plus an L2 penalty and
   then frozen.
2. **Fusion.** The decoder output (everything but the last linear layer) is the reanalysis embedding.
   It is appended to the flattened statistical data of the 8 steps.
3. **Forecast.** One XGBoost regressor per target (intensity; latitude displacement; longitude
   displacement) is trained on these vectors.

:class:`Hurricast` holds both stages: ``network`` (a :class:`HurricastEncoderDecoder`) and ``xgboost``
(a :class:`pyhazards.models.classical.EstimatorModule`). ``predictor="network"`` uses the network's
own linear head instead of XGBoost; ``use_embeddings=False`` gives HUML-(stat, xgb), XGBoost on the
statistical data alone.

Paper and released code disagree in places; the released code is followed where it is the only
executable reference (it is what the oracle verifies) and the differences are listed in the model card:
the code's CNN has dense layers 4096 -> 576 -> 256 -> 128 (the paper's Fig. A1 shows 576 -> 128); the
code concatenates ``[statistics, embedding]`` (the paper writes ``[embedding, statistics]``); the paper's
GRU decoder (two unidirectional layers, all 8 hidden states concatenated, fully connected
1024-512-128-c) is not in the released code, whose recurrent presets are bidirectional LSTM / GRU /
RNN decoders reading the last hidden state (``lstm_config*``); the paper lists 31 statistical
features per step, the code's feature table has 30 (:data:`HURRICAST_STAT_FEATURES`) and the network
reads the first 14 (``d_model = 142 = 128 + 14``).
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------------------------------
# Inputs

#: The 30 statistical features of one time step, in the column order of the official feature tables
#: (``names`` in notebooks/Compute_results_*_Round2.ipynb). ``cat_*`` columns are categorical: the
#: day-of-year encoding, the Saffir-Simpson category value, and one-hot basin (``AN`` is IBTrACS' North
#: Atlantic ``NA``) and storm nature. ``STORM_DISPLACEMENT_X`` / ``_Y`` are the latitude / longitude
#: change (degrees) since the previous 3-hour step.
HURRICAST_STAT_FEATURES: Tuple[str, ...] = (
    "LAT", "LON", "WMO_WIND", "WMO_PRES", "DIST2LAND",
    "STORM_SPEED", "cat_cos_day", "cat_sign_day", "COS_STORM_DIR", "SIN_STORM_DIR",
    "COS_LAT", "SIN_LAT", "COS_LON", "SIN_LON", "cat_storm_category", "cat_basin_AN",
    "cat_basin_EP", "cat_basin_NI", "cat_basin_SA",
    "cat_basin_SI", "cat_basin_SP", "cat_basin_WP", "cat_nature_DS", "cat_nature_ET",
    "cat_nature_MX", "cat_nature_NR", "cat_nature_SS", "cat_nature_TS",
    "STORM_DISPLACEMENT_X", "STORM_DISPLACEMENT_Y",
)
#: The encoder-decoder reads the first 14 statistical features of every step (``x_stat[:, :, :14]``).
HURRICAST_NETWORK_STAT_FEATURES = 14
#: Reanalysis channels of one step: u, v, z (official variable order) at 225, 500, 700 hPa.
HURRICAST_MAP_CHANNELS: Tuple[str, ...] = tuple(f"{var}{level}" for var in ("u", "v", "z") for level in (225, 500, 700))
HURRICAST_WINDOW = 8  # 8 steps of 3 hours: t-21 h ... t
HURRICAST_GRID = 25  # 25 x 25 one-degree pixels

# ---------------------------------------------------------------------------------------------------
# Configuration presets of the released code (scripts/config.py). Values only; the networks below are
# written from the paper. ``transformer_config`` is the paper's Appendix A1c decoder.

_CNN_LAYOUT = (
    ("conv", 64),
    ("conv", 64),
    ("maxpool", None),
    ("conv", 256),
    ("maxpool", None),
    ("flatten", 256 * 4 * 4),
    ("linear", 576),
    ("linear", 256),
    ("fc", 128),
)

ENCODER_CONFIGS: Dict[str, Dict[str, Any]] = {
    # One CNN over the 9 maps of a step.
    "full_encoder_config": {"n_in": 9, "n_out": 128, "hidden_configuration": _CNN_LAYOUT},
    # Three CNNs (u maps, v maps, z maps) and a linear layer that recombines their outputs.
    "split_encoder_config": {"n_in": 3, "n_out": 128, "hidden_configuration": _CNN_LAYOUT},
}

DECODER_CONFIGS: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "transformer_config": ("transformer", {"n_in": 142, "n_head": 2, "dim_feedforward": 128, "num_layers": 2, "dropout": 0.0, "window_size": None, "n_out_unroll": None, "max_len_pe": 10, "pool_method": "default", "activation": "tanh"}),
    "transformer_config_4": ("transformer", {"n_in": 142, "n_head": 2, "dim_feedforward": 128, "num_layers": 4, "dropout": 0.0, "window_size": None, "n_out_unroll": None, "max_len_pe": 10, "pool_method": "default", "activation": "tanh"}),
    "transformer_config_68": ("transformer", {"n_in": 142, "n_head": 1, "dim_feedforward": 64, "num_layers": 1, "dropout": 0.0, "window_size": None, "n_out_unroll": None, "max_len_pe": 10, "pool_method": "default", "activation": "tanh"}),
    "transformer_config_noviz": ("transformer", {"n_in": 10, "n_head": 2, "dim_feedforward": 512, "num_layers": 6, "dropout": 0.2, "window_size": None, "n_out_unroll": None, "max_len_pe": 10, "pool_method": "default", "activation": "tanh"}),
    "lstm_config": ("recurrent", {"n_in": 142, "hidden_dim": 128, "rnn_num_layers": 2, "N_OUT": 128, "rnn_type": "lstm", "dropout": 0.1, "activation_fn": "tanh", "bidir": True}),
    "lstm_config_4layers": ("recurrent", {"n_in": 142, "hidden_dim": 128, "rnn_num_layers": 4, "N_OUT": 128, "rnn_type": "gru", "dropout": 0.1, "activation_fn": "tanh", "bidir": True}),
    "lstm_config_best_dis": ("recurrent", {"n_in": 142, "hidden_dim": 128, "rnn_num_layers": 4, "N_OUT": 128, "rnn_type": "gru", "dropout": 0.1, "activation_fn": "tanh", "bidir": True}),
    "lstm_config_test_dis": ("recurrent", {"n_in": 142, "hidden_dim": 128, "rnn_num_layers": 2, "N_OUT": 128, "rnn_type": "rnn", "dropout": 0.1, "activation_fn": "tanh", "bidir": True}),
}

#: XGBoost hyperparameters: the defaults of ``train_xgb_track`` / ``train_xgb_intensity`` in the official
#: scripts/run_embeddings.py. The paper gives only ranges (depth 6-9, 100-300 trees, learning rate
#: 0.03-0.15, subsample 0.6-0.9, column sampling 0.7-1, minimum child weight 1-5); these lie inside them.
HURRICAST_XGBOOST_PARAMS: Dict[str, Any] = {
    "max_depth": 8,
    "n_estimators": 140,
    "learning_rate": 0.15,
    "subsample": 0.7,
    "min_child_weight": 5,
}

#: Training settings of the encoder-decoder from the paper (Appendix A3c): Adam, batch 64, L2 weight 0.01,
#: learning rate 1e-3 for intensity and 4e-4 for track; best validation after about 30 epochs.
HURRICAST_TRAINING = {
    "batch_size": 64,
    "l2_reg": 0.01,
    "learning_rate": {"intensity": 1e-3, "displacement": 4e-4},
    "epochs": 30,
}

TARGETS = {"intensity": 1, "displacement": 2}


# ---------------------------------------------------------------------------------------------------
# Network


class HurricastCNNEncoder(nn.Module):
    """CNN that maps the reanalysis maps of one time step ``(batch, n_in, 25, 25)`` to ``(batch, n_out)``.

    ``hidden_configuration`` lists the cells in order: ``("conv", c)`` is a 3 x 3 convolution (stride 1,
    no padding) with ``c`` channels, BatchNorm and ReLU; ``("maxpool", None)`` a 2 x 2 max pooling with
    stride 2; ``("flatten", n)`` flattens to ``n`` features; ``("linear", n)`` a dense layer with
    BatchNorm and ReLU; ``("fc", n)`` a plain dense layer. All modules sit in ``layers``.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        hidden_configuration: Sequence[Tuple[str, Optional[int]]],
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        pool_kernel_size: int = 2,
        pool_stride: int = 2,
        pool_padding: int = 0,
    ):
        super().__init__()
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.hidden_configuration = tuple(tuple(cell) for cell in hidden_configuration)
        layers: List[nn.Module] = []
        width = self.n_in
        for kind, size in self.hidden_configuration:
            if kind == "conv":
                layers += [nn.Conv2d(width, size, kernel_size, stride=stride, padding=padding, groups=groups, bias=True), nn.BatchNorm2d(size), nn.ReLU()]
            elif kind == "maxpool":
                layers.append(nn.MaxPool2d(pool_kernel_size, stride=pool_stride, padding=pool_padding))
            elif kind == "flatten":
                layers.append(nn.Flatten())
            elif kind == "linear":
                layers += [nn.Linear(width, size), nn.BatchNorm1d(size), nn.ReLU()]
            elif kind == "fc":
                layers.append(nn.Linear(width, size))
            else:
                raise ValueError(f"unknown CNN cell {kind!r}; use conv, maxpool, flatten, linear or fc")
            if size is not None:
                width = int(size)
        if width != self.n_out:
            raise ValueError(f"the last cell gives {width} features, expected n_out={self.n_out}")
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _PositionalEncoding(nn.Module):
    """Sinusoidal position encoding, ``P[i, 2j] = sin(i / 10000^(2j/d))``, ``P[i, 2j+1] = cos(...)``, then dropout."""

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


def _activation(name: str):
    if name == "relu":
        return F.relu
    if name == "tanh":
        return torch.tanh
    raise ValueError(f"activation must be 'relu' or 'tanh', got {name!r}")


class HurricastTransformerDecoder(nn.Module):
    """Transformer decoder: positional encoding, ``num_layers`` post-norm encoder layers, pooling, activation.

    Input ``(batch, T, n_in)``; output ``(batch, n_in)`` (``pool_method`` ``"default"`` / ``"mean"``:
    mean over time; ``"unroll"``: a linear layer on the flattened sequence, output ``n_out_unroll``).
    """

    def __init__(
        self,
        n_in: int = 512,
        n_head: int = 4,
        dim_feedforward: int = 2048,
        num_layers: int = 6,
        dropout: float = 0.1,
        window_size: Optional[int] = None,
        n_out_unroll: Optional[int] = None,
        max_len_pe: int = 10,
        pool_method: str = "default",
        activation: str = "tanh",
    ):
        super().__init__()
        self.n_in = int(n_in)
        self.n_head = int(n_head)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.window_size = window_size
        self.n_out_unroll = n_out_unroll
        layer = nn.TransformerEncoderLayer(d_model=self.n_in, nhead=self.n_head, dim_feedforward=self.dim_feedforward, dropout=self.dropout)
        # The nested-tensor fast path only applies to padded batch-first input; it never changes outputs here.
        self.transformer_layers = nn.TransformerEncoder(layer, num_layers=self.num_layers, enable_nested_tensor=False)
        if pool_method not in ("default", "mean", "unroll"):
            raise ValueError("pool_method must be 'default', 'mean' or 'unroll'")
        self.pool_method = pool_method
        if pool_method == "unroll":
            if n_out_unroll is None or window_size is None:
                raise ValueError("pool_method='unroll' needs window_size and n_out_unroll")
            self.pool_linear = nn.Linear(int(window_size) * self.n_in, int(n_out_unroll))
        self.activation = activation
        self.activation_fn = _activation(activation)
        self.max_len_pe = int(max_len_pe)
        self.pe = _PositionalEncoding(self.n_in, self.dropout, max_len=self.max_len_pe)
        self.N_OUT = int(n_out_unroll) if pool_method == "unroll" else self.n_in

    def forward(self, x: torch.Tensor, xgb: bool = False):
        out = self.pe(x)
        out = self.transformer_layers(out.transpose(0, 1)).transpose(0, 1)  # sequence-first inside
        sequence = out
        pooled = self.pool_linear(out.flatten(start_dim=1)) if self.pool_method == "unroll" else out.mean(1)
        pooled = self.activation_fn(pooled)
        return (pooled, sequence) if xgb else pooled


class HurricastRecurrentDecoder(nn.Module):
    """Recurrent decoder of the released code (``ExpLSTM``): a bidirectional LSTM / GRU / RNN over the
    sequence; the last layer's final forward and backward states go through a linear layer and an activation.

    Input ``(batch, T, n_in)``; output ``(batch, N_OUT)``.
    """

    def __init__(
        self,
        n_in: int,
        hidden_dim: int,
        rnn_num_layers: int,
        N_OUT: int,
        rnn_type: str = "gru",
        dropout: float = 0.0,
        activation_fn: str = "tanh",
        bidir: bool = True,
    ):
        super().__init__()
        cells = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}
        if rnn_type not in cells:
            raise ValueError(f"rnn_type must be one of {sorted(cells)}, got {rnn_type!r}")
        if not bidir:
            raise ValueError("the recurrent decoder reads the last forward and backward states; it needs bidir=True")
        self.n_in = int(n_in)
        self.hidden_dim = int(hidden_dim)
        self.rnn_num_layers = int(rnn_num_layers)
        self.N_OUT = int(N_OUT)
        self.rnn_type = rnn_type
        self.bidir = bool(bidir)
        self.dropout = float(dropout)
        self.activation = activation_fn
        self.rnn = cells[rnn_type](
            input_size=self.n_in,
            hidden_size=self.hidden_dim,
            num_layers=self.rnn_num_layers,
            bidirectional=self.bidir,
            batch_first=True,
            dropout=self.dropout,
        )
        self.fc = nn.Linear(self.hidden_dim * (1 + int(self.bidir)), self.N_OUT)
        self.activation_fn = _activation(activation_fn)

    def forward(self, x: torch.Tensor, xgb: bool = False):
        _, hidden = self.rnn(x)
        if self.rnn_type == "lstm":
            hidden = hidden[0]
        out = self.activation_fn(self.fc(torch.cat((hidden[-2], hidden[-1]), dim=1)))
        return (out, hidden) if xgb else out


class HurricastEncoderDecoder(nn.Module):
    """The Hurricast encoder-decoder network (the official ``ExperimentalHurricast``).

    ``forward(x_stat, x_viz)`` takes ``x_stat`` ``(batch, T, n_stat)`` standardised statistical features
    (14 for the paper's configuration) and ``x_viz`` ``(batch, T, 9, 25, 25)`` standardised reanalysis
    maps and returns ``(batch, n_pred)``, the standardised 24-hour intensity (``n_pred = 1``) or latitude /
    longitude displacement (``n_pred = 2``). Without an encoder (``encoder_config=None``) or with
    ``x_viz=None`` the decoder reads ``x_stat`` alone; with ``no_stat=True`` it reads the map embeddings
    alone. :meth:`get_embeddings` returns the decoder output, the input of the last linear layer.
    """

    def __init__(
        self,
        n_pred: int,
        decoder_config: Mapping[str, Any],
        encoder_config: Optional[Mapping[str, Any]] = None,
        decoder_name: str = "transformer",
        split_cnns: bool = False,
        no_stat: bool = False,
    ):
        super().__init__()
        self.n_pred = int(n_pred)
        self.no_stat = bool(no_stat)
        self.split_cnns = bool(split_cnns)
        self.encoder_config = dict(encoder_config) if encoder_config is not None else None
        self.decoder_name = decoder_name
        self.decoder_config = dict(decoder_config)
        # Creation order of the official network: encoder(s), recombination layer, decoder, predictor.
        encoder: Optional[nn.Module] = None
        recombine: Optional[nn.Linear] = None
        if self.encoder_config is not None:
            encoder = HurricastCNNEncoder(**self.encoder_config)
            if self.split_cnns:
                # Three copies of one initialised CNN (identical initial weights, as in the official code).
                encoder = nn.ModuleList([copy.deepcopy(encoder) for _ in range(3)])
                recombine = nn.Linear(3 * self.encoder_config["n_out"], self.encoder_config["n_out"])
        self.encoder = encoder
        self.recombine_encoders = recombine
        if decoder_name == "transformer":
            self.decoder = HurricastTransformerDecoder(**self.decoder_config)
        elif decoder_name == "recurrent":
            self.decoder = HurricastRecurrentDecoder(**self.decoder_config)
        else:
            raise ValueError("decoder_name must be 'transformer' or 'recurrent'")
        self.predictor = nn.Linear(self.decoder.N_OUT, self.n_pred)

    @property
    def embedding_dim(self) -> int:
        return int(self.decoder.N_OUT)

    @property
    def n_stat(self) -> int:
        """Statistical features per step that the decoder expects (0 without statistics)."""
        if self.no_stat and self.encoder is not None:
            return 0
        width = self.decoder.n_in
        if self.encoder is not None:
            width -= self.encoder_config["n_out"]
        return int(width)

    def _encode_step(self, maps: torch.Tensor) -> torch.Tensor:
        if not self.split_cnns:
            return self.encoder(maps).unsqueeze(1)
        parts = torch.split(maps, 3, dim=1)  # (u maps, v maps, z maps)
        out = torch.cat([cnn(part) for part, cnn in zip(parts, self.encoder)], -1)
        return self.recombine_encoders(out).unsqueeze(1)

    def encode(self, x_viz: torch.Tensor) -> torch.Tensor:
        """``(batch, T, channels, 25, 25)`` -> ``(batch, T, 128)``; the CNN runs once per time step."""
        return torch.cat([self._encode_step(step) for step in x_viz.unbind(1)], dim=1)

    def _check(self, x_stat: Optional[torch.Tensor], x_viz: Optional[torch.Tensor]) -> None:
        if x_viz is not None and self.encoder is not None:
            n_in = self.encoder_config["n_in"] * (3 if self.split_cnns else 1)
            if x_viz.ndim != 5 or x_viz.size(2) != n_in:
                raise ValueError(f"x_viz must be shaped (batch, T, {n_in}, 25, 25), got shape {tuple(x_viz.shape)}")
        uses_stat = x_viz is None or self.encoder is None or not (self.no_stat or x_stat is None)
        if uses_stat:
            if x_stat is None or x_stat.ndim != 3:
                raise ValueError(f"x_stat must be shaped (batch, T, features), got {None if x_stat is None else tuple(x_stat.shape)}")
            expected = self.decoder.n_in if (x_viz is None or self.encoder is None) else self.n_stat
            if x_stat.size(-1) != expected:
                raise ValueError(f"x_stat must have {expected} features per step, got shape {tuple(x_stat.shape)}")

    def fuse(self, x_stat: Optional[torch.Tensor], x_viz: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The decoder input: statistics, map embeddings, or ``[statistics, embeddings]`` per step."""
        self._check(x_stat, x_viz)
        if x_viz is None or self.encoder is None:
            return x_stat
        if self.no_stat or x_stat is None:
            return self.encode(x_viz)
        return torch.cat([x_stat, self.encode(x_viz)], -1)

    def get_embeddings(self, x_stat: Optional[torch.Tensor], x_viz: Optional[torch.Tensor] = None, xgb: bool = False):
        """Decoder output ``(batch, embedding_dim)``; with ``xgb=True`` also the decoder's sequence output
        (Transformer) or final hidden states (recurrent), as the official ``get_embeddings(..., xgb=True)``."""
        return self.decoder(self.fuse(x_stat, x_viz), xgb=xgb)

    def forward(self, x_stat: Optional[torch.Tensor], x_viz: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.predictor(self.get_embeddings(x_stat, x_viz))


def hurricast_l2_penalty(module: nn.Module) -> torch.Tensor:
    """Sum of squares of every parameter whose name contains ``weight`` (the paper's L2 term)."""
    terms = [(param**2).sum() for name, param in module.named_parameters() if "weight" in name]
    return torch.stack(terms).sum() if terms else torch.zeros(())


def hurricast_training_loss(output: torch.Tensor, target: torch.Tensor, network: nn.Module, l2_reg: float) -> torch.Tensor:
    """MSE plus ``2 / batch * l2_reg * sum(W**2)``, the scaling of the official training loop.

    ``output`` and ``target`` must have the same shape (the official loop compared ``(batch, 1)`` intensity
    outputs with ``(batch,)`` targets, which broadcasts to ``(batch, batch)``; see the model card).
    """
    if output.shape != target.shape:
        raise ValueError(f"output shape {tuple(output.shape)} and target shape {tuple(target.shape)} differ")
    loss = F.mse_loss(output, target)
    if l2_reg > 0:
        loss = loss + 2.0 / target.size(0) * l2_reg * hurricast_l2_penalty(network)
    return loss


# ---------------------------------------------------------------------------------------------------
# XGBoost stage


def hurricast_xgboost_columns(window_size: int = HURRICAST_WINDOW, features: Sequence[str] = HURRICAST_STAT_FEATURES) -> List[str]:
    """Names of the statistical columns XGBoost reads, in order.

    The official feature table holds the ``features`` of every step, step after step, with a ``_<step>``
    suffix; it keeps a column when its name ends in ``_0`` (the oldest step, t-21 h) or does not start
    with ``cat``: all numerical features of every step and the categorical ones of the first step only.
    """
    names = [f"{name}_{index // len(features)}" for index, name in enumerate(list(features) * int(window_size))]
    return [name for name in names if name.lower()[-2:] == "_0" or name.lower()[:3] != "cat"]


def hurricast_xgboost_features(
    x_stat: Union[torch.Tensor, np.ndarray],
    embeddings: Optional[Union[torch.Tensor, np.ndarray]] = None,
    features: Sequence[str] = HURRICAST_STAT_FEATURES,
) -> np.ndarray:
    """XGBoost input: the selected statistical columns of ``x_stat`` ``(batch, T, 30)``, then the embeddings."""
    stat = x_stat.detach().cpu().numpy() if isinstance(x_stat, torch.Tensor) else np.asarray(x_stat)
    if stat.ndim != 3 or stat.shape[-1] != len(features):
        raise ValueError(f"x_stat must be shaped (batch, T, {len(features)}) for the XGBoost stage, got shape {stat.shape}")
    window = stat.shape[1]
    all_names = [f"{name}_{index // len(features)}" for index, name in enumerate(list(features) * window)]
    keep = set(hurricast_xgboost_columns(window, features))
    columns = [i for i, name in enumerate(all_names) if name in keep]
    flat = stat.reshape(stat.shape[0], -1)[:, columns]
    if embeddings is None:
        return flat
    emb = embeddings.detach().cpu().numpy() if isinstance(embeddings, torch.Tensor) else np.asarray(embeddings)
    return np.concatenate([flat, emb.reshape(emb.shape[0], -1)], axis=1)


def _xgboost_estimator(params: Mapping[str, Any], n_targets: int):
    from .classical import _import_xgboost

    xgboost = _import_xgboost()
    estimator = xgboost.XGBRegressor(**params)
    if n_targets == 1:
        return estimator
    from sklearn.multioutput import MultiOutputRegressor

    # One regressor per target (latitude and longitude displacement), as the official xgb_x / xgb_y.
    return MultiOutputRegressor(estimator)


# ---------------------------------------------------------------------------------------------------
# Two-stage model


def _as_tensor(value: Any) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else torch.as_tensor(np.asarray(value), dtype=torch.float32)


class Hurricast(nn.Module):
    """Hurricast forecaster: encoder-decoder embeddings plus XGBoost (or the network's own head).

    Input: a mapping with ``x_stat`` ``(batch, T, 30)`` (the statistical features of
    :data:`HURRICAST_STAT_FEATURES`; the network reads the first 14, which must be standardised as for
    training), ``x_viz`` ``(batch, T, 9, 25, 25)`` standardised ERA5 maps (not needed for
    ``use_embeddings=False``), and for track forecasts ``position`` ``(batch, 2)``, the latitude and
    longitude at the forecast time in degrees.

    ``forward`` returns ``(batch, 1)`` intensity 24 hours ahead (in the training targets' unit, knots for
    IBTrACS) or ``(batch, 2)`` latitude / longitude displacement over 24 hours (degrees).
    ``forecast`` turns them into benchmark forecasts: ``{"intensity": ...}`` or ``{"lat", "lon"}``.

    Train with :meth:`fit` (encoder-decoder by gradient descent, then XGBoost on its embeddings);
    :class:`pyhazards.engine.Trainer` refuses ``fit``. The XGBoost stage is not part of ``state_dict()``;
    use :meth:`save` / :meth:`load`. The network's parameters are ``network.*`` with the official names.
    """

    def __init__(
        self,
        target: str = "intensity",
        predictor: str = "xgboost",
        network: Optional[HurricastEncoderDecoder] = None,
        use_embeddings: bool = True,
        xgboost_params: Optional[Mapping[str, Any]] = None,
        estimator: Any = None,
        n_stat_features: int = len(HURRICAST_STAT_FEATURES),
        config: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        self.config = dict(config or {})  # builder arguments, used by load()
        if target not in TARGETS:
            raise ValueError(f"target must be one of {sorted(TARGETS)}, got {target!r}")
        if predictor not in ("xgboost", "network"):
            raise ValueError("predictor must be 'xgboost' or 'network'")
        if predictor == "network" and network is None:
            raise ValueError("predictor='network' needs the encoder-decoder network")
        if predictor == "xgboost" and use_embeddings and network is None:
            raise ValueError("use_embeddings=True needs the encoder-decoder network")
        self.target = target
        self.n_targets = TARGETS[target]
        self.predictor = predictor
        self.use_embeddings = bool(use_embeddings) and network is not None
        self.n_stat_features = int(n_stat_features)
        self.network = network
        if network is not None and network.n_pred != self.n_targets:
            raise ValueError(f"the network predicts {network.n_pred} values, target {target!r} needs {self.n_targets}")
        self.xgboost_params = dict(HURRICAST_XGBOOST_PARAMS if xgboost_params is None else xgboost_params)
        self.xgboost = None
        if predictor == "xgboost":
            from .classical import EstimatorModule

            model = estimator if estimator is not None else _xgboost_estimator(self.xgboost_params, self.n_targets)
            self.xgboost = EstimatorModule(model, task="regression", name="hurricast_xgboost")
        # Training-target standardisation (the official pipeline fits both stages on standardised targets).
        self.register_buffer("target_mean", torch.zeros(self.n_targets))
        self.register_buffer("target_std", torch.ones(self.n_targets))

    # -- inputs --------------------------------------------------------------------------------------
    def _unpack(self, inputs: Any, x_viz: Optional[torch.Tensor] = None):
        if isinstance(inputs, Mapping):
            x_stat, x_viz, position = inputs.get("x_stat"), inputs.get("x_viz"), inputs.get("position")
        else:
            x_stat, position = inputs, None
        x_stat = None if x_stat is None else _as_tensor(x_stat).float()
        x_viz = None if x_viz is None else _as_tensor(x_viz).float()
        if x_stat is None or x_stat.ndim != 3:
            raise ValueError(f"Hurricast needs x_stat shaped (batch, T, features), got {None if x_stat is None else tuple(x_stat.shape)}")
        if self.predictor == "xgboost" and x_stat.size(-1) != self.n_stat_features:
            raise ValueError(f"the XGBoost stage needs x_stat shaped (batch, T, {self.n_stat_features}), got shape {tuple(x_stat.shape)}")
        if self.network is not None and self.network.encoder is not None and (self.predictor == "network" or self.use_embeddings) and x_viz is None:
            raise ValueError("Hurricast with a map encoder needs x_viz shaped (batch, T, 9, 25, 25)")
        return x_stat, x_viz, position

    def _network_stat(self, x_stat: torch.Tensor, x_viz: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        network = self.network
        uses_maps = x_viz is not None and network.encoder is not None
        width = network.n_stat if uses_maps else network.decoder.n_in
        if width == 0:
            return None
        if x_stat.size(-1) < width:
            raise ValueError(f"the network reads {width} statistical features per step, x_stat has shape {tuple(x_stat.shape)}")
        return x_stat[..., :width]

    def embeddings(self, inputs: Any) -> torch.Tensor:
        """Reanalysis embeddings ``(batch, embedding_dim)`` of the frozen encoder-decoder (always in eval mode)."""
        x_stat, x_viz, _ = self._unpack(inputs)
        network, was_training = self.network, self.network.training
        network.eval()
        try:
            return network.get_embeddings(self._network_stat(x_stat, x_viz), x_viz)
        finally:
            network.train(was_training)

    def xgboost_features(self, inputs: Any) -> np.ndarray:
        """The vectors XGBoost reads: flattened statistics, then the embeddings (``use_embeddings``)."""
        x_stat, _, _ = self._unpack(inputs)
        embeddings = None
        if self.use_embeddings:
            with torch.no_grad():
                embeddings = self.embeddings(inputs)
        return hurricast_xgboost_features(x_stat, embeddings)

    # -- targets -------------------------------------------------------------------------------------
    def model_targets(self, inputs: Any, targets: Any) -> torch.Tensor:
        """Targets as the model predicts them: intensity ``(N, 1)`` or displacement ``(N, 2)``.

        Track targets may be given as displacement ``(N, 2)`` or as the observed position 24 hours
        ahead ``(N, 1, 2)`` (latitude, longitude, the ``tc.track_intensity`` layout), which needs
        ``position`` in ``inputs``.
        """
        y = _as_tensor(targets).float()
        if self.target == "intensity":
            return y.reshape(-1, 1)
        if y.ndim == 3:
            position = inputs.get("position") if isinstance(inputs, Mapping) else None
            if position is None or y.shape[1:] != (1, 2):
                raise ValueError("track targets (N, 1, 2) need inputs['position'] (N, 2); or pass displacements (N, 2)")
            return y[:, 0, :] - _as_tensor(position).float()
        if y.ndim != 2 or y.size(1) != 2:
            raise ValueError(f"displacement targets must be shaped (N, 2), got shape {tuple(y.shape)}")
        return y

    # -- prediction ----------------------------------------------------------------------------------
    def forward(self, inputs: Any) -> torch.Tensor:
        x_stat, x_viz, _ = self._unpack(inputs)
        if self.predictor == "network":
            scaled = self.network(self._network_stat(x_stat, x_viz), x_viz)
        else:
            scaled = self.xgboost(hurricast_xgboost_features(x_stat, self.embeddings(inputs).detach() if self.use_embeddings else None))
            scaled = scaled.to(self.target_mean.device).reshape(-1, self.n_targets)
        return scaled * self.target_std + self.target_mean

    def forecast(self, inputs: Any) -> Dict[str, torch.Tensor]:
        """Benchmark forecasts: ``{"intensity": (batch, 1)}`` or ``{"lat", "lon"}`` ``(batch, 1)`` 24 h ahead."""
        out = self.forward(inputs)
        if self.target == "intensity":
            return {"intensity": out}
        position = inputs.get("position") if isinstance(inputs, Mapping) else None
        if position is None:
            raise ValueError("track forecasts need inputs['position'] (batch, 2): latitude and longitude at the forecast time")
        position = _as_tensor(position).to(out)
        return {"lat": position[:, :1] + out[:, :1], "lon": position[:, 1:2] + out[:, 1:2]}

    # -- training ------------------------------------------------------------------------------------
    @property
    def custom_fit_reason(self) -> str:
        return (
            "hurricast trains in two stages (the encoder-decoder by gradient descent on standardised "
            "targets, then XGBoost on its frozen embeddings). Use model.fit(inputs, targets, val_inputs, "
            "val_targets); Trainer.evaluate and Trainer.predict work on the fitted model."
        )

    def fit(
        self,
        inputs: Mapping[str, Any],
        targets: Any,
        val_inputs: Optional[Mapping[str, Any]] = None,
        val_targets: Any = None,
        train_network: bool = True,
        epochs: int = HURRICAST_TRAINING["epochs"],
        batch_size: int = HURRICAST_TRAINING["batch_size"],
        learning_rate: Optional[float] = None,
        l2_reg: float = HURRICAST_TRAINING["l2_reg"],
        seed: int = 0,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "Hurricast":
        """Fit the target scaling, the encoder-decoder (``train_network``) and the XGBoost stage.

        Targets are standardised with their training mean and standard deviation. The network is trained
        with Adam (learning rate 1e-3 for intensity, 4e-4 for displacement unless ``learning_rate`` is
        given), shuffled mini-batches of ``batch_size`` (the last incomplete batch dropped), MSE plus the
        L2 term of :func:`hurricast_training_loss`; with validation data the weights of the epoch with the
        lowest validation MSE are kept. XGBoost is then fitted on :meth:`xgboost_features`.
        """
        y = self.model_targets(inputs, targets)
        mean, std = y.mean(0), y.std(0)
        std = torch.where(std > 0, std, torch.ones_like(std))
        self.target_mean.copy_(mean)
        self.target_std.copy_(std)
        scaled = (y - mean) / std
        if train_network and self.network is not None:
            val = None
            if val_inputs is not None and val_targets is not None:
                val = (val_inputs, (self.model_targets(val_inputs, val_targets) - mean) / std)
            lr = learning_rate if learning_rate is not None else HURRICAST_TRAINING["learning_rate"][self.target]
            self.fit_network(inputs, scaled, val=val, epochs=epochs, batch_size=batch_size, learning_rate=lr, l2_reg=l2_reg, seed=seed, device=device)
        if self.xgboost is not None:
            if self.network is not None:
                self.network.eval()
            features = self.xgboost_features(inputs)
            labels = scaled.numpy() if self.n_targets > 1 else scaled[:, 0].numpy()
            self.xgboost.fit(features, labels)
        return self

    def fit_network(
        self,
        inputs: Mapping[str, Any],
        scaled_targets: torch.Tensor,
        val: Optional[Tuple[Mapping[str, Any], torch.Tensor]] = None,
        epochs: int = HURRICAST_TRAINING["epochs"],
        batch_size: int = HURRICAST_TRAINING["batch_size"],
        learning_rate: float = 1e-3,
        l2_reg: float = HURRICAST_TRAINING["l2_reg"],
        seed: int = 0,
        device: Optional[Union[str, torch.device]] = None,
    ) -> List[float]:
        """Train the encoder-decoder on standardised targets; returns the validation MSE per epoch."""
        network = self.network
        device = torch.device(device) if device is not None else next(network.parameters()).device
        network.to(device)
        x_stat, x_viz, _ = self._unpack(inputs)
        n = x_stat.size(0)
        if n < batch_size:
            raise ValueError(f"need at least batch_size={batch_size} training samples, got {n}")
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        generator = torch.Generator().manual_seed(int(seed))
        best, best_state, history = float("inf"), None, []
        for _ in range(int(epochs)):
            network.train()
            order = torch.randperm(n, generator=generator)
            for start in range(0, n - batch_size + 1, batch_size):
                index = order[start : start + batch_size]
                stat = self._network_stat(x_stat[index], x_viz)
                maps = None if x_viz is None else x_viz[index].to(device)
                out = network(None if stat is None else stat.to(device), maps)
                loss = hurricast_training_loss(out, scaled_targets[index].to(device), network, l2_reg)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if val is not None:
                network.eval()
                v_stat, v_viz, _ = self._unpack(val[0])
                with torch.no_grad():
                    stat = self._network_stat(v_stat, v_viz)
                    pred = network(None if stat is None else stat.to(device), None if v_viz is None else v_viz.to(device))
                    mse = float(F.mse_loss(pred.cpu(), val[1].reshape(pred.shape)))
                history.append(mse)
                if mse < best:
                    best, best_state = mse, copy.deepcopy(network.state_dict())
        if best_state is not None:
            network.load_state_dict(best_state)
        network.eval()
        return history

    # -- persistence ---------------------------------------------------------------------------------
    def save(self, path) -> None:
        """Save the network weights, target scaling and fitted XGBoost stage (joblib; trusted files only)."""
        import joblib

        joblib.dump(
            {
                "format": "pyhazards.Hurricast",
                "config": self.config,
                "state_dict": {k: v.cpu() for k, v in self.state_dict().items()},
                "estimator": None if self.xgboost is None else self.xgboost.estimator,
            },
            path,
        )

    @classmethod
    def load(cls, path) -> "Hurricast":
        """Load a model written by :meth:`save` (unpickles the file: only load files you trust)."""
        import joblib

        payload = joblib.load(path)
        if not isinstance(payload, dict) or payload.get("format") != "pyhazards.Hurricast":
            raise ValueError(f"{path} was not written by Hurricast.save")
        model = hurricast_builder("regression", estimator=payload["estimator"], **payload["config"])
        model.load_state_dict(payload["state_dict"])
        return model.eval()


def hurricast_network(
    target: str = "intensity",
    decoder_config: str = "transformer_config",
    encoder_config: Optional[str] = "full_encoder_config",
    no_stat: bool = False,
) -> HurricastEncoderDecoder:
    """The official encoder-decoder for a preset of the released code (``scripts/config.py``)."""
    if target not in TARGETS:
        raise ValueError(f"target must be one of {sorted(TARGETS)}, got {target!r}")
    if decoder_config not in DECODER_CONFIGS:
        raise ValueError(f"unknown decoder_config {decoder_config!r}; choose from {sorted(DECODER_CONFIGS)}")
    if encoder_config is not None and encoder_config not in ENCODER_CONFIGS:
        raise ValueError(f"unknown encoder_config {encoder_config!r}; choose from {sorted(ENCODER_CONFIGS)} or None")
    decoder_name, decoder_kwargs = DECODER_CONFIGS[decoder_config]
    return HurricastEncoderDecoder(
        n_pred=TARGETS[target],
        decoder_config=decoder_kwargs,
        encoder_config=None if encoder_config is None else ENCODER_CONFIGS[encoder_config],
        decoder_name=decoder_name,
        split_cnns=encoder_config == "split_encoder_config",
        no_stat=no_stat,
    )


def hurricast_builder(
    task: str,
    target: str = "intensity",
    predictor: str = "xgboost",
    decoder_config: str = "transformer_config",
    encoder_config: Optional[str] = "full_encoder_config",
    no_stat: bool = False,
    use_embeddings: bool = True,
    xgboost_params: Optional[Mapping[str, Any]] = None,
    estimator: Any = None,
    **kwargs: Any,
) -> nn.Module:
    """Hurricast for ``target="intensity"`` or ``"displacement"`` (24-hour track).

    Defaults: HUML-(stat/viz, xgb/cnn/transfo), the paper's best model (CNN encoder, Transformer decoder,
    XGBoost with the official defaults). ``predictor="network"`` predicts with the network's linear head;
    ``use_embeddings=False`` is HUML-(stat, xgb) (no network). Other keyword arguments (``n_jobs``, ...)
    go to ``xgboost.XGBRegressor``.
    """
    if task.lower() != "regression":
        raise ValueError("hurricast is a regression model (24-hour intensity or displacement); use task='regression'.")
    kwargs.pop("name", None)
    params = dict(HURRICAST_XGBOOST_PARAMS if xgboost_params is None else xgboost_params)
    params.update(kwargs)
    needs_network = predictor == "network" or use_embeddings
    network = hurricast_network(target, decoder_config, encoder_config, no_stat) if needs_network else None
    config = {
        "target": target,
        "predictor": predictor,
        "decoder_config": decoder_config,
        "encoder_config": encoder_config,
        "no_stat": no_stat,
        "use_embeddings": use_embeddings,
        "xgboost_params": params,
    }
    return Hurricast(
        target=target,
        predictor=predictor,
        network=network,
        use_embeddings=use_embeddings,
        xgboost_params=params,
        estimator=estimator,
        config=config,
    )


__all__ = [
    "DECODER_CONFIGS",
    "ENCODER_CONFIGS",
    "HURRICAST_MAP_CHANNELS",
    "HURRICAST_NETWORK_STAT_FEATURES",
    "HURRICAST_STAT_FEATURES",
    "HURRICAST_TRAINING",
    "HURRICAST_XGBOOST_PARAMS",
    "Hurricast",
    "HurricastCNNEncoder",
    "HurricastEncoderDecoder",
    "HurricastRecurrentDecoder",
    "HurricastTransformerDecoder",
    "hurricast_builder",
    "hurricast_l2_penalty",
    "hurricast_network",
    "hurricast_training_loss",
    "hurricast_xgboost_columns",
    "hurricast_xgboost_features",
]
