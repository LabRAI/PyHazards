"""EQTransformer: simultaneous earthquake detection and P/S phase picking.

Mousavi, Ellsworth, Zhu, Chuang & Beroza, "Earthquake transformer—an attentive deep-learning model
for simultaneous earthquake detection and phase picking", Nature Communications 11, 3952 (2020),
https://doi.org/10.1038/s41467-020-17591-w.

Port of the official Keras code, smousavi05/EQTransformer at commit
``a589da6504b11df8341fe1dc859a025f49955409`` (MIT License, Copyright (c) 2020-2025 S. Mostafa
Mousavi): the network built by ``cred2`` and its custom layers ``SeqSelfAttention``, ``FeedForward``
and ``LayerNormalization`` in ``EQTransformer/core/EqT_utils.py``, and the pick post-processing of the
official tester (``picker`` in the same file). The three custom layers are, as their docstrings say,
modified from CyberZHG's keras-self-attention (MIT License, Copyright (c) 2018 PoW),
keras-position-wise-feed-forward and keras-layer-normalization (MIT License, Copyright (c) 2018 Zhao HG).

The architecture follows the graphs stored in the two released models, which differ from the
defaults of ``cred2`` at the pinned commit:

- ``variant="original"``: ``ModelsAndSampleData/EqT_original_model.h5`` ("the one in the paper"):
  two BiLSTM blocks, dropout rate 0.2 everywhere, 371,639 trainable parameters;
- ``variant="conservative"``: ``ModelsAndSampleData/EqT_model_conservative.h5`` (fewer false
  positives, saved by Keras 2.2.4): three BiLSTM blocks, dropout rate 0.1, LSTMs with the Keras 2.2
  defaults ``recurrent_activation="hard_sigmoid"`` and ``implementation=1``, 376,423 trainable
  parameters.

Encoder: seven Conv1D ('same', ReLU) + MaxPooling1D(2, 'same') layers with 8, 16, 16, 32, 32, 64, 64
filters and kernels 11, 9, 7, 7, 5, 5, 3 (6000 -> 47 samples); seven residual CNN blocks (BN, ReLU,
SpatialDropout1D, Conv1D, twice, plus the input; kernels 3, 3, 3, 3, 2, 3, 2); BiLSTM blocks
(Bidirectional LSTM(16), 1x1 Conv1D(16), BN); two transformer blocks (additive self-attention, add,
layer norm, feed-forward(128), add, layer norm). Decoders: the detection decoder reads the encoder
output; the P and S decoders first apply an LSTM(16) and a local self-attention of width 3. Each
decoder is seven UpSampling1D(2) + Conv1D ('same', ReLU) steps (64, 64, 32, 32, 16, 16, 8 filters,
kernels 3, 5, 5, 7, 7, 9, 11; Cropping1D(1, 1) before the fourth conv), followed by Conv1D(1, 11)
with a sigmoid.

Changes made for PyHazards (outputs are unchanged):

- PyTorch layout ``(batch, channels, samples)``: input ``(batch, 3, 6000)`` with components in the
  order E, N, Z of STEAD (the training data), output ``(batch, 3, 6000)`` with the detection, P and S
  probabilities stacked along dim 1 (the Keras model returns three ``(batch, 6000, 1)`` tensors).
- The Keras LSTM is reimplemented (one bias vector, gates i, f, c, o, sigmoid recurrent activation,
  input and recurrent dropout masks fixed over time), so parameter counts equal the Keras ones.
- SpatialDropout1D is called with ``training=True`` in the released graphs, i.e. it is active at
  inference as well (Monte Carlo dropout). Here it is active in train mode, and in eval mode only with
  ``mc_dropout=True``; the default eval output is deterministic.
- BatchNorm keeps the running-statistics update of the Keras version that trained each model
  (:class:`KerasBatchNorm1d`); normalisation itself is unchanged.
- :meth:`EQTransformer.extract_picks` / :meth:`EQTransformer.extract_detections` port ``picker``;
  :func:`load_eqtransformer_keras_weights` reads the released ``.h5`` files with h5py.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._pretrained import cached_download
from ._seismic import KerasBatchNorm1d, TracePicks, WaveformPicker, check_waveforms, std_normalize

KERAS_EPSILON = 1e-7  # keras.backend.epsilon()
LAYER_NORM_EPSILON = KERAS_EPSILON * KERAS_EPSILON  # LayerNormalization default (9.999999999999998e-15)
TRUNCATED_NORMAL_STD = 0.87962566103423978  # std of a unit normal truncated at +-2

ENCODER_FILTERS = (8, 16, 16, 32, 32, 64, 64)
ENCODER_KERNELS = (11, 9, 7, 7, 5, 5, 3)
RES_CNN_KERNELS = (3, 3, 3, 3, 2, 3, 2)  # cred2: 5 blocks of kernel 3, blocks 4 and 5 followed by kernel 2
DECODER_FILTERS = tuple(reversed(ENCODER_FILTERS))
DECODER_KERNELS = tuple(reversed(ENCODER_KERNELS))
DECODER_CROP_STEP = 3
LSTM_UNITS = 16
ATTENTION_UNITS = 32
FEED_FORWARD_UNITS = 128
IN_SAMPLES = 6000

EQTRANSFORMER_VARIANTS: Dict[str, Dict[str, Any]] = {
    # EqT_original_model.h5 (Keras 2.3.0): the paper model.
    "original": {
        "lstm_blocks": 2,
        "drop_rate": 0.2,
        "recurrent_activation": "sigmoid",
        "lstm_implementation": 2,
        "bn_zero_debias": False,  # Keras 2.3.0 moving averages
        "loss_weights": (0.05, 0.40, 0.55),
    },
    # EqT_model_conservative.h5 (Keras 2.2.4).
    "conservative": {
        "lstm_blocks": 3,
        "drop_rate": 0.1,
        "recurrent_activation": "hard_sigmoid",
        "lstm_implementation": 1,
        "bn_zero_debias": True,  # Keras 2.2.4 moving averages
        "loss_weights": (0.02, 0.40, 0.58),
    },
}

_WEIGHTS_COMMIT = "a589da6504b11df8341fe1dc859a025f49955409"
EQTRANSFORMER_WEIGHTS: Dict[str, Dict[str, Any]] = {
    "original": {
        "filename": "EqT_original_model.h5",
        "urls": (
            f"https://raw.githubusercontent.com/smousavi05/EQTransformer/{_WEIGHTS_COMMIT}/ModelsAndSampleData/EqT_original_model.h5",
        ),
        "sha256": "07b3ffc8065164ece44a367b145b0db66e25586d89269ec4d8c340c34966d452",
        "license": "MIT",
    },
    "conservative": {
        "filename": "EqT_model_conservative.h5",
        "urls": (
            f"https://raw.githubusercontent.com/smousavi05/EQTransformer/{_WEIGHTS_COMMIT}/ModelsAndSampleData/EqT_model_conservative.h5",
        ),
        "sha256": "0a9179fd40f9cda4fbc5372dd5f660bc35e69620d91dd7bdffe7f65baf7bd99b",
        "license": "MIT",
    },
}


def _glorot_normal_(tensor: torch.Tensor, fan_in: int, fan_out: int) -> torch.Tensor:
    """Keras ``glorot_normal``: truncated normal (+-2 std) with std ``sqrt(2 / (fan_in + fan_out))``."""
    std = math.sqrt(1.0 / ((fan_in + fan_out) / 2.0)) / TRUNCATED_NORMAL_STD
    with torch.no_grad():
        return nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)


class KerasConv1d(nn.Conv1d):
    """Conv1D with stride 1 and Keras 'same' padding (an even kernel pads one sample more on the right).

    Initialised like Keras: ``glorot_uniform`` kernel, zero bias.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__(in_channels, out_channels, kernel_size, padding=0)
        self._pad = ((kernel_size - 1) // 2, kernel_size - 1 - (kernel_size - 1) // 2)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, self._pad))


def _spatial_dropout(x: torch.Tensor, rate: float, active: bool) -> torch.Tensor:
    """Keras SpatialDropout1D on ``(batch, channels, samples)``: whole channels are dropped."""
    if not active or rate <= 0.0:
        return x
    return F.dropout1d(x, p=rate, training=True)


def _max_pool_same(x: torch.Tensor) -> torch.Tensor:
    # MaxPooling1D(2, padding='same'): ceil(L / 2) outputs, the last window may be partial.
    return F.max_pool1d(x, kernel_size=2, stride=2, ceil_mode=True)


def _hard_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Keras 2.x ``hard_sigmoid``: ``clip(0.2 * x + 0.5, 0, 1)``."""
    return torch.clamp(0.2 * x + 0.5, 0.0, 1.0)


class KerasLSTM(nn.Module):
    """Keras ``LSTM(units, return_sequences=True)`` on ``(batch, time, features)``.

    One bias vector, gate order (i, f, c, o), tanh activation and ``recurrent_activation`` "sigmoid"
    (Keras >= 2.3 default, paper model) or "hard_sigmoid" (Keras 2.2 default, conservative model). In
    train mode, ``dropout`` and ``recurrent_dropout`` multiply the inputs and the previous hidden state
    by masks drawn once per sequence: one mask shared by the four gates with ``implementation=2``, one
    per gate with ``implementation=1`` (the stored setting of each released model). Weights keep the
    Keras names and layouts: ``kernel`` ``(features, 4 * units)``, ``recurrent_kernel``
    ``(units, 4 * units)``, ``bias`` ``(4 * units,)``.
    """

    def __init__(
        self,
        input_size: int,
        units: int,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        recurrent_activation: str = "sigmoid",
        implementation: int = 2,
    ):
        super().__init__()
        if recurrent_activation not in ("sigmoid", "hard_sigmoid"):
            raise ValueError(f"Unsupported recurrent_activation {recurrent_activation!r}.")
        if implementation not in (1, 2):
            raise ValueError(f"Keras LSTM implementation must be 1 or 2, got {implementation}.")
        self.units = units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.recurrent_activation = recurrent_activation
        self.implementation = implementation
        self.kernel = nn.Parameter(torch.empty(input_size, 4 * units))
        self.recurrent_kernel = nn.Parameter(torch.empty(units, 4 * units))
        self.bias = nn.Parameter(torch.empty(4 * units))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Keras: glorot_uniform kernel, orthogonal recurrent kernel, zero bias with unit_forget_bias.
        nn.init.xavier_uniform_(self.kernel)
        nn.init.orthogonal_(self.recurrent_kernel)
        with torch.no_grad():
            self.bias.zero_()
            self.bias[self.units : 2 * self.units] = 1.0

    @staticmethod
    def _mask(shape: Tuple[int, ...], rate: float, reference: torch.Tensor) -> torch.Tensor:
        keep = 1.0 - rate
        return torch.bernoulli(torch.full(shape, keep, dtype=reference.dtype, device=reference.device)) / keep

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        batch, steps, features = x.shape
        units = self.units
        gates = 4 if self.implementation == 1 else 1
        activation = torch.sigmoid if self.recurrent_activation == "sigmoid" else _hard_sigmoid
        input_mask = recurrent_mask = None
        if self.training and 0.0 < self.dropout < 1.0:
            input_mask = self._mask((gates, batch, 1, features), self.dropout, x)
        if self.training and 0.0 < self.recurrent_dropout < 1.0:
            recurrent_mask = self._mask((gates, batch, units), self.recurrent_dropout, x)
        if input_mask is None:
            projected = torch.matmul(x, self.kernel) + self.bias  # (batch, time, 4 * units)
        else:
            kernels = self.kernel.view(features, 4, units)
            parts = [
                torch.matmul(x * input_mask[g if gates == 4 else 0], kernels[:, g]) for g in range(4)
            ]
            projected = torch.cat(parts, dim=-1) + self.bias
        recurrent = self.recurrent_kernel.view(units, 4, units)
        h = x.new_zeros(batch, units)
        c = x.new_zeros(batch, units)
        outputs: List[torch.Tensor] = [x.new_empty(0)] * steps
        order = range(steps - 1, -1, -1) if reverse else range(steps)
        for t in order:
            if recurrent_mask is None:
                z = projected[:, t] + torch.matmul(h, self.recurrent_kernel)
            else:
                z = projected[:, t] + torch.cat(
                    [torch.matmul(h * recurrent_mask[g if gates == 4 else 0], recurrent[:, g]) for g in range(4)],
                    dim=-1,
                )
            i = activation(z[:, :units])
            f = activation(z[:, units : 2 * units])
            c = f * c + i * torch.tanh(z[:, 2 * units : 3 * units])
            h = activation(z[:, 3 * units :]) * torch.tanh(c)
            outputs[t] = h
        return torch.stack(outputs, dim=1)


class SeqSelfAttention(nn.Module):
    """Additive self-attention of ``EqT_utils.SeqSelfAttention`` on ``(batch, time, features)``.

    ``e[t, t'] = Wa^T tanh(Wt^T x_t + Wx^T x_t' + bh) + ba``; the weights are ``exp(e - max_t' e)``,
    multiplied by a local window mask when ``attention_width`` is set (``t - w // 2 <= t' <
    t - w // 2 + w``) and divided by their sum plus ``1e-7``. As in the reference, the maximum is taken
    over the whole row before the window mask is applied. Returns ``sum_t' a[t, t'] x_t'``.
    """

    def __init__(self, features: int, units: int = ATTENTION_UNITS, attention_width: Optional[int] = None):
        super().__init__()
        self.attention_width = attention_width
        self.Wt = nn.Parameter(torch.empty(features, units))
        self.Wx = nn.Parameter(torch.empty(features, units))
        self.bh = nn.Parameter(torch.zeros(units))
        self.Wa = nn.Parameter(torch.empty(units, 1))
        self.ba = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        features, units = self.Wt.shape
        _glorot_normal_(self.Wt, features, units)
        _glorot_normal_(self.Wx, features, units)
        _glorot_normal_(self.Wa, units, 1)
        nn.init.zeros_(self.bh)
        nn.init.zeros_(self.ba)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        steps = x.shape[1]
        q = torch.matmul(x, self.Wt).unsqueeze(2)
        k = torch.matmul(x, self.Wx).unsqueeze(1)
        h = torch.tanh(q + k + self.bh)
        e = (torch.matmul(h, self.Wa) + self.ba).squeeze(-1)  # (batch, time, time)
        e = torch.exp(e - e.max(dim=-1, keepdim=True).values)
        if self.attention_width is not None:
            index = torch.arange(steps, device=x.device)
            lower = (index - self.attention_width // 2).unsqueeze(-1)
            upper = lower + self.attention_width
            window = (lower <= index.unsqueeze(0)) & (index.unsqueeze(0) < upper)
            e = e * window.to(e.dtype)
        a = e / (e.sum(dim=-1, keepdim=True) + KERAS_EPSILON)
        v = torch.matmul(a, x)
        return (v, a) if return_attention else v


class FeedForward(nn.Module):
    """Position-wise feed-forward layer of ``EqT_utils.FeedForward`` (ReLU, dropout in train mode)."""

    def __init__(self, features: int, units: int = FEED_FORWARD_UNITS, dropout_rate: float = 0.0):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.W1 = nn.Parameter(torch.empty(features, units))
        self.b1 = nn.Parameter(torch.zeros(units))
        self.W2 = nn.Parameter(torch.empty(units, features))
        self.b2 = nn.Parameter(torch.zeros(features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        features, units = self.W1.shape
        _glorot_normal_(self.W1, features, units)
        _glorot_normal_(self.W2, units, features)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(torch.matmul(x, self.W1) + self.b1)
        if self.training and 0.0 < self.dropout_rate < 1.0:
            h = F.dropout(h, p=self.dropout_rate, training=True)
        return torch.matmul(h, self.W2) + self.b2


class LayerNormalization(nn.Module):
    """``EqT_utils.LayerNormalization``: ``(x - mean) / sqrt(var + 1e-14) * gamma + beta`` over features."""

    def __init__(self, features: int, epsilon: float = LAYER_NORM_EPSILON):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        return (x - mean) / torch.sqrt(variance + self.epsilon) * self.gamma + self.beta


class ResCNNBlock(nn.Module):
    """``_block_CNN_1``: ``x + conv2(drop(relu(bn2(conv1(drop(relu(bn1(x))))))))``."""

    def __init__(self, channels: int, kernel_size: int, zero_debias: bool = False):
        super().__init__()
        self.bn1 = KerasBatchNorm1d(channels, zero_debias)
        self.conv1 = KerasConv1d(channels, channels, kernel_size)
        self.bn2 = KerasBatchNorm1d(channels, zero_debias)
        self.conv2 = KerasConv1d(channels, channels, kernel_size)

    def forward(self, x: torch.Tensor, drop_rate: float, dropout_active: bool) -> torch.Tensor:
        h = self.conv1(_spatial_dropout(F.relu(self.bn1(x)), drop_rate, dropout_active))
        h = self.conv2(_spatial_dropout(F.relu(self.bn2(h)), drop_rate, dropout_active))
        return x + h


class BiLSTMBlock(nn.Module):
    """``_block_BiLSTM``: Bidirectional(LSTM(16)) (outputs concatenated), 1x1 Conv1D(16), BN."""

    def __init__(self, input_size: int, units: int, drop_rate: float, zero_debias: bool = False, **lstm_options: Any):
        super().__init__()
        self.forward_lstm = KerasLSTM(input_size, units, drop_rate, drop_rate, **lstm_options)
        self.backward_lstm = KerasLSTM(input_size, units, drop_rate, drop_rate, **lstm_options)
        self.conv = KerasConv1d(2 * units, units, 1)
        self.bn = KerasBatchNorm1d(units, zero_debias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (batch, time, features) -> (batch, time, units)
        h = torch.cat([self.forward_lstm(x), self.backward_lstm(x, reverse=True)], dim=-1)
        return self.bn(self.conv(h.transpose(1, 2))).transpose(1, 2)


class TransformerBlock(nn.Module):
    """``_transformer``: self-attention, add, layer norm, feed-forward, add, layer norm."""

    def __init__(self, features: int, drop_rate: float):
        super().__init__()
        self.attention = SeqSelfAttention(features)
        self.norm1 = LayerNormalization(features)
        self.feed_forward = FeedForward(features, FEED_FORWARD_UNITS, drop_rate)
        self.norm2 = LayerNormalization(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x + self.attention(x))
        return self.norm2(h + self.feed_forward(h))


class Decoder(nn.Module):
    """``_decoder``: seven UpSampling1D(2) + Conv1D('same', ReLU) steps, Cropping1D(1, 1) before step 4."""

    def __init__(self, in_channels: int):
        super().__init__()
        channels = [in_channels, *DECODER_FILTERS]
        self.convs = nn.ModuleList(
            KerasConv1d(channels[i], channels[i + 1], DECODER_KERNELS[i]) for i in range(len(DECODER_FILTERS))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for step, conv in enumerate(self.convs):
            x = torch.repeat_interleave(x, 2, dim=-1)
            if step == DECODER_CROP_STEP:
                x = x[..., 1:-1]
            x = F.relu(conv(x))
        return x


def _output_length(samples: int) -> int:
    length = samples
    for _ in ENCODER_FILTERS:
        length = math.ceil(length / 2)
    for step in range(len(DECODER_FILTERS)):
        length *= 2
        if step == DECODER_CROP_STEP:
            length -= 2
    return length


class EQTransformer(nn.Module, WaveformPicker):
    """EQTransformer (Mousavi et al., 2020), the released Keras models in PyTorch.

    ``forward`` maps normalised waveforms ``(batch, 3, 6000)`` (60 s at 100 Hz, components E, N, Z)
    to ``(batch, 3, 6000)`` probabilities: detection, P and S (``logits=True`` returns the values
    before the sigmoid). :meth:`annotate` applies the official per-trace normalisation first.
    """

    sampling_rate = 100.0
    component_order = "ENZ"
    output_names = ("detection", "P", "S")
    phase_channels = {"P": 1, "S": 2}
    detection_channel = 0

    def __init__(
        self,
        variant: str = "original",
        in_channels: int = 3,
        drop_rate: Optional[float] = None,
        mc_dropout: bool = False,
    ):
        super().__init__()
        if variant not in EQTRANSFORMER_VARIANTS:
            raise ValueError(f"Unknown EQTransformer variant {variant!r}; expected one of {sorted(EQTRANSFORMER_VARIANTS)}.")
        spec = EQTRANSFORMER_VARIANTS[variant]
        self.variant = variant
        self.in_channels = int(in_channels)
        self.lstm_blocks = int(spec["lstm_blocks"])
        self.drop_rate = float(spec["drop_rate"] if drop_rate is None else drop_rate)
        self.mc_dropout = bool(mc_dropout)
        lstm_options = {
            "recurrent_activation": spec["recurrent_activation"],
            "implementation": spec["lstm_implementation"],
        }

        channels = [self.in_channels, *ENCODER_FILTERS]
        self.encoder = nn.ModuleList(
            KerasConv1d(channels[i], channels[i + 1], ENCODER_KERNELS[i]) for i in range(len(ENCODER_FILTERS))
        )
        width = ENCODER_FILTERS[-1]
        zero_debias = bool(spec["bn_zero_debias"])
        self.res_cnn = nn.ModuleList(ResCNNBlock(width, kernel, zero_debias) for kernel in RES_CNN_KERNELS)
        self.bilstm = nn.ModuleList(
            BiLSTMBlock(width if i == 0 else LSTM_UNITS, LSTM_UNITS, self.drop_rate, zero_debias, **lstm_options)
            for i in range(self.lstm_blocks)
        )
        self.transformer_d0 = TransformerBlock(LSTM_UNITS, self.drop_rate)
        self.transformer_d = TransformerBlock(LSTM_UNITS, self.drop_rate)
        self.decoder_d = Decoder(LSTM_UNITS)
        self.detector = KerasConv1d(DECODER_FILTERS[-1], 1, 11)
        self.lstm_p = KerasLSTM(LSTM_UNITS, LSTM_UNITS, self.drop_rate, self.drop_rate, **lstm_options)
        self.attention_p = SeqSelfAttention(LSTM_UNITS, ATTENTION_UNITS, attention_width=3)
        self.decoder_p = Decoder(LSTM_UNITS)
        self.picker_p = KerasConv1d(DECODER_FILTERS[-1], 1, 11)
        self.lstm_s = KerasLSTM(LSTM_UNITS, LSTM_UNITS, self.drop_rate, self.drop_rate, **lstm_options)
        self.attention_s = SeqSelfAttention(LSTM_UNITS, ATTENTION_UNITS, attention_width=3)
        self.decoder_s = Decoder(LSTM_UNITS)
        self.picker_s = KerasConv1d(DECODER_FILTERS[-1], 1, 11)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder output ``(batch, 47, 16)`` (time-major, as the Keras ``layer_normalization_4``)."""
        h = x
        for conv in self.encoder:
            h = _max_pool_same(F.relu(conv(h)))
        dropout_active = self.training or self.mc_dropout
        for block in self.res_cnn:
            h = block(h, self.drop_rate, dropout_active)
        s = h.transpose(1, 2)
        for block in self.bilstm:
            s = block(s)
        return self.transformer_d(self.transformer_d0(s))

    def forward(self, x: torch.Tensor, logits: bool = False) -> torch.Tensor:
        check_waveforms(x, self.in_channels, "EQTransformer")
        if _output_length(x.shape[-1]) != x.shape[-1]:
            raise ValueError(
                f"EQTransformer expects waveforms shaped (batch, {self.in_channels}, {IN_SAMPLES}) "
                f"(60 s at 100 Hz); got shape {tuple(x.shape)}."
            )
        encoded = self.encode(x)
        detection = self.detector(self.decoder_d(encoded.transpose(1, 2)))
        p = self.picker_p(self.decoder_p(self.attention_p(self.lstm_p(encoded)).transpose(1, 2)))
        s = self.picker_s(self.decoder_s(self.attention_s(self.lstm_s(encoded)).transpose(1, 2)))
        out = torch.cat([detection, p, s], dim=1)
        return out if logits else torch.sigmoid(out)

    def annotate(self, waveforms: torch.Tensor, mc_samples: int = 1) -> torch.Tensor:
        """Official preprocessing (``normalize(data, 'std')``) and forward; mean of ``mc_samples`` passes.

        With ``mc_samples > 1`` the passes are drawn with the SpatialDropout1D of the residual blocks
        active, as the official tester does with ``estimate_uncertainty=True``.
        """
        x = std_normalize(waveforms)
        if mc_samples <= 1:
            return self(x)
        previous = self.mc_dropout
        self.mc_dropout = True
        try:
            return torch.stack([self(x) for _ in range(int(mc_samples))]).mean(dim=0)
        finally:
            self.mc_dropout = previous

    def extract_detections(
        self,
        annotations: torch.Tensor,
        detection_threshold: float = 0.2,
        p_threshold: float = 0.1,
        s_threshold: float = 0.1,
    ) -> List[List[Tuple[int, int, float]]]:
        """Events reported by the official tester: ``(on, off, mean detection probability)`` per trace.

        These are the ``matches`` of ``EqT_utils.picker``: detection triggers of at least 10 samples
        that have a P or an S pick (the tester's ``number_of_detections``).
        """
        return [
            [(on, event["off"], event["detection_probability"]) for on, event in events]
            for events in self._match(annotations, detection_threshold, p_threshold, s_threshold)
        ]

    def extract_picks(
        self,
        annotations: torch.Tensor,
        detection_threshold: float = 0.2,
        p_threshold: float = 0.1,
        s_threshold: float = 0.1,
        all_events: bool = False,
    ) -> List[TracePicks]:
        """P and S picks per trace with the official tester's ``picker`` (``EqT_utils.py``).

        Defaults are the thresholds of the official ``tester()`` (detection 0.2, P 0.1, S 0.1). Like the
        tester's output table, only the first detected event of each trace is reported unless
        ``all_events=True``. Probabilities are rounded to three decimals, as in the reference.
        """
        picks: List[TracePicks] = []
        for events in self._match(annotations, detection_threshold, p_threshold, s_threshold):
            trace: TracePicks = {"P": [], "S": []}
            for _, event in events if all_events else events[:1]:
                for phase in ("P", "S"):
                    if event[phase] is not None:
                        trace[phase].append(event[phase])
            picks.append(trace)
        return picks

    @staticmethod
    def _match(annotations: torch.Tensor, detection_threshold: float, p_threshold: float, s_threshold: float):
        """``EqT_utils.picker`` without uncertainties: per trace, a list of ``(on, event)``."""
        from ..metrics.picking import detect_peaks, trigger_onset

        if annotations.ndim != 3 or annotations.shape[1] != 3:
            raise ValueError(
                f"Expected EQTransformer annotations shaped (batch, 3, samples), got {tuple(annotations.shape)}."
            )
        values = annotations.detach().cpu().numpy()  # keep the dtype: the reference rounds float32 values
        results = []
        for yh1, yh2, yh3 in values:
            detection = trigger_onset(yh1, detection_threshold, detection_threshold)
            # Probabilities are rounded to three decimals in the array's dtype, as the reference does
            # before choosing the most probable P.
            p_picks = {int(i): np.round(yh2[int(i)], 3) for i in detect_peaks(yh2, mph=p_threshold, mpd=1) if i}
            s_picks = {int(i): np.round(yh3[int(i)], 3) for i in detect_peaks(yh3, mph=s_threshold, mpd=1) if i}
            events = []
            for on, off in detection:
                on, off = int(on), int(off)
                if off - on < 10:
                    continue
                with np.errstate(invalid="ignore"):
                    probability = round(float(np.round(np.mean(yh1[on:off]), 3)), 3)
                s_candidates = [s for s in s_picks if on < s < off]
                s_pick = s_candidates[0] if s_candidates else None  # the earliest S inside the window
                if s_pick:
                    p_candidates = [p for p in p_picks if on - 100 < p < s_pick - 10]
                else:
                    p_candidates = [p for p in p_picks if on - 100 < p < off]
                p_pick, best = None, 0
                for p in p_candidates:  # the most probable P (the first one on ties)
                    if p_picks[p] > best:
                        p_pick, best = p, p_picks[p]
                if s_pick or p_pick:
                    events.append(
                        (
                            on,
                            {
                                "off": off,
                                "detection_probability": probability,
                                "P": None if p_pick is None else (float(p_pick), round(float(p_picks[p_pick]), 3)),
                                "S": None if s_pick is None else (float(s_pick), round(float(s_picks[s_pick]), 3)),
                            },
                        )
                    )
            results.append(events)
        return results


def keras_weight_map(lstm_blocks: int) -> Dict[str, Tuple[str, str]]:
    """``{pytorch state-dict key: (Keras layer name, Keras weight name)}`` for a released model.

    Keras numbers unnamed layers per class in creation order (``cred2``): encoder convs 1-7, residual
    convs 8-21 and batch norms 1-14, the BiLSTM blocks' 1x1 convs and batch norms, then the detection,
    P and S decoders' convs; the two decoder LSTMs come after the bidirectional ones.
    """
    mapping: Dict[str, Tuple[str, str]] = {}

    def conv(prefix: str, layer: str) -> None:
        mapping[f"{prefix}.weight"] = (layer, "kernel")
        mapping[f"{prefix}.bias"] = (layer, "bias")

    def bn(prefix: str, layer: str) -> None:
        for torch_name, keras_name in (("weight", "gamma"), ("bias", "beta"), ("running_mean", "moving_mean"), ("running_var", "moving_variance")):
            mapping[f"{prefix}.{torch_name}"] = (layer, keras_name)

    def lstm(prefix: str, layer: str, inner: str) -> None:
        for name in ("kernel", "recurrent_kernel", "bias"):
            mapping[f"{prefix}.{name}"] = (layer, f"{inner}{name}")

    conv_id, bn_id = 0, 0
    for i in range(len(ENCODER_FILTERS)):
        conv_id += 1
        conv(f"encoder.{i}", f"conv1d_{conv_id}")
    for i in range(len(RES_CNN_KERNELS)):
        bn_id += 1
        bn(f"res_cnn.{i}.bn1", f"batch_normalization_{bn_id}")
        conv_id += 1
        conv(f"res_cnn.{i}.conv1", f"conv1d_{conv_id}")
        bn_id += 1
        bn(f"res_cnn.{i}.bn2", f"batch_normalization_{bn_id}")
        conv_id += 1
        conv(f"res_cnn.{i}.conv2", f"conv1d_{conv_id}")
    for i in range(lstm_blocks):
        layer = f"bidirectional_{i + 1}"
        lstm(f"bilstm.{i}.forward_lstm", layer, f"forward_lstm_{i + 1}/")
        lstm(f"bilstm.{i}.backward_lstm", layer, f"backward_lstm_{i + 1}/")
        conv_id += 1
        conv(f"bilstm.{i}.conv", f"conv1d_{conv_id}")
        bn_id += 1
        bn(f"bilstm.{i}.bn", f"batch_normalization_{bn_id}")
    for block, attention, norms, feed_forward in (
        ("transformer_d0", "attentionD0", (1, 2), 1),
        ("transformer_d", "attentionD", (3, 4), 2),
    ):
        for name in ("Wt", "Wx", "bh", "Wa", "ba"):
            mapping[f"{block}.attention.{name}"] = (attention, f"{attention}_Add_{name}")
        for norm, index in zip(("norm1", "norm2"), norms):
            mapping[f"{block}.{norm}.gamma"] = (f"layer_normalization_{index}", "gamma")
            mapping[f"{block}.{norm}.beta"] = (f"layer_normalization_{index}", "beta")
        for name in ("W1", "b1", "W2", "b2"):
            mapping[f"{block}.feed_forward.{name}"] = (f"feed_forward_{feed_forward}", f"feed_forward_{feed_forward}_{name}")
    for decoder in ("decoder_d", "decoder_p", "decoder_s"):
        for i in range(len(DECODER_FILTERS)):
            conv_id += 1
            conv(f"{decoder}.convs.{i}", f"conv1d_{conv_id}")
    conv("detector", "detector")
    conv("picker_p", "picker_P")
    conv("picker_s", "picker_S")
    for offset, branch in ((1, "p"), (2, "s")):
        lstm(f"lstm_{branch}", f"lstm_{lstm_blocks + offset}", "")
        for name in ("Wt", "Wx", "bh", "Wa", "ba"):
            key = "attentionP" if branch == "p" else "attentionS"
            mapping[f"attention_{branch}.{name}"] = (key, f"{key}_Add_{name}")
    return mapping


def read_keras_h5_weights(path: Union[str, Path]) -> Dict[Tuple[str, str], np.ndarray]:
    """All weights of a Keras HDF5 model file as ``{(layer, weight name without ':0'): array}``."""
    import h5py

    weights: Dict[Tuple[str, str], np.ndarray] = {}
    with h5py.File(str(path), "r") as handle:
        group = handle["model_weights"] if "model_weights" in handle else handle
        for raw_layer in group.attrs["layer_names"]:
            layer = raw_layer.decode() if isinstance(raw_layer, bytes) else str(raw_layer)
            for raw_name in group[layer].attrs["weight_names"]:
                name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
                short = name.split(":")[0]
                short = short[len(layer) + 1 :] if short.startswith(layer + "/") else short
                weights[(layer, short)] = np.asarray(group[layer][name])
    return weights


def keras_state_dict(model: EQTransformer, path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """State dict of ``model`` filled from a released Keras ``.h5`` file (every Keras weight used once)."""
    weights = read_keras_h5_weights(path)
    mapping = keras_weight_map(model.lstm_blocks)
    state = model.state_dict()
    expected = {key for key in state if not key.endswith("num_batches_tracked")}
    if set(mapping) != expected:
        raise RuntimeError(f"Weight map does not cover the model: {sorted(expected ^ set(mapping))}")
    used = set()
    converted: Dict[str, torch.Tensor] = {}
    for key, (layer, name) in mapping.items():
        if (layer, name) not in weights:
            raise KeyError(f"{path}: Keras weight {layer}/{name} not found (variant {model.variant!r}?)")
        value = torch.from_numpy(np.array(weights[(layer, name)], dtype=np.float32))
        if name == "kernel" and value.ndim == 3:
            value = value.permute(2, 1, 0)  # Conv1D (width, in, out) -> (out, in, width)
        if tuple(value.shape) != tuple(state[key].shape):
            raise ValueError(f"{layer}/{name}: shape {tuple(value.shape)} does not match {key} {tuple(state[key].shape)}")
        converted[key] = value.contiguous()
        used.add((layer, name))
    unused = set(weights) - used
    if unused:
        raise ValueError(f"{path}: Keras weights not used by the port: {sorted(unused)}")
    for key, value in state.items():
        if key.endswith("num_batches_tracked"):
            converted[key] = value
    return converted


def eqtransformer_weights_path(name: str) -> Path:
    """Local path of an official model (``"original"`` or ``"conservative"``), downloaded on first use."""
    if name not in EQTRANSFORMER_WEIGHTS:
        raise ValueError(f"Unknown EQTransformer weights {name!r}; expected one of {sorted(EQTRANSFORMER_WEIGHTS)}.")
    spec = EQTRANSFORMER_WEIGHTS[name]
    return cached_download("eqtransformer", spec["filename"], spec["urls"], spec["sha256"])


def load_eqtransformer_keras_weights(model: EQTransformer, source: Union[str, Path]) -> EQTransformer:
    """Load a released Keras model (``"original"``, ``"conservative"`` or a path to an ``.h5``) strictly."""
    path = eqtransformer_weights_path(str(source)) if str(source) in EQTRANSFORMER_WEIGHTS else Path(source)
    model.load_state_dict(keras_state_dict(model, path), strict=True)
    return model


def eqtransformer_builder(
    task: str,
    variant: str = "original",
    in_channels: int = 3,
    pretrained: Optional[Union[bool, str, Path]] = None,
    mc_dropout: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """EQTransformer for ``task="picking"``: ``(batch, 3, 6000)`` -> ``(batch, 3, 6000)`` probabilities.

    ``variant``: ``"original"`` (paper model, default) or ``"conservative"``. ``pretrained``: ``True``
    (the released weights of ``variant``), ``"original"`` / ``"conservative"`` (must equal ``variant``)
    or a path to a Keras ``.h5`` file of the same variant. ``mc_dropout=True`` keeps the residual
    blocks' SpatialDropout1D active in eval mode, as in the released Keras graphs.
    """
    kwargs.pop("name", None)
    if kwargs:
        raise TypeError(f"eqtransformer_builder got unexpected arguments: {sorted(kwargs)}")
    if task.lower() != "picking":
        raise ValueError(f"eqtransformer supports task='picking', got {task!r}.")
    if variant not in EQTRANSFORMER_VARIANTS:
        raise ValueError(f"Unknown EQTransformer variant {variant!r}; expected one of {sorted(EQTRANSFORMER_VARIANTS)}.")
    model = EQTransformer(variant=variant, in_channels=in_channels, mc_dropout=mc_dropout)
    if pretrained is not None and pretrained is not False:
        if int(in_channels) != 3:
            raise ValueError("The released EQTransformer weights need in_channels=3.")
        source = variant if pretrained is True else pretrained
        if str(source) in EQTRANSFORMER_WEIGHTS and str(source) != variant:
            raise ValueError(f"Weights {source!r} belong to variant {source!r}, but variant={variant!r}.")
        load_eqtransformer_keras_weights(model, source)
    return model


__all__ = [
    "EQTRANSFORMER_VARIANTS",
    "EQTRANSFORMER_WEIGHTS",
    "EQTransformer",
    "KerasBatchNorm1d",
    "eqtransformer_builder",
    "eqtransformer_weights_path",
    "keras_state_dict",
    "keras_weight_map",
    "load_eqtransformer_keras_weights",
]
