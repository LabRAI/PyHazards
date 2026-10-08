"""WaveCastNet: ConvLEM sequence-to-sequence forecasting of earthquake ground-motion wavefields.

Lyu, Nakata, Ren, Mahoney, Pitarka, Nakata & Erichson, "Rapid wavefield forecasting for earthquake early
warning via deep sequence to sequence learning", Nature Communications 16:10622 (2025),
doi:10.1038/s41467-025-65435-2 (arXiv 2405.20516).

Port of the official PyTorch code, dwlyu/WaveCastNet at commit
``c859e04c85acd6657306f08bce8c6f9129ba1480`` (MIT License, Copyright (c) 2023 dwlyu):
``src/models_earthquake/ConvLEMCell.py`` (``ConvLEMCell``, ``ConvLEMCell_1``), ``EncoderDecoder.py``
(``encoder_layer``, ``encoder1``, ``encoder_sparser``, ``decoder_layer``, ``decoder2_48``),
``AEConvLEM_dense.py``, ``AEConvLEM_sparse.py``, ``EncodingSparse.py`` (``Encoder1d``) and
``earthquake_train.py`` (``Huber`` loss, rolling validation forecast).

Architecture (dense configuration, 10,093,242 parameters): every input frame ``(3, 344, 224)`` goes
through an embedding of three ``Conv2d(k4, s2, p1) -> BatchNorm2d -> LeakyReLU`` layers (3 -> 36 -> 72 ->
144 channels, latent grid 43 x 28); two stacked ConvLEM cells with reset gate (``ConvLEMCell_1``) encode
the sequence; two more ConvLEM cells, started from the encoder states, decode ``future_seq`` latent frames;
each latent frame is reconstructed by ``ConvTranspose2d(k4, s2, p1) -> LeakyReLU -> PixelShuffle(4)`` and
a 3x3 convolution. The sparse variant (``WaveCastNetSparse``, official ``AEConvLEM_sparse``) samples the
wavefield at station grid points, zeroes a random subset of the stations (masked-autoencoder training)
and embeds the station vector with two fully connected layers and two convolutions.

Faithfulness notes:

- Module and parameter names equal the official ones (``encoder.model.encoder_layer1.layer.0.weight``,
  ``encoder_1_convlem.convx.weight``, ``W_z1``, ...) and modules are created in the official order with
  the official initialisation (xavier on every tensor with two or more dimensions, including the per-pixel
  peephole weights ``W_z*`` of shape ``(channels, H/8, W/8)``; zeros for biases), so the same seed gives
  the same weights and ``best_lem_dense_.pt`` loads with ``strict=True``.
- The decoder's first input is uniform noise, ``torch.rand_like`` of the latent state, in training *and*
  evaluation, so two forward passes on the same input differ. ``forward(..., generator=g)`` draws that
  noise (and the sparse station mask) from ``g`` instead of the global generator, and
  ``decoder_noise=`` supplies it explicitly; without either, the global generator is used in the official
  order (station mask first, then decoder noise), so ``torch.manual_seed`` reproduces the official outputs.
- The sparse station mask is drawn in evaluation too (the official ``Encoder1d.forward`` always masks);
  ``station_mask=`` (``True`` = station kept) replaces the random draw.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from ..metrics.wavefield import wavefield_acc, wavefield_rfne, wavefield_rmse
from ._pretrained import cached_download

WAVECASTNET_COMMIT = "c859e04c85acd6657306f08bce8c6f9129ba1480"
_RAW = f"https://raw.githubusercontent.com/dwlyu/WaveCastNet/{WAVECASTNET_COMMIT}/src/models_earthquake/"
_DRIVE = "https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"

# Released checkpoints (README "Data Availability", Google Drive folder 10pe6Zc1NEzIunwJv80214dB9fmb-X8yw).
WAVECASTNET_WEIGHTS: Dict[str, Dict[str, str]] = {
    "dense": {
        "filename": "best_lem_dense_.pt",
        "file_id": "1vFoo1eqO2WoZH0R6kkRZJ8XhNG3zF8nO",
        "sha256": "cccadaa7ae02736eb1835c0dc2b04a6291cdd348c901128fb03c92f4c7647acc",
        "license": "not stated (Google Drive release of the MIT-licensed dwlyu/WaveCastNet)",
    },
}

# Station grid indices (row, column) on the 344 x 224 grid, shipped with the official code (MIT).
WAVECASTNET_STATIONS: Dict[str, Dict[str, str]] = {
    "candidates": {
        "filename": "filtered_coord.npy",
        "sha256": "c6cb49cf61b4f4229bd8f8206538218889368d5bd3c96fbbb3350e7ec8e8bb1e",
    },
    "shakealert": {
        "filename": "shakealert_coords.npy",
        "sha256": "b64f6b37cfe15755214e8c2e07ac2cb772d6b80f167cf14a41ec5d901ee6ccc4",
    },
}

# README / load_pretrained_model.ipynb: mask_ratio = 1 - len(shakealert_coords) / len(filtered_coord).
DEFAULT_MASK_RATIO = 1.0 - 101.0 / 564.0

_ACTIVATIONS = {"tanh": torch.tanh, "relu": torch.relu}


def _pair(value: Union[int, Sequence[int]], name: str) -> Tuple[int, int]:
    if isinstance(value, int):
        return int(value), int(value)
    values = tuple(int(item) for item in value)
    if len(values) != 2:
        raise ValueError(f"{name} must be an int or a pair, got {value!r}.")
    return values


class ConvLEMCell(nn.Module):
    """Convolutional Long Expressive Memory cell (official ``ConvLEMCell_1`` / ``ConvLEMCell``).

    ``reset_gate=True`` is ``ConvLEMCell_1`` (the cell of both WaveCastNet models: an extra gate scales the
    convolved memory; xavier-uniform initialisation); ``reset_gate=False`` is ``ConvLEMCell``
    (xavier-normal initialisation). Peephole weights ``W_z*`` are per pixel, shaped
    ``(out_channels, *frame_size)``, so the cell only runs on that grid. ``forward(x, h, c)`` takes
    ``(batch, channels, *frame_size)`` tensors and returns the new ``(h, c)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        frame_size: Union[int, Sequence[int]],
        kernel_size: Union[int, Sequence[int]] = 3,
        padding: Optional[Union[int, Sequence[int]]] = None,
        dt: float = 1.0,
        activation: str = "tanh",
        reset_gate: bool = True,
    ):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unsupported activation {activation!r}; use 'tanh' or 'relu'.")
        self.activation = _ACTIVATIONS[activation]
        self.reset_gate = bool(reset_gate)
        self.frame_size = _pair(frame_size, "frame_size")
        kernel_size = _pair(kernel_size, "kernel_size")
        padding = tuple(k // 2 for k in kernel_size) if padding is None else _pair(padding, "padding")
        x_gates, h_gates = (5, 4) if self.reset_gate else (4, 3)
        self.convx = nn.Conv2d(in_channels, x_gates * out_channels, kernel_size, padding=padding)
        self.convy = nn.Conv2d(out_channels, h_gates * out_channels, kernel_size, padding=padding)
        self.convz = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.dt = dt
        self.W_z1 = nn.Parameter(torch.empty(out_channels, *self.frame_size))
        self.W_z2 = nn.Parameter(torch.empty(out_channels, *self.frame_size))
        if self.reset_gate:
            self.W_z4 = nn.Parameter(torch.empty(out_channels, *self.frame_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Official order: the cell's own peephole tensors first, then convx, convy, convz.
        xavier = init.xavier_uniform_ if self.reset_gate else init.xavier_normal_
        for param in self.parameters():
            if param.ndim > 1:
                xavier(param)
            else:
                nn.init.constant_(param, 0)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or h.shape != c.shape or tuple(h.shape[-2:]) != self.frame_size or x.shape[-2:] != h.shape[-2:]:
            raise ValueError(
                f"ConvLEMCell expects x, h, c shaped (batch, channels, {self.frame_size[0]}, {self.frame_size[1]}); "
                f"got shapes {tuple(x.shape)}, {tuple(h.shape)}, {tuple(c.shape)}."
            )
        transformed_inp = self.convx(x)
        transformed_hid = self.convy(h)
        if self.reset_gate:
            i_dt1, i_dt2, g_dx2, i_z, i_y = torch.chunk(transformed_inp, chunks=5, dim=1)
            h_dt1, h_dt2, h_y, g_dy2 = torch.chunk(transformed_hid, chunks=4, dim=1)
        else:
            i_dt1, i_dt2, i_z, i_y = torch.chunk(transformed_inp, chunks=4, dim=1)
            h_dt1, h_dt2, h_y = torch.chunk(transformed_hid, chunks=3, dim=1)
        ms_dt = self.dt * torch.sigmoid(i_dt2 + h_dt2 + self.W_z2 * c)
        c = (1.0 - ms_dt) * c + ms_dt * self.activation(i_y + h_y)
        if self.reset_gate:
            ms_dt_bar = self.dt * torch.sigmoid(i_dt1 + h_dt1 + self.W_z1 * c)
            gate2 = self.dt * torch.sigmoid(g_dx2 + g_dy2 + self.W_z4 * c)
            transformed_z = gate2 * self.convz(c)
        else:
            transformed_z = self.convz(c)
            ms_dt_bar = self.dt * torch.sigmoid(i_dt1 + h_dt1 + self.W_z1 * c)
        h = (1.0 - ms_dt_bar) * h + ms_dt_bar * self.activation(transformed_z + i_z)
        return h, c


class _EncoderLayer(nn.Module):
    """Official ``encoder_layer``: ``Conv2d(k4, s2, p1) -> BatchNorm2d -> LeakyReLU``."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class DenseEmbedding(nn.Module):
    """Official ``encoder1``: three encoder layers, ``C -> 12C -> 24C -> 48C`` channels, grid / 8."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.model = nn.Sequential()
        self.model.add_module("encoder_layer1", _EncoderLayer(num_channels, 12 * num_channels))
        self.model.add_module("encoder_layer2", _EncoderLayer(12 * num_channels, 24 * num_channels))
        self.model.add_module("encoder_layer3", _EncoderLayer(24 * num_channels, 48 * num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _SparseConvEncoder(nn.Module):
    """Official ``encoder_sparser``: ``Conv2d(k3) -> BatchNorm2d -> LeakyReLU`` then one encoder layer."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(num_channels, 24 * num_channels, kernel_size=(3, 3), padding=(1, 1), bias=True),
            nn.BatchNorm2d(24 * num_channels),
            nn.LeakyReLU(),
        )
        self.model.add_module("encoder_layer2", _EncoderLayer(24 * num_channels, 48 * num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SparseEmbedding(nn.Module):
    """Official ``Encoder1d``: station sampling, masking, two dense layers and a convolutional encoder.

    ``station_coords`` are ``(stations, 2)`` (row, column) grid indices. A frame ``(batch, C, H, W)`` is
    sampled at the stations, the stations with ``keep == False`` are zeroed, ``FC1`` / ``FC2`` (each
    ``Linear -> LeakyReLU -> BatchNorm1d(C)``) map the ``stations`` values of each channel to an
    ``(H/4, W/4)`` grid, and the convolutional encoder brings it to ``(48C, H/8, W/8)``. At the official
    344 x 224 grid the hidden width is 1,204 and the ``FC2`` output 86 x 56 = 4,816.
    """

    def __init__(self, num_channels: int, station_coords: torch.Tensor, height: int, width: int):
        super().__init__()
        coords = torch.as_tensor(np.asarray(station_coords)).round().long()
        if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) == 0:
            raise ValueError(f"station_coords must be shaped (stations, 2), got shape {tuple(coords.shape)}.")
        if (coords < 0).any() or (coords[:, 0] >= height).any() or (coords[:, 1] >= width).any():
            raise ValueError(f"station_coords must index the {height} x {width} grid.")
        self.num_channels = int(num_channels)
        self.coarse = (height // 4, width // 4)
        coarse_size = self.coarse[0] * self.coarse[1]
        hidden = coarse_size // 4
        self.FC1 = nn.Sequential(nn.Linear(len(coords), hidden), nn.LeakyReLU(), nn.BatchNorm1d(num_channels))
        self.FC2 = nn.Sequential(nn.Linear(hidden, coarse_size), nn.LeakyReLU(), nn.BatchNorm1d(num_channels))
        self.encoder = _SparseConvEncoder(num_channels)
        self.register_buffer("station_coords", coords, persistent=False)

    @property
    def num_stations(self) -> int:
        return int(self.station_coords.shape[0])

    def forward(self, x: torch.Tensor, keep: Optional[torch.Tensor] = None) -> torch.Tensor:
        sampled = x[:, :, self.station_coords[:, 0], self.station_coords[:, 1]]  # (batch, C, stations)
        if keep is not None:
            sampled = sampled.masked_fill(~keep.unsqueeze(1).expand_as(sampled), 0)
        out = self.FC2(self.FC1(sampled))
        return self.encoder(out.reshape(x.shape[0], self.num_channels, *self.coarse))


class _DecoderLayer(nn.Module):
    """Official ``decoder_layer``: ``ConvTranspose2d(k4, s2, p1) -> LeakyReLU``."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class Reconstruction(nn.Module):
    """Official ``decoder2_48``: ``(48C, H/8, W/8) -> ConvTranspose2d -> PixelShuffle(4) -> (3C, H, W)``."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.model = nn.Sequential()
        self.model.add_module("decoder_layer1", _DecoderLayer(48 * num_channels, 48 * num_channels))
        self.model.add_module("decoder_layer2", nn.PixelShuffle(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class WaveCastNet(nn.Module):
    """WaveCastNet with dense input (official ``AEConvLEM_dense``).

    ``forward(x, future_seq=None)``: ``x`` is ``(batch, channels, time, height, width)``; returns
    ``(batch, channels, future_seq, height, width)``. ``future_seq`` defaults to the value given at
    construction (30 steps, the official training horizon). ``height`` and ``width`` must be multiples of
    8 (latent grid ``(height / 8, width / 8)``, the shape of the per-pixel peepholes); the latent width is
    fixed by the official code at ``48 * channels`` (144 for three components).

    Keyword-only hooks (see the module docstring): ``generator`` (a ``torch.Generator`` for the decoder
    noise and the sparse station mask) and ``decoder_noise`` (the first decoder input itself, shaped like
    the latent state ``(batch, 48 * channels, height / 8, width / 8)``). :meth:`rollout` repeats the
    forecast on its own output, as the official validation does.
    """

    sampling = "dense"

    def __init__(
        self,
        num_channels: int = 3,
        height: int = 344,
        width: int = 224,
        kernel_size: Union[int, Sequence[int]] = 3,
        padding: Optional[Union[int, Sequence[int]]] = None,
        dt: float = 1.0,
        activation: str = "tanh",
        future_seq: int = 30,
        num_kernels: Optional[int] = None,
        station_coords: Optional[torch.Tensor] = None,
        mask_mode: bool = False,
        mask_ratio: float = 0.0,
    ):
        super().__init__()
        if int(num_channels) < 1:
            raise ValueError(f"num_channels must be positive, got {num_channels}.")
        if int(height) % 8 or int(width) % 8 or int(height) < 8 or int(width) < 8:
            raise ValueError(
                f"WaveCastNet needs height and width that are multiples of 8 (latent grid / 8), got {height} x {width}."
            )
        latent = 48 * int(num_channels)
        if num_kernels is not None and int(num_kernels) != latent:
            raise ValueError(
                f"The official embedding fixes the latent width at 48 * num_channels = {latent}; got num_kernels={num_kernels}."
            )
        if int(future_seq) < 1:
            raise ValueError(f"future_seq must be at least 1, got {future_seq}.")
        if not 0.0 <= float(mask_ratio) <= 1.0:
            raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}.")
        self.num_channels = int(num_channels)
        self.height, self.width = int(height), int(width)
        self.frame_size = (self.height // 8, self.width // 8)
        self.out_channels = latent
        self.future_seq = int(future_seq)
        self.mask_mode = bool(mask_mode)
        self.mask_ratio = float(mask_ratio)
        kernel_size = _pair(kernel_size, "kernel_size")
        padding = tuple(k // 2 for k in kernel_size) if padding is None else _pair(padding, "padding")

        # Creation order of the official __init__: embedding, reconstruction, four cells, conv1.
        if self.sampling == "dense":
            self.encoder = DenseEmbedding(self.num_channels)
        else:
            if station_coords is None:
                raise ValueError("The sparse WaveCastNet needs station_coords shaped (stations, 2).")
            self.encoder = SparseEmbedding(self.num_channels, station_coords, self.height, self.width)
        self.decoder = Reconstruction(self.num_channels)
        cell = dict(
            in_channels=latent,
            out_channels=latent,
            frame_size=self.frame_size,
            kernel_size=kernel_size,
            padding=padding,
            dt=dt,
            activation=activation,
            reset_gate=True,
        )
        self.encoder_1_convlem = ConvLEMCell(**cell)
        self.encoder_2_convlem = ConvLEMCell(**cell)
        self.decoder_1_convlem = ConvLEMCell(**cell)
        self.decoder_2_convlem = ConvLEMCell(**cell)
        self.conv1 = nn.Conv2d(3 * self.num_channels, self.num_channels, kernel_size=kernel_size, padding=padding)

    def _check_input(self, x: torch.Tensor) -> None:
        if x.ndim != 5 or x.shape[1] != self.num_channels or tuple(x.shape[-2:]) != (self.height, self.width):
            raise ValueError(
                f"WaveCastNet expects x shaped (batch, {self.num_channels}, time, {self.height}, {self.width}), "
                f"got shape {tuple(x.shape)}."
            )

    def _station_keep(self, batch: int, x: torch.Tensor, generator, station_mask) -> Optional[torch.Tensor]:
        return None

    def _decoder_start(self, h: torch.Tensor, generator, decoder_noise) -> torch.Tensor:
        if decoder_noise is not None:
            if tuple(decoder_noise.shape) != tuple(h.shape):
                raise ValueError(
                    f"decoder_noise must have the latent shape {tuple(h.shape)}, got shape {tuple(decoder_noise.shape)}."
                )
            return decoder_noise.to(device=h.device, dtype=h.dtype)
        if generator is None:
            return torch.rand_like(h)  # official: uniform noise from the global generator
        return torch.rand(h.shape, generator=generator, dtype=h.dtype, device=generator.device).to(h.device)

    def forward(
        self,
        x: torch.Tensor,
        future_seq: Optional[int] = None,
        *,
        generator: Optional[torch.Generator] = None,
        decoder_noise: Optional[torch.Tensor] = None,
        station_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._check_input(x)
        steps = self.future_seq if future_seq is None else int(future_seq)
        if steps < 1:
            raise ValueError(f"future_seq must be at least 1, got {future_seq}.")
        batch, _, seq_len = x.shape[:3]
        state = (batch, self.out_channels, *self.frame_size)
        h1, c1, h2, c2 = (x.new_zeros(state) for _ in range(4))
        keep = self._station_keep(batch, x, generator, station_mask)
        for t in range(seq_len):
            frame = x[:, :, t]
            embedded = self.encoder(frame) if keep is None else self.encoder(frame, keep)
            h1, c1 = self.encoder_1_convlem(embedded, h1, c1)
            h2, c2 = self.encoder_2_convlem(h1, h2, c2)
        h3, c3, h4, c4 = h1, c1, h2, c2
        decoder_input = self._decoder_start(h1, generator, decoder_noise)
        outputs = []
        for _ in range(steps):
            h3, c3 = self.decoder_1_convlem(decoder_input, h3, c3)
            h4, c4 = self.decoder_2_convlem(h3, h4, c4)
            decoder_input = h4
            outputs.append(self.conv1(self.decoder(h4)))
        return torch.stack(outputs, dim=2)

    def rollout(
        self,
        x: torch.Tensor,
        steps: int,
        *,
        step: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        station_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forecast ``steps`` frames by repeated calls of ``step`` frames (default ``future_seq``), each
        taking the previous output as its input; the last call is shortened to fit. With 30-step inputs,
        ``step=30`` and ``steps=195`` this is the official validation forecast (six calls of 30 frames and
        one of 15, 101.4 s at 0.52 s per frame)."""
        step = self.future_seq if step is None else int(step)
        if int(steps) < 1 or step < 1:
            raise ValueError(f"rollout needs steps >= 1 and step >= 1, got {steps} and {step}.")
        outputs, remaining, current = [], int(steps), x
        while remaining > 0:
            count = min(step, remaining)
            current = self(current, count, generator=generator, station_mask=station_mask)
            outputs.append(current)
            remaining -= count
        return torch.cat(outputs, dim=2)


class WaveCastNetSparse(WaveCastNet):
    """WaveCastNet with sparse station input (official ``AEConvLEM_sparse``).

    The input is still the dense wavefield ``(batch, channels, time, height, width)``: the embedding
    reads it at ``station_coords`` (``(stations, 2)`` row / column indices; the official 564 candidate
    stations are :func:`wavecastnet_station_coords` ``("candidates")``). With ``mask_mode`` (the
    official configuration), one uniform draw per (sample, station) is taken per call and stations with a
    draw below ``mask_ratio`` are zeroed at every time step, in evaluation too; ``station_mask``
    (``(stations,)`` or ``(batch, stations)`` booleans, ``True`` = kept) replaces that draw.
    """

    sampling = "sparse"

    def __init__(
        self,
        station_coords: torch.Tensor,
        num_channels: int = 3,
        height: int = 344,
        width: int = 224,
        kernel_size: Union[int, Sequence[int]] = 3,
        padding: Optional[Union[int, Sequence[int]]] = None,
        dt: float = 1.0,
        activation: str = "tanh",
        future_seq: int = 30,
        num_kernels: Optional[int] = None,
        mask_mode: bool = True,
        mask_ratio: float = DEFAULT_MASK_RATIO,
    ):
        super().__init__(
            num_channels=num_channels,
            height=height,
            width=width,
            kernel_size=kernel_size,
            padding=padding,
            dt=dt,
            activation=activation,
            future_seq=future_seq,
            num_kernels=num_kernels,
            station_coords=station_coords,
            mask_mode=mask_mode,
            mask_ratio=mask_ratio,
        )

    def _station_keep(self, batch: int, x: torch.Tensor, generator, station_mask) -> Optional[torch.Tensor]:
        stations = self.encoder.num_stations
        if station_mask is not None:
            keep = torch.as_tensor(station_mask, device=x.device).bool()
            if keep.shape == (stations,):
                keep = keep.unsqueeze(0).expand(batch, stations)
            if keep.shape != (batch, stations):
                raise ValueError(
                    f"station_mask must be shaped ({stations},) or ({batch}, {stations}), got shape {tuple(keep.shape)}."
                )
            return keep
        if not self.mask_mode:
            return None
        if generator is None:
            draw = torch.rand(batch, stations).to(x.device)  # official: torch.rand(batch, stations).to(device)
        else:
            draw = torch.rand((batch, stations), generator=generator, device=generator.device).to(x.device)
        return draw >= self.mask_ratio


def wavecastnet_station_coords(name: str = "candidates") -> torch.Tensor:
    """Official station grid indices ``(stations, 2)``: ``"candidates"`` (``filtered_coord.npy``, 564
    stations, the sparse model's input) or ``"shakealert"`` (``shakealert_coords.npy``, 101 ShakeAlert
    stations). Downloaded from the pinned repository commit and checked by sha256."""
    if name not in WAVECASTNET_STATIONS:
        raise ValueError(f"Unknown station set {name!r}; expected one of {sorted(WAVECASTNET_STATIONS)}.")
    spec = WAVECASTNET_STATIONS[name]
    path = cached_download("wavecastnet", spec["filename"], [_RAW + spec["filename"]], spec["sha256"])
    return torch.as_tensor(np.load(path)).round().long()


def load_wavecastnet_checkpoint(model: WaveCastNet, checkpoint: Union[str, Path]) -> WaveCastNet:
    """Load an official state dict (``module.`` prefixes of DataParallel removed) with ``strict=True``."""
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    state = {key.replace("module.", "", 1) if key.startswith("module.") else key: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model


def wavecastnet_checkpoint_path(name: str = "dense") -> Path:
    if name not in WAVECASTNET_WEIGHTS:
        raise ValueError(f"Unknown WaveCastNet weights {name!r}; expected one of {sorted(WAVECASTNET_WEIGHTS)}.")
    spec = WAVECASTNET_WEIGHTS[name]
    return cached_download("wavecastnet", spec["filename"], [_DRIVE.format(file_id=spec["file_id"])], spec["sha256"])


class WaveCastNetLoss(nn.Module):
    """Training loss of WaveCastNet.

    ``variant="official"`` (default) is the ``Huber`` class of ``earthquake_train.py`` with its default
    ``delta=0.2``: ``mean(where(|e| <= delta, 0.5 e^2 / delta + 0.5 delta, |e|))``, which equals the
    standard Huber loss divided by ``delta`` plus ``delta / 2``. ``variant="paper"`` is equation (4) of
    the paper, the standard Huber loss ``0.5 e^2`` / ``delta (|e| - delta / 2)``.
    """

    def __init__(self, delta: float = 0.2, variant: str = "official"):
        super().__init__()
        if variant not in {"official", "paper"}:
            raise ValueError(f"variant must be 'official' or 'paper', got {variant!r}.")
        if delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}.")
        self.delta = float(delta)
        self.variant = variant

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"pred and target must have the same shape, got {tuple(pred.shape)} and {tuple(target.shape)}.")
        error = pred - target
        abs_error = torch.abs(error)
        if self.variant == "official":
            loss = torch.where(abs_error <= self.delta, 0.5 * error**2 / self.delta + 0.5 * self.delta, abs_error)
        else:
            loss = torch.where(abs_error <= self.delta, 0.5 * error**2, self.delta * (abs_error - 0.5 * self.delta))
        return torch.mean(loss)


class WavefieldMetrics:
    """ACC, RFNE and RMSE of the official validation (``Validation_pixel.py``): one value per
    (sample, channel) over time and space, averaged. See :mod:`pyhazards.metrics.wavefield`."""

    @staticmethod
    def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
        return float(wavefield_acc(pred.detach(), target.detach()).mean())

    @staticmethod
    def rfne(pred: torch.Tensor, target: torch.Tensor) -> float:
        return float(wavefield_rfne(pred.detach(), target.detach()).mean())

    @staticmethod
    def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
        return float(wavefield_rmse(pred.detach(), target.detach()).mean())

    @staticmethod
    def compute_all(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        return {
            "ACC": WavefieldMetrics.accuracy(pred, target),
            "RFNE": WavefieldMetrics.rfne(pred, target),
            "RMSE": WavefieldMetrics.rmse(pred, target),
        }


_OLD_ARGUMENTS = {"temporal_in", "temporal_out", "hidden_dim", "num_layers", "dropout"}


def wavecastnet_builder(
    task: str = "forecasting",
    variant: str = "dense",
    in_channels: int = 3,
    height: int = 344,
    width: int = 224,
    future_seq: int = 30,
    kernel_size: Union[int, Sequence[int]] = 3,
    padding: Optional[Union[int, Sequence[int]]] = None,
    dt: float = 1.0,
    activation: str = "tanh",
    num_kernels: Optional[int] = None,
    station_coords: Optional[Union[str, torch.Tensor, np.ndarray]] = None,
    mask_mode: bool = True,
    mask_ratio: float = DEFAULT_MASK_RATIO,
    pretrained: Optional[Union[bool, str, Path]] = None,
    **kwargs,
) -> WaveCastNet:
    """WaveCastNet for ``task="forecasting"`` (``"regression"`` is accepted as an older alias).

    ``variant="dense"`` (default, ``AEConvLEM_dense``) or ``"sparse"`` (``AEConvLEM_sparse``, with
    ``station_coords``: a ``(stations, 2)`` index array or the name of an official station set, default
    ``"candidates"``). ``pretrained``: ``True`` / ``"dense"`` loads the released ``best_lem_dense_.pt``
    (dense model at 344 x 224, three channels; downloaded from the authors' Google Drive and checked by
    sha256; no licence is stated for it), a path loads any official state dict.
    """
    kwargs.pop("name", None)
    old = sorted(set(kwargs) & _OLD_ARGUMENTS)
    if old:
        raise TypeError(
            f"WaveCastNet no longer takes {old}: the official model has two ConvLEM layers per side, latent width "
            "48 * in_channels, no dropout, any input length, and the horizon is future_seq."
        )
    if kwargs:
        raise TypeError(f"Unexpected WaveCastNet arguments: {sorted(kwargs)}.")
    if task.lower() not in {"forecasting", "regression"}:
        raise ValueError(f"WaveCastNet supports task='forecasting', got {task!r}.")
    common = dict(
        num_channels=in_channels,
        height=height,
        width=width,
        kernel_size=kernel_size,
        padding=padding,
        dt=dt,
        activation=activation,
        future_seq=future_seq,
        num_kernels=num_kernels,
    )
    if variant == "dense":
        model: WaveCastNet = WaveCastNet(**common)
    elif variant == "sparse":
        coords = station_coords if station_coords is not None else "candidates"
        if isinstance(coords, str):
            coords = wavecastnet_station_coords(coords)
        model = WaveCastNetSparse(station_coords=coords, mask_mode=mask_mode, mask_ratio=mask_ratio, **common)
    else:
        raise ValueError(f"WaveCastNet variant must be 'dense' or 'sparse', got {variant!r}.")
    if pretrained:
        name = "dense" if pretrained is True else str(pretrained)
        if name in WAVECASTNET_WEIGHTS:
            if variant != "dense" or (in_channels, height, width) != (3, 344, 224):
                raise ValueError("The released dense checkpoint needs variant='dense', 3 channels and a 344 x 224 grid.")
            path = wavecastnet_checkpoint_path(name)
        else:
            path = Path(name)
        load_wavecastnet_checkpoint(model, path)
    return model


__all__ = [
    "ConvLEMCell",
    "DEFAULT_MASK_RATIO",
    "DenseEmbedding",
    "Reconstruction",
    "SparseEmbedding",
    "WAVECASTNET_STATIONS",
    "WAVECASTNET_WEIGHTS",
    "WaveCastNet",
    "WaveCastNetLoss",
    "WaveCastNetSparse",
    "WavefieldMetrics",
    "load_wavecastnet_checkpoint",
    "wavecastnet_builder",
    "wavecastnet_checkpoint_path",
    "wavecastnet_station_coords",
]
