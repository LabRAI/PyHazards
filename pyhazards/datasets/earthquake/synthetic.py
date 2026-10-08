"""Synthetic earthquake datasets for smoke runs. Neither reads real data."""

from __future__ import annotations

import math

import torch

from ..base import DataBundle, DataSplit, Dataset, FeatureSpec, LabelSpec


class SyntheticEarthquakeWaveformDataset(Dataset):
    """Synthetic three-component windows with P and S arrivals, for phase-picking smoke runs.

    Not real data. Each event window is Gaussian background noise plus two damped sinusoids that start
    at the P arrival (about 8 Hz, strongest on Z) and at the S arrival (about 4 Hz, strongest on N and
    E); noise windows have no arrival (targets NaN). Values are drawn from a seeded generator, so a
    given configuration always gives the same data.

    Targets are ``(n, 2)`` arrival samples ``[P, S]`` with NaN for "no arrival", the layout that the
    ``earthquake.picking`` benchmark scores. ``length`` defaults to 6000 samples at 100 Hz (60 s, the
    EQTransformer window; PhaseNet and GPD take any length >= 400).
    """

    name = "earthquake_waveforms_synthetic"
    component_order = "ZNE"

    def __init__(
        self,
        cache_dir: str | None = None,
        samples: int = 96,
        length: int = 6000,
        sampling_rate: float = 100.0,
        noise_fraction: float = 0.25,
        seed: int = 0,
        micro: bool = False,
    ):
        super().__init__(cache_dir=cache_dir)
        self.samples = 24 if micro else int(samples)
        self.length = int(length)
        self.sampling_rate = float(sampling_rate)
        self.noise_fraction = float(noise_fraction)
        self.seed = int(seed)
        if self.samples < 3:
            raise ValueError(f"samples must be at least 3 (one per split), got {self.samples}.")
        if self.length < 2 * int(self.sampling_rate):
            raise ValueError(f"length must cover at least 2 s ({2 * int(self.sampling_rate)} samples), got {self.length}.")
        if not 0.0 <= self.noise_fraction < 1.0:
            raise ValueError(f"noise_fraction must be in [0, 1), got {self.noise_fraction}.")

    def _wavelet(self, onset: float, frequency: float, decay_s: float, generator: torch.Generator) -> torch.Tensor:
        t = (torch.arange(self.length, dtype=torch.float64) - onset) / self.sampling_rate
        phase = float(torch.rand(1, generator=generator)) * 2.0 * math.pi
        envelope = torch.where(t >= 0, (1.0 - torch.exp(-t / 0.05)) * torch.exp(-t / decay_s), torch.zeros_like(t))
        return envelope * torch.sin(2.0 * math.pi * frequency * t.clamp_min(0.0) + phase)

    def _load(self) -> DataBundle:
        generator = torch.Generator().manual_seed(self.seed)
        rate = self.sampling_rate
        x = torch.randn(self.samples, 3, self.length, generator=generator, dtype=torch.float64)
        y = torch.full((self.samples, 2), float("nan"), dtype=torch.float32)
        n_noise = int(round(self.noise_fraction * self.samples))
        # Interleave noise windows so that every split gets some (deterministic positions).
        noise_rows = set(torch.linspace(0, self.samples - 1, n_noise).round().long().tolist()) if n_noise else set()
        for idx in range(self.samples):
            if idx in noise_rows:
                continue
            p_pick = float(torch.randint(int(0.1 * self.length), int(0.4 * self.length), (1,), generator=generator))
            gap = float(torch.empty(1).uniform_(1.0, 12.0, generator=generator)) * rate
            s_pick = min(p_pick + gap, self.length - 0.5 * rate)
            amplitude = float(torch.empty(1).uniform_(4.0, 20.0, generator=generator))
            p_wave = self._wavelet(p_pick, 8.0, 1.0, generator)
            s_wave = self._wavelet(s_pick, 4.0, 2.5, generator)
            for channel, (p_gain, s_gain) in enumerate(((1.0, 0.4), (0.4, 1.2), (0.4, 1.0))):  # Z, N, E
                x[idx, channel] += amplitude * (p_gain * p_wave + s_gain * s_wave)
            y[idx, 0] = p_pick
            y[idx, 1] = round(s_pick)
        x = x.float()

        train_end = max(1, int(0.7 * self.samples))
        val_end = max(train_end + 1, int(0.85 * self.samples))
        splits = {
            "train": DataSplit(x[:train_end], y[:train_end]),
            "val": DataSplit(x[train_end:val_end], y[train_end:val_end]),
            "test": DataSplit(x[val_end:], y[val_end:]),
        }
        return DataBundle(
            splits=splits,
            feature_spec=FeatureSpec(
                channels=3,
                description="Synthetic three-component waveforms (Z, N, E) with P and S wavelets; not real data.",
                extra={"length": self.length, "sampling_rate": rate, "component_order": self.component_order},
            ),
            label_spec=LabelSpec(
                num_targets=2,
                task_type="picking",
                description="P and S arrival samples (NaN = no arrival).",
            ),
            metadata={
                "dataset": self.name,
                "source_dataset": self.name,
                "hazard_task": "earthquake.picking",
                "synthetic": True,
                "sampling_rate": rate,
                "component_order": self.component_order,
            },
        )


class SyntheticEarthquakeWavefieldDataset(Dataset):
    """Synthetic ground-motion wavefields for wavefield-forecasting smoke runs (not real data).

    Each sequence is a point source at a random grid position and onset time radiating a P and an S
    wavefront (Ricker pulses travelling at ``vp`` and ``vs`` grid cells per frame, amplitudes decaying as
    ``1 / sqrt(1 + r)``) on a ``height x width`` grid. The three channels are the X, Y and Z particle
    velocities: P moves particles radially (and vertically, half amplitude), S transversally. This only
    mimics the layout of WaveCastNet's data (three velocity components on a regular grid, Lyu et al.
    2025); it is not an elastic simulation. Values come from a seeded generator.

    Inputs ``(n, 3, temporal_in, height, width)`` and targets ``(n, 3, temporal_out, height, width)``,
    the layout of the ``earthquake.forecasting`` task; ``height`` and ``width`` default to multiples of 8
    (WaveCastNet's latent grid is 1/8 of the input).
    """

    name = "earthquake_wavefield_synthetic"
    channel_names = ("x", "y", "z")

    def __init__(
        self,
        cache_dir: str | None = None,
        samples: int = 24,
        temporal_in: int = 6,
        temporal_out: int = 6,
        height: int = 32,
        width: int = 24,
        vp: float = 2.0,
        vs: float = 1.2,
        seed: int = 0,
        micro: bool = False,
    ):
        super().__init__(cache_dir=cache_dir)
        self.samples = 6 if micro else int(samples)
        self.temporal_in = int(temporal_in)
        self.temporal_out = int(temporal_out)
        self.height = int(height)
        self.width = int(width)
        self.vp = float(vp)
        self.vs = float(vs)
        self.seed = int(seed)
        if self.samples < 3:
            raise ValueError(f"samples must be at least 3 (one per split), got {self.samples}.")
        if min(self.temporal_in, self.temporal_out, self.height, self.width) < 1:
            raise ValueError("temporal_in, temporal_out, height and width must be positive.")
        if not self.vp > self.vs > 0:
            raise ValueError(f"Need vp > vs > 0, got vp={self.vp}, vs={self.vs}.")

    @staticmethod
    def _ricker(t: torch.Tensor, width: float) -> torch.Tensor:
        a = (t / width) ** 2
        return (1.0 - 2.0 * a) * torch.exp(-a)

    def _load(self) -> DataBundle:
        generator = torch.Generator().manual_seed(self.seed)
        steps = self.temporal_in + self.temporal_out
        rows = torch.arange(self.height, dtype=torch.float64).view(self.height, 1)
        cols = torch.arange(self.width, dtype=torch.float64).view(1, self.width)
        frames = torch.arange(steps, dtype=torch.float64).view(steps, 1, 1)
        data = torch.zeros(self.samples, 3, steps, self.height, self.width, dtype=torch.float64)
        for idx in range(self.samples):
            u = torch.rand(4, generator=generator, dtype=torch.float64)
            row, col = u[0] * (self.height - 1), u[1] * (self.width - 1)
            onset = -2.0 + 3.0 * float(u[2])
            amplitude = 0.5 + 1.5 * float(u[3])
            dy, dx = rows - row, cols - col
            r = torch.sqrt(dx**2 + dy**2)
            radial_x, radial_y = dx / r.clamp_min(1e-6), dy / r.clamp_min(1e-6)
            spreading = amplitude / torch.sqrt(1.0 + r)
            p_wave = spreading * self._ricker(frames - onset - r / self.vp, 1.0)
            s_wave = 1.5 * spreading * self._ricker(frames - onset - r / self.vs, 1.5)
            data[idx, 0] = radial_x * p_wave - radial_y * s_wave
            data[idx, 1] = radial_y * p_wave + radial_x * s_wave
            data[idx, 2] = 0.5 * p_wave
        data = data.float()
        x, y = data[:, :, : self.temporal_in], data[:, :, self.temporal_in :]

        train_end = max(1, int(0.7 * self.samples))
        val_end = max(train_end + 1, int(0.85 * self.samples))
        splits = {
            "train": DataSplit(x[:train_end], y[:train_end]),
            "val": DataSplit(x[train_end:val_end], y[train_end:val_end]),
            "test": DataSplit(x[val_end:], y[val_end:]),
        }
        return DataBundle(
            splits=splits,
            feature_spec=FeatureSpec(
                channels=3,
                description="Synthetic X, Y, Z velocity wavefields of point sources (P and S Ricker wavefronts); not real data.",
                extra={
                    "temporal_in": self.temporal_in,
                    "temporal_out": self.temporal_out,
                    "height": self.height,
                    "width": self.width,
                    "channel_names": list(self.channel_names),
                },
            ),
            label_spec=LabelSpec(
                num_targets=3 * self.temporal_out,
                task_type="regression",
                description="Future wavefield frames (n, 3, temporal_out, height, width).",
            ),
            metadata={
                "dataset": self.name,
                "source_dataset": self.name,
                "hazard_task": "earthquake.forecasting",
                "synthetic": True,
                "channel_names": list(self.channel_names),
            },
        )


# Deprecated name of the synthetic wavefield generator (kept for existing imports).
SyntheticEarthquakeForecastDataset = SyntheticEarthquakeWavefieldDataset


__all__ = [
    "SyntheticEarthquakeForecastDataset",
    "SyntheticEarthquakeWaveformDataset",
    "SyntheticEarthquakeWavefieldDataset",
]
