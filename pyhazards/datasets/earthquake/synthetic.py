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


class SyntheticEarthquakeForecastDataset(Dataset):
    """Synthetic dense-grid wavefield sequences for earthquake forecasting smoke runs (not real data)."""

    name = "earthquake_forecast_synthetic"

    def __init__(
        self,
        cache_dir: str | None = None,
        samples: int = 40,
        channels: int = 3,
        temporal_in: int = 5,
        temporal_out: int = 4,
        height: int = 12,
        width: int = 10,
        micro: bool = False,
    ):
        super().__init__(cache_dir=cache_dir)
        self.samples = 10 if micro else int(samples)
        self.channels = int(channels)
        self.temporal_in = int(temporal_in)
        self.temporal_out = int(temporal_out)
        self.height = int(height)
        self.width = int(width)

    def _load(self) -> DataBundle:
        grid_y = torch.linspace(-1.0, 1.0, steps=self.height, dtype=torch.float32).view(self.height, 1)
        grid_x = torch.linspace(-1.0, 1.0, steps=self.width, dtype=torch.float32).view(1, self.width)
        total_steps = self.temporal_in + self.temporal_out

        x = torch.zeros(
            self.samples,
            self.channels,
            self.temporal_in,
            self.height,
            self.width,
            dtype=torch.float32,
        )
        y = torch.zeros(
            self.samples,
            self.channels,
            self.temporal_out,
            self.height,
            self.width,
            dtype=torch.float32,
        )

        row_index = torch.arange(self.height, dtype=torch.float32).view(self.height, 1)
        col_index = torch.arange(self.width, dtype=torch.float32).view(1, self.width)

        for idx in range(self.samples):
            sequence = torch.zeros(
                self.channels,
                total_steps,
                self.height,
                self.width,
                dtype=torch.float32,
            )
            for step in range(total_steps):
                center_r = 2.0 + ((idx + step) % max(3, self.height - 2))
                center_c = 1.0 + ((2 * idx + step) % max(2, self.width - 1))
                gaussian = torch.exp(
                    -0.18 * ((row_index - center_r) ** 2 + (col_index - center_c) ** 2)
                )
                for channel in range(self.channels):
                    phase = 0.5 * channel + 0.2 * step
                    base = torch.sin(
                        math.pi * (channel + 1) * grid_y + phase
                    ) + torch.cos(math.pi * (channel + 1) * grid_x - phase)
                    sequence[channel, step] = base + (0.6 + 0.1 * channel) * gaussian

            x[idx] = sequence[:, : self.temporal_in]
            y[idx] = sequence[:, self.temporal_in :]

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
                channels=self.channels,
                description="Synthetic dense-grid wavefield history tensors for forecasting benchmarks.",
                extra={
                    "temporal_in": self.temporal_in,
                    "temporal_out": self.temporal_out,
                    "height": self.height,
                    "width": self.width,
                },
            ),
            label_spec=LabelSpec(
                num_targets=self.channels * self.temporal_out,
                task_type="regression",
                description="Future dense-grid wavefield frames over the forecast horizon.",
            ),
            metadata={
                "dataset": self.name,
                "source_dataset": self.name,
                "hazard_task": "earthquake.forecasting",
            },
        )


__all__ = [
    "SyntheticEarthquakeForecastDataset",
    "SyntheticEarthquakeWaveformDataset",
]
