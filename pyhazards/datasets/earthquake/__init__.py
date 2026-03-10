from __future__ import annotations

import math

import torch

from ..base import DataBundle, DataSplit, Dataset, FeatureSpec, LabelSpec


class SyntheticEarthquakeWaveformDataset(Dataset):
    """Synthetic waveform dataset for earthquake phase-picking smoke runs."""

    name = "earthquake_waveforms"

    def __init__(
        self,
        cache_dir: str | None = None,
        samples: int = 96,
        channels: int = 3,
        length: int = 256,
        micro: bool = False,
    ):
        super().__init__(cache_dir=cache_dir)
        self.samples = 24 if micro else int(samples)
        self.channels = int(channels)
        self.length = int(length)

    def _load(self) -> DataBundle:
        timeline = torch.linspace(0.0, 1.0, steps=self.length, dtype=torch.float32)
        x = torch.zeros(self.samples, self.channels, self.length, dtype=torch.float32)
        y = torch.zeros(self.samples, 2, dtype=torch.float32)

        for idx in range(self.samples):
            p_pick = 32 + (idx % 40)
            s_pick = min(self.length - 12, p_pick + 24 + (idx % 24))

            for channel in range(self.channels):
                phase = 0.5 * channel
                base = torch.sin(2.0 * math.pi * (channel + 1) * timeline + phase)
                pulse_p = torch.exp(-0.5 * ((torch.arange(self.length) - p_pick) / 6.0) ** 2)
                pulse_s = 0.8 * torch.exp(-0.5 * ((torch.arange(self.length) - s_pick) / 8.0) ** 2)
                x[idx, channel] = base + pulse_p + pulse_s

            y[idx, 0] = float(p_pick)
            y[idx, 1] = float(s_pick)

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
                description="Synthetic multichannel seismic waveforms with Gaussian phase arrivals.",
                extra={"length": self.length},
            ),
            label_spec=LabelSpec(
                num_targets=2,
                task_type="regression",
                description="P- and S-arrival sample indices.",
            ),
            metadata={
                "dataset": self.name,
                "source_dataset": self.name,
                "hazard_task": "earthquake.picking",
            },
        )


class SyntheticEarthquakeForecastDataset(Dataset):
    """Synthetic wavefield dataset for earthquake forecasting smoke runs."""

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


class SeisBenchWaveformDataset(SyntheticEarthquakeWaveformDataset):
    """Synthetic-backed adapter with the SeisBench public dataset surface."""

    name = "seisbench_waveforms"

    def _load(self) -> DataBundle:
        bundle = super()._load()
        bundle.metadata.update({"adapter": "SeisBench", "source_dataset": self.name})
        return bundle


class PickBenchmarkWaveformDataset(SyntheticEarthquakeWaveformDataset):
    """Synthetic-backed adapter with the pick-benchmark public dataset surface."""

    name = "pick_benchmark_waveforms"

    def _load(self) -> DataBundle:
        bundle = super()._load()
        bundle.metadata.update({"adapter": "pick-benchmark", "source_dataset": self.name})
        return bundle


class AEFADataset(SyntheticEarthquakeForecastDataset):
    """Synthetic-backed adapter for AEFA-style earthquake forecasting inputs."""

    name = "aefa_forecast"

    def _load(self) -> DataBundle:
        bundle = super()._load()
        bundle.metadata.update({"adapter": "AEFA", "source_dataset": self.name})
        return bundle


__all__ = [
    "AEFADataset",
    "PickBenchmarkWaveformDataset",
    "SeisBenchWaveformDataset",
    "SyntheticEarthquakeForecastDataset",
    "SyntheticEarthquakeWaveformDataset",
]
