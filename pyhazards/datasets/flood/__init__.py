"""Flood datasets: real streamflow readers (CAMELS-US, Caravan) and synthetic smoke-test data.

Only ``camels_us_streamflow`` and ``caravan_streamflow`` read real data (from a local copy of the official
release). The ``*_synthetic`` datasets generate random numbers in the layout of a task so that models
and evaluators can be exercised without data; they carry no benchmark's name.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..base import DataBundle, DataSplit, Dataset, FeatureSpec, LabelSpec
from ..graph import GraphTemporalDataset
from .camels_us import (
    CAMELS_US_TARGET,
    KRATZERT2019_CHECKPOINT_STATIC_ORDER,
    KRATZERT2019_DYNAMIC_INPUTS,
    KRATZERT2019_PERIODS,
    KRATZERT2019_STATIC_ATTRIBUTES,
    CamelsUSStreamflowDataset,
    load_camels_us_attributes,
    load_camels_us_basin,
    load_camels_us_discharge,
    load_camels_us_forcings,
)
from .caravan import CaravanStreamflowDataset, load_caravan_attributes, load_caravan_timeseries
from .streamflow import (
    StreamflowScaler,
    StreamflowWindows,
    build_streamflow_bundle,
    read_basin_list,
    scaler_from_metadata,
)


class SyntheticFloodStreamflowDataset(Dataset):
    """Synthetic daily basins in the streamflow layout (random forcings, toy linear-reservoir discharge).

    Same structure as the real readers (``x_d`` windows, static attributes, per-basin dates, train /
    val / test periods, NeuralHydrology-style normalisation), so streamflow models and the flood
    benchmark run without data. The numbers mean nothing hydrologically.
    """

    name = "flood_streamflow_synthetic"

    def __init__(
        self,
        cache_dir: str | None = None,
        basins: int = 6,
        days: int = 730,
        n_dynamic: int = 5,
        n_static: int = 27,
        seq_length: int = 30,
        predict_last_n: int = 1,
        seed: int = 0,
        micro: bool = False,
    ):
        super().__init__(cache_dir=cache_dir)
        self.n_basins = 3 if micro else int(basins)
        self.days = 300 if micro else int(days)
        self.n_dynamic = int(n_dynamic)
        self.n_static = int(n_static)
        self.seq_length = int(seq_length)
        self.predict_last_n = int(predict_last_n)
        self.seed = int(seed)
        if self.n_basins < 2:
            raise ValueError("flood_streamflow_synthetic needs at least 2 basins (static attributes are standardised).")
        if self.n_dynamic < 1:
            raise ValueError("n_dynamic must be positive.")
        if self.days < 3 * self.seq_length:
            raise ValueError(f"days ({self.days}) must be at least 3 * seq_length ({3 * self.seq_length}).")

    def _load(self) -> DataBundle:
        rng = np.random.default_rng(self.seed)
        dates = pd.date_range("2000-01-01", periods=self.days, freq="1D")
        dynamic = [f"synthetic_forcing_{i}" for i in range(self.n_dynamic)]
        static = [f"synthetic_attribute_{i:02d}" for i in range(self.n_static)]
        target = "discharge"
        attributes = pd.DataFrame(
            rng.normal(size=(self.n_basins, self.n_static)),
            index=[f"synthetic_{i:03d}" for i in range(self.n_basins)],
            columns=static,
        )
        frames = {}
        season = np.sin(2 * np.pi * np.arange(self.days) / 365.25)
        for b, basin in enumerate(attributes.index):
            rain = rng.gamma(0.8, 6.0, size=self.days) * (rng.random(self.days) < 0.35)
            forcings = {dynamic[0]: rain}
            for i in range(1, self.n_dynamic):
                forcings[dynamic[i]] = 10 * season * (1 + 0.1 * i) + rng.normal(scale=2.0, size=self.days)
            k = 0.05 + 0.2 / (1 + np.exp(-attributes.iloc[b, 0])) if self.n_static else 0.1
            storage, discharge = 0.0, np.empty(self.days)
            for t in range(self.days):
                storage = storage * (1 - k) + rain[t]
                discharge[t] = k * storage
            discharge[rng.choice(self.days, size=max(1, self.days // 50), replace=False)] = np.nan
            frames[basin] = pd.DataFrame({**forcings, target: discharge}, index=pd.DatetimeIndex(dates, name="date"))
        train_end = dates[int(0.5 * self.days)]
        val_end = dates[int(0.7 * self.days)]
        periods = {
            "train": (dates[self.seq_length], train_end),
            "val": (train_end + pd.Timedelta(days=1), val_end),
            "test": (val_end + pd.Timedelta(days=1), dates[-1]),
        }
        return build_streamflow_bundle(
            frames,
            attributes if static else None,
            dynamic,
            static,
            [target],
            periods,
            self.seq_length,
            self.predict_last_n,
            dataset_name=self.name,
            metadata={"synthetic": True},
        )


class SyntheticFloodMeshDataset(Dataset):
    """Synthetic graph-temporal node series (water depth on mesh nodes) for mesh-flood smoke runs."""

    name = "flood_mesh_synthetic"

    def __init__(
        self,
        cache_dir: str | None = None,
        samples: int = 40,
        history: int = 4,
        nodes: int = 6,
        features: int = 2,
        micro: bool = False,
    ):
        super().__init__(cache_dir=cache_dir)
        self.samples = 12 if micro else int(samples)
        self.history = int(history)
        self.nodes = int(nodes)
        self.features = int(features)

    def _make_split(self, x: torch.Tensor, y: torch.Tensor, adj: torch.Tensor) -> DataSplit:
        dataset = GraphTemporalDataset(x, y, adjacency=adj)
        return DataSplit(inputs=dataset, targets=None)

    def _load(self) -> DataBundle:
        x = torch.randn(self.samples, self.history, self.nodes, self.features, dtype=torch.float32)
        adjacency = torch.eye(self.nodes, dtype=torch.float32)
        adjacency += torch.diag(torch.ones(self.nodes - 1), diagonal=1)
        adjacency += torch.diag(torch.ones(self.nodes - 1), diagonal=-1)
        y = x[:, -1, :, :1] * 0.7 + 0.1

        train_end = max(1, int(0.7 * self.samples))
        val_end = max(train_end + 1, int(0.85 * self.samples))
        splits = {
            "train": self._make_split(x[:train_end], y[:train_end], adjacency),
            "val": self._make_split(x[train_end:val_end], y[train_end:val_end], adjacency),
            "test": self._make_split(x[val_end:], y[val_end:], adjacency),
        }
        return DataBundle(
            splits=splits,
            feature_spec=FeatureSpec(
                input_dim=self.features,
                description="Synthetic node features on a line graph (random numbers).",
                extra={"nodes": self.nodes, "history": self.history},
            ),
            label_spec=LabelSpec(
                num_targets=1,
                task_type="regression",
                description="Synthetic next-step nodewise water depth.",
            ),
            metadata={
                "dataset": self.name,
                "source_dataset": self.name,
                "hazard_task": "flood.inundation",
                "synthetic": True,
            },
        )


class SyntheticFloodInundationDataset(Dataset):
    """Synthetic raster dataset for flood inundation smoke runs."""

    name = "flood_inundation_synthetic"

    def __init__(
        self,
        cache_dir: str | None = None,
        samples: int = 40,
        history: int = 4,
        channels: int = 3,
        height: int = 16,
        width: int = 16,
        micro: bool = False,
    ):
        super().__init__(cache_dir=cache_dir)
        self.samples = 12 if micro else int(samples)
        self.history = int(history)
        self.channels = int(channels)
        self.height = int(height)
        self.width = int(width)

    def _load(self) -> DataBundle:
        x = torch.randn(
            self.samples,
            self.history,
            self.channels,
            self.height,
            self.width,
            dtype=torch.float32,
        )
        y = torch.zeros(self.samples, 1, self.height, self.width, dtype=torch.float32)
        rows = torch.arange(self.height, dtype=torch.float32).view(self.height, 1)
        cols = torch.arange(self.width, dtype=torch.float32).view(1, self.width)

        for idx in range(self.samples):
            waterline = float(self.height // 3 + (idx % max(2, self.height // 3)))
            slope = 0.25 + 0.05 * (idx % 4)
            rain_band = rows >= (waterline - slope * cols)
            depth = rain_band.float() * (0.4 + 0.1 * (idx % 3))
            y[idx, 0] = depth
            x[idx, -1, 0] = x[idx, -1, 0] + depth
            x[idx, :, 1] = x[idx, :, 1] + torch.linspace(0.0, 1.0, self.history).view(self.history, 1, 1)

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
                description="Synthetic rainfall, terrain, and antecedent-state tensors for inundation forecasting.",
                extra={
                    "history": self.history,
                    "height": self.height,
                    "width": self.width,
                },
            ),
            label_spec=LabelSpec(
                num_targets=1,
                task_type="regression",
                description="Next-horizon inundation depth raster.",
            ),
            metadata={
                "dataset": self.name,
                "source_dataset": self.name,
                "hazard_task": "flood.inundation",
                "synthetic": True,
            },
        )


__all__ = [
    "CAMELS_US_TARGET",
    "CamelsUSStreamflowDataset",
    "CaravanStreamflowDataset",
    "KRATZERT2019_CHECKPOINT_STATIC_ORDER",
    "KRATZERT2019_DYNAMIC_INPUTS",
    "KRATZERT2019_PERIODS",
    "KRATZERT2019_STATIC_ATTRIBUTES",
    "StreamflowScaler",
    "StreamflowWindows",
    "SyntheticFloodInundationDataset",
    "SyntheticFloodMeshDataset",
    "SyntheticFloodStreamflowDataset",
    "build_streamflow_bundle",
    "load_camels_us_attributes",
    "load_camels_us_basin",
    "load_camels_us_discharge",
    "load_camels_us_forcings",
    "load_caravan_attributes",
    "load_caravan_timeseries",
    "read_basin_list",
    "scaler_from_metadata",
]
