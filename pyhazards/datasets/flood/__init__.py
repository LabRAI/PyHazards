from __future__ import annotations

import torch

from ..base import DataBundle, DataSplit, Dataset, FeatureSpec, LabelSpec
from ..graph import GraphTemporalDataset


class SyntheticFloodStreamflowDataset(Dataset):
    """Synthetic graph-temporal flood dataset for streamflow smoke runs."""

    name = "flood_streamflow_synthetic"

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
                description="Synthetic node features for streamflow forecasting on a line graph.",
                extra={"nodes": self.nodes, "history": self.history},
            ),
            label_spec=LabelSpec(
                num_targets=1,
                task_type="regression",
                description="Next-step nodewise streamflow target.",
            ),
            metadata={"hazard_task": "flood.streamflow"},
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
            metadata={"hazard_task": "flood.inundation"},
        )


__all__ = ["SyntheticFloodInundationDataset", "SyntheticFloodStreamflowDataset"]
