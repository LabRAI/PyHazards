from __future__ import annotations

import torch

from ..base import DataBundle, DataSplit, Dataset, FeatureSpec, LabelSpec


class SyntheticWildfireSpreadDataset(Dataset):
    """Synthetic raster dataset for wildfire spread smoke runs."""

    name = "wildfire_spread_synthetic"

    def __init__(
        self,
        cache_dir: str | None = None,
        samples: int = 64,
        channels: int = 12,
        height: int = 32,
        width: int = 32,
        micro: bool = False,
    ):
        super().__init__(cache_dir=cache_dir)
        self.samples = 16 if micro else int(samples)
        self.channels = int(channels)
        self.height = int(height)
        self.width = int(width)

    def _load(self) -> DataBundle:
        x = torch.randn(self.samples, self.channels, self.height, self.width, dtype=torch.float32)
        y = torch.zeros(self.samples, 1, self.height, self.width, dtype=torch.float32)
        rows = torch.arange(self.height).view(1, self.height, 1)
        cols = torch.arange(self.width).view(1, 1, self.width)

        for idx in range(self.samples):
            center_r = (idx * 3) % self.height
            center_c = (idx * 5) % self.width
            radius = 4 + (idx % 5)
            mask = ((rows - center_r).float().pow(2) + (cols - center_c).float().pow(2)) <= radius**2
            y[idx, 0] = mask.float()
            x[idx, 0] = x[idx, 0] + 2.5 * mask.float()

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
                description="Synthetic raster weather and fuel covariates for wildfire spread.",
            ),
            label_spec=LabelSpec(
                num_targets=1,
                task_type="segmentation",
                description="Binary spread mask for the next forecast horizon.",
            ),
            metadata={
                "dataset": self.name,
                "source_dataset": self.name,
                "hazard_task": "wildfire.spread",
            },
        )


class SyntheticWildfireSpreadTemporalDataset(Dataset):
    """Synthetic temporal wildfire spread dataset for sequence-based spread baselines."""

    name = "wildfire_spread_temporal_synthetic"

    def __init__(
        self,
        cache_dir: str | None = None,
        samples: int = 48,
        history: int = 4,
        channels: int = 6,
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
        rows = torch.arange(self.height).view(1, self.height, 1)
        cols = torch.arange(self.width).view(1, 1, self.width)

        for idx in range(self.samples):
            center_r = (idx * 2 + 3) % self.height
            center_c = (idx * 3 + 5) % self.width
            radius = 3 + (idx % 4)
            final_mask = (
                ((rows - center_r).float().pow(2) + (cols - center_c).float().pow(2))
                <= radius**2
            ).float()
            y[idx, 0] = final_mask
            for step in range(self.history):
                inner_radius = max(1, radius - (self.history - step - 1))
                history_mask = (
                    ((rows - center_r).float().pow(2) + (cols - center_c).float().pow(2))
                    <= inner_radius**2
                ).float()
                x[idx, step, 0] = x[idx, step, 0] + history_mask

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
                description="Synthetic temporal wildfire spread covariates over forecast history windows.",
                extra={"history": self.history},
            ),
            label_spec=LabelSpec(
                num_targets=1,
                task_type="segmentation",
                description="Binary spread mask for the next forecast horizon.",
            ),
            metadata={
                "dataset": self.name,
                "source_dataset": self.name,
                "hazard_task": "wildfire.spread",
            },
        )


__all__ = ["SyntheticWildfireSpreadDataset", "SyntheticWildfireSpreadTemporalDataset"]
