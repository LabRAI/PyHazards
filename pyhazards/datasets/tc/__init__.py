from __future__ import annotations

import torch

from ..base import DataBundle, DataSplit, Dataset, FeatureSpec, LabelSpec


class SyntheticTropicalCycloneDataset(Dataset):
    """Synthetic storm-history dataset for track/intensity smoke runs."""

    name = "tc_tracks_synthetic"

    def __init__(
        self,
        cache_dir: str | None = None,
        samples: int = 64,
        history: int = 6,
        horizon: int = 5,
        features: int = 8,
        micro: bool = False,
    ):
        super().__init__(cache_dir=cache_dir)
        self.samples = 20 if micro else int(samples)
        self.history = int(history)
        self.horizon = int(horizon)
        self.features = int(features)

    def _load(self) -> DataBundle:
        x = torch.randn(self.samples, self.history, self.features, dtype=torch.float32)
        last_state = x[:, -1, :3]
        deltas = torch.linspace(0.2, 1.0, steps=self.horizon, dtype=torch.float32).view(1, self.horizon, 1)
        direction = torch.tensor([0.4, 0.2, 1.5], dtype=torch.float32).view(1, 1, 3)
        y = last_state.unsqueeze(1) + deltas * direction

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
                input_dim=self.features,
                description="Synthetic storm history with environmental context features.",
                extra={"history": self.history, "horizon": self.horizon},
            ),
            label_spec=LabelSpec(
                num_targets=3,
                task_type="regression",
                description="Forecast track latitude/longitude and intensity trajectory.",
            ),
            metadata={
                "dataset": self.name,
                "source_dataset": self.name,
                "hazard_task": "tc.track_intensity",
            },
        )


class IBTrACSTropicalCycloneDataset(SyntheticTropicalCycloneDataset):
    """Synthetic-backed adapter for IBTrACS-style storm tracks."""

    name = "ibtracs_tracks"

    def _load(self) -> DataBundle:
        bundle = super()._load()
        bundle.metadata.update({"adapter": "IBTrACS", "source_dataset": self.name})
        return bundle


class TCBenchAlphaDataset(SyntheticTropicalCycloneDataset):
    """Synthetic-backed adapter for TCBench Alpha evaluation runs."""

    name = "tcbench_alpha"

    def _load(self) -> DataBundle:
        bundle = super()._load()
        bundle.metadata.update({"adapter": "TCBench Alpha", "source_dataset": self.name})
        return bundle


class TropiCycloneNetDataset(SyntheticTropicalCycloneDataset):
    """Synthetic-backed adapter for TropiCycloneNet-Dataset style smoke runs."""

    name = "tropicyclonenet_dataset"

    def _load(self) -> DataBundle:
        bundle = super()._load()
        bundle.metadata.update({"adapter": "TropiCycloneNet-Dataset", "source_dataset": self.name})
        return bundle


__all__ = [
    "IBTrACSTropicalCycloneDataset",
    "SyntheticTropicalCycloneDataset",
    "TCBenchAlphaDataset",
    "TropiCycloneNetDataset",
]
