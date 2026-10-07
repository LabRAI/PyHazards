from __future__ import annotations

import os
from typing import Optional, Sequence, Union

import torch

from ..base import DataBundle, DataSplit, Dataset, FeatureSpec, LabelSpec
from .reader import read_wrf_sfire_fire_grid


class WRFSFireSpreadDataset(Dataset):
    """Next-step fire-spread samples cut from the outputs of a WRF-SFIRE run.

    Each sample pairs the burned mask (``LFN < 0``) at output frame ``t`` plus the requested fire-grid
    ``features`` at ``t`` with the burned mask at frame ``t + horizon``. Splits are chronological.
    There is no synthetic fallback: ``paths`` must point at real ``wrfout`` files produced by the
    official WRF-SFIRE (https://github.com/openwfm/WRF-SFIRE), which PyHazards does not run.
    """

    name = "wrf_sfire_spread"

    def __init__(
        self,
        paths: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]], None] = None,
        cache_dir: Optional[str] = None,
        horizon: int = 1,
        features: Sequence[str] = (),
        origin: str = "upper",
        train_fraction: float = 0.7,
        val_fraction: float = 0.15,
    ):
        super().__init__(cache_dir=cache_dir)
        if paths is None:
            raise ValueError(
                "wrf_sfire_spread reads WRF-SFIRE wrfout files: pass paths='run/wrfout_d01_*' "
                "(PyHazards does not run WRF-SFIRE or ship its outputs)."
            )
        if not (0.0 < train_fraction < 1.0 and 0.0 <= val_fraction < 1.0 and train_fraction + val_fraction < 1.0):
            raise ValueError("need 0 < train_fraction, 0 <= val_fraction and train_fraction + val_fraction < 1")
        self.paths = paths
        self.horizon = int(horizon)
        self.features = tuple(features)
        self.origin = origin
        self.train_fraction = float(train_fraction)
        self.val_fraction = float(val_fraction)

    def _load(self) -> DataBundle:
        variables = ("LFN", "TIGN_G", *self.features)
        grid = read_wrf_sfire_fire_grid(self.paths, variables=variables, origin=self.origin)
        inputs, targets, times = grid.spread_pairs(horizon=self.horizon, features=self.features)
        x = torch.from_numpy(inputs)
        y = torch.from_numpy(targets)
        n = x.shape[0]
        train_end = max(1, int(round(self.train_fraction * n)))
        val_end = min(n, train_end + int(round(self.val_fraction * n)))
        splits = {
            "train": DataSplit(x[:train_end], y[:train_end], metadata={"times_s": times[:train_end]}),
            "val": DataSplit(x[train_end:val_end], y[train_end:val_end], metadata={"times_s": times[train_end:val_end]}),
            "test": DataSplit(x[val_end:], y[val_end:], metadata={"times_s": times[val_end:]}),
        }
        return DataBundle(
            splits=splits,
            feature_spec=FeatureSpec(
                channels=int(x.shape[1]),
                description="WRF-SFIRE burned mask (LFN < 0) at frame t, then the requested fire-grid fields.",
                extra={"channels": ["burned_mask", *self.features], "fire_dx_m": grid.fire_dx, "fire_dy_m": grid.fire_dy},
            ),
            label_spec=LabelSpec(
                num_targets=1,
                task_type="segmentation",
                description=f"WRF-SFIRE burned mask {self.horizon} output frame(s) later.",
            ),
            metadata={
                "dataset": self.name,
                "source_dataset": "WRF-SFIRE outputs",
                "hazard_task": "wildfire.spread",
                "files": grid.source_files,
                "horizon_frames": self.horizon,
                "origin": self.origin,
                "sr_x": grid.sr_x,
                "sr_y": grid.sr_y,
            },
        )


__all__ = ["WRFSFireSpreadDataset"]
