"""Synthetic tropical cyclone datasets for smoke tests.

Every dataset here generates random numbers in the input layout of a model; none of them contains
or imitates real storms, and scores on them mean nothing. Real data: ``ibtracs_tracks`` (IBTrACS
best tracks), ``ships_xu2021`` (SHIPS predictors of Xu et al. 2021) and ``tropicyclonenet_dataset``
(TropiCycloneNet Dataset).
"""

from __future__ import annotations

from typing import Dict

import torch

from ...models.tropicalcyclone_mlp import SHIPS_PREDICTORS
from ...models.tropicyclonenet import ENV_FEATURES, TCND_NORMALIZATION
from ..base import DataBundle, DataSplit, Dataset, FeatureSpec, LabelSpec


def _split_bounds(samples: int):
    train_end = max(1, int(0.7 * samples))
    val_end = max(train_end + 1, int(0.85 * samples))
    return train_end, val_end


def _slice(value, start, end, batch_dim: int = 0):
    if isinstance(value, dict):
        return {key: _slice(item, start, end, batch_dim) for key, item in value.items()}
    index = [slice(None)] * value.ndim
    index[batch_dim] = slice(start, end)
    return value[tuple(index)]


class SyntheticTropicalCycloneDataset(Dataset):
    """Random "storm histories" ``(samples, history, features)`` and targets ``(samples, horizon, 3)``.

    The targets are the last input step's first three features plus a fixed linear drift; they
    are labelled ``lat``, ``lon``, ``wind`` only so that the cyclone evaluator can run. Kept for
    smoke tests of the experimental generic storm adapters.
    """

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
        if self.features < 3:
            raise ValueError("features must be at least 3")

    def _load(self) -> DataBundle:
        x = torch.randn(self.samples, self.history, self.features, dtype=torch.float32)
        last_state = x[:, -1, :3]
        deltas = torch.linspace(0.2, 1.0, steps=self.horizon, dtype=torch.float32).view(1, self.horizon, 1)
        direction = torch.tensor([0.4, 0.2, 1.5], dtype=torch.float32).view(1, 1, 3)
        y = last_state.unsqueeze(1) + deltas * direction
        train_end, val_end = _split_bounds(self.samples)
        splits = {
            "train": DataSplit(x[:train_end], y[:train_end]),
            "val": DataSplit(x[train_end:val_end], y[train_end:val_end]),
            "test": DataSplit(x[val_end:], y[val_end:]),
        }
        return DataBundle(
            splits=splits,
            feature_spec=FeatureSpec(
                input_dim=self.features,
                description="Synthetic random storm-history features (no real storms).",
                extra={"history": self.history, "horizon": self.horizon},
            ),
            label_spec=LabelSpec(num_targets=3, task_type="regression", description="Synthetic lat / lon / intensity targets."),
            metadata={
                "dataset": self.name,
                "source_dataset": self.name,
                "hazard_task": "tc.track_intensity",
                "lead_hours": [6 * (step + 1) for step in range(self.horizon)],
                "target_variables": ["lat", "lon", "wind"],
                "units": {"wind": "synthetic"},
                "synthetic": True,
            },
        )


class SyntheticSHIPSDataset(Dataset):
    """Random inputs in the layout of ``ships_xu2021``: 121 predictors -> 24-hour intensity change (kt)."""

    name = "ships_xu2021_synthetic"

    def __init__(self, cache_dir: str | None = None, samples: int = 64, micro: bool = False):
        super().__init__(cache_dir=cache_dir)
        self.samples = 24 if micro else int(samples)

    def _load(self) -> DataBundle:
        x = torch.randn(self.samples, len(SHIPS_PREDICTORS))
        y = (5.0 * x[:, 0] - 3.0 * x[:, -1] + 2.0 * torch.randn(self.samples)).round()
        years = [2017 + (i % 2) for i in range(self.samples)]
        train_end, val_end = _split_bounds(self.samples)
        splits = {
            name: DataSplit(x[a:b], y[a:b], metadata={"groups": years[a:b]})
            for name, (a, b) in {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, self.samples)}.items()
        }
        return DataBundle(
            splits=splits,
            feature_spec=FeatureSpec(input_dim=len(SHIPS_PREDICTORS), description="Synthetic standard-normal SHIPS-like predictors."),
            label_spec=LabelSpec(num_targets=1, task_type="regression", description="Synthetic 24-hour intensity change (kt)."),
            metadata={
                "dataset": self.name,
                "source_dataset": self.name,
                "hazard_task": "tc.intensity",
                "lead_hours": [24],
                "units": {"wind": "kt"},
                "synthetic": True,
            },
        )


class SyntheticSAFNetDataset(Dataset):
    """Random inputs in the SAF-Net layout: 96 wide predictors and ``(2, 4, 31, 31, 4)`` u/v fields.

    Inputs are uniform in [0, 1] like the MinMax-scaled official inputs; targets are uniform
    "24-hour intensities" in m/s, and ``prediction_transform`` maps the model's scaled output back
    to m/s with the synthetic target range.
    """

    name = "safnet_cma_era_interim_synthetic"
    wind_range = (10.0, 70.0)

    def __init__(self, cache_dir: str | None = None, samples: int = 32, micro: bool = False):
        super().__init__(cache_dir=cache_dir)
        self.samples = 8 if micro else int(samples)

    def _load(self) -> DataBundle:
        wide = torch.rand(self.samples, 96)
        deep = torch.rand(self.samples, 2, 4, 31, 31, 4)
        low, high = self.wind_range
        y = low + (high - low) * torch.rand(self.samples)
        years = [2015 + (i % 4) for i in range(self.samples)]
        train_end, val_end = _split_bounds(self.samples)
        splits = {}
        for name, (a, b) in {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, self.samples)}.items():
            splits[name] = DataSplit({"wide": wide[a:b], "deep": deep[a:b]}, y[a:b], metadata={"groups": years[a:b]})
        return DataBundle(
            splits=splits,
            feature_spec=FeatureSpec(
                input_dim=96,
                description="Synthetic MinMax-scaled wide predictors and u/v wind cubes (batch, 2, level, 31, 31, time).",
            ),
            label_spec=LabelSpec(num_targets=1, task_type="regression", description="Synthetic 24-hour maximum sustained wind (m/s)."),
            metadata={
                "dataset": self.name,
                "source_dataset": self.name,
                "hazard_task": "tc.intensity",
                "lead_hours": [24],
                "units": {"wind": "m/s"},
                "prediction_transform": {"kind": "minmax", "min": low, "max": high},
                "synthetic": True,
            },
        )


def _one_hot(index: torch.Tensor, width: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(index.long() % width, width).float()


def synthetic_tcnd_batch(samples: int, obs_len: int = 8, pred_len: int = 4, generator: torch.Generator | None = None) -> Dict[str, object]:
    """Random TropiCycloneNet inputs and physical targets (lat, lon, pres, wind)."""
    g = generator
    start = torch.stack(
        [torch.rand(samples, generator=g) * 2 - 1, torch.rand(samples, generator=g) * 4 - 2, torch.rand(samples, generator=g), torch.rand(samples, generator=g) - 0.5],
        dim=-1,
    )
    steps = 0.05 * torch.randn(obs_len + pred_len, samples, 4, generator=g)
    steps[0] = 0
    track = start.unsqueeze(0) + torch.cumsum(steps, dim=0)
    obs = track[:obs_len]
    rel = torch.zeros_like(obs)
    rel[1:] = obs[1:] - obs[:-1]
    future = track[obs_len:]
    physical = torch.stack(
        [future[..., 1] * TCND_NORMALIZATION["lat"][0] + TCND_NORMALIZATION["lat"][1],
         future[..., 0] * TCND_NORMALIZATION["lon"][0] + TCND_NORMALIZATION["lon"][1],
         future[..., 2] * TCND_NORMALIZATION["pres"][0] + TCND_NORMALIZATION["pres"][1],
         future[..., 3] * TCND_NORMALIZATION["wind"][0] + TCND_NORMALIZATION["wind"][1]],
        dim=-1,
    ).permute(1, 0, 2)
    env = {}
    for key, width in ENV_FEATURES:
        if width == 1:
            env[key] = torch.rand(samples, obs_len, 1, generator=g)
        else:
            env[key] = _one_hot(torch.randint(0, width, (samples, obs_len), generator=g), width)
    inputs = {
        "obs_traj": obs,
        "obs_traj_rel": rel,
        "image_obs": torch.rand(samples, 1, obs_len, 64, 64, generator=g),
        "env_data": env,
    }
    return {"inputs": inputs, "targets": physical}


class SyntheticTCNDDataset(Dataset):
    """Random inputs in the TropiCycloneNet (TCND) layout with random-walk targets.

    Inputs are the official generator's: ``obs_traj`` / ``obs_traj_rel`` ``(8, samples, 4)``
    (normalised lon, lat, pressure, wind and their steps), ``image_obs`` ``(samples, 1, 8, 64, 64)``
    and ``env_data`` (nine one-hot or scalar features per step). Targets ``(samples, 4, 4)`` hold
    latitude, longitude, pressure (hPa) and wind (m/s) at 6, 12, 18 and 24 h.
    """

    name = "tropicyclonenet_dataset_synthetic"

    def __init__(self, cache_dir: str | None = None, samples: int = 32, micro: bool = False, seed: int = 0):
        super().__init__(cache_dir=cache_dir)
        self.samples = 6 if micro else int(samples)
        self.seed = int(seed)

    def _load(self) -> DataBundle:
        generator = torch.Generator().manual_seed(self.seed)
        batch = synthetic_tcnd_batch(self.samples, generator=generator)
        inputs, targets = batch["inputs"], batch["targets"]
        train_end, val_end = _split_bounds(self.samples)
        splits = {}
        for name, (a, b) in {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, self.samples)}.items():
            split_inputs = {
                "obs_traj": inputs["obs_traj"][:, a:b],
                "obs_traj_rel": inputs["obs_traj_rel"][:, a:b],
                "image_obs": inputs["image_obs"][a:b],
                "env_data": _slice(inputs["env_data"], a, b),
            }
            splits[name] = DataSplit(split_inputs, targets[a:b])
        return DataBundle(
            splits=splits,
            feature_spec=FeatureSpec(description="Synthetic TCND-layout inputs (track, GPH frames, Env-Data)."),
            label_spec=LabelSpec(num_targets=4, task_type="regression", description="Synthetic lat, lon, pressure and wind at 6-24 h."),
            metadata={
                "dataset": self.name,
                "source_dataset": self.name,
                "hazard_task": "tc.track_intensity",
                "lead_hours": [6, 12, 18, 24],
                "target_variables": ["lat", "lon", "pres", "wind"],
                "units": {"wind": "m/s", "pres": "hPa"},
                "batch_dims": {"obs_traj": 1, "obs_traj_rel": 1},
                "synthetic": True,
            },
        )


__all__ = [
    "SyntheticSAFNetDataset",
    "SyntheticSHIPSDataset",
    "SyntheticTCNDDataset",
    "SyntheticTropicalCycloneDataset",
    "synthetic_tcnd_batch",
]
