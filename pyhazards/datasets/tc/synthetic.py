"""Synthetic tropical cyclone datasets for smoke tests.

Every dataset here generates random numbers in the input layout of a model; none of them contains
or imitates real storms, and scores on them mean nothing. Real data: ``ibtracs_tracks`` (IBTrACS
best tracks), ``ships_xu2021`` (SHIPS predictors of Xu et al. 2021), ``tropicyclonenet_dataset``
(TropiCycloneNet Dataset) and ``hurricast_ibtracs_era5`` (IBTrACS statistics and ERA5 maps).
"""

from __future__ import annotations

from typing import Dict

import torch

from ...models.hurricast import HURRICAST_STAT_FEATURES
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


class SyntheticHurricastDataset(Dataset):
    """Random inputs in the ``hurricast_ibtracs_era5`` layout.

    ``x_stat`` ``(samples, 8, 30)``: standard-normal numerical features, cyclic encodings in [-1, 1],
    a category value and one-hot basin / nature columns at the positions of
    :data:`pyhazards.models.hurricast.HURRICAST_STAT_FEATURES`; ``x_viz`` ``(samples, 8, 9, 25, 25)``
    standard normal; ``position`` random latitude / longitude. Targets: random 24-hour winds (kt) or
    positions ``(samples, 1, 2)`` (``target="displacement"``).
    """

    name = "hurricast_synthetic"

    def __init__(self, cache_dir: str | None = None, samples: int = 24, micro: bool = False, target: str = "intensity", window_size: int = 8, seed: int = 0):
        super().__init__(cache_dir=cache_dir)
        if target not in ("intensity", "displacement"):
            raise ValueError("target must be 'intensity' or 'displacement'")
        self.samples = 8 if micro else int(samples)
        self.target = target
        self.window_size = int(window_size)
        self.seed = int(seed)

    def _load(self) -> DataBundle:
        g = torch.Generator().manual_seed(self.seed)
        n, t = self.samples, self.window_size
        names = list(HURRICAST_STAT_FEATURES)
        x_stat = torch.randn(n, t, len(names), generator=g)
        for j, name in enumerate(names):
            if name.startswith(("COS_", "SIN_", "cat_cos", "cat_sign")):
                x_stat[..., j] = torch.rand(n, t, generator=g) * 2 - 1
            elif name == "cat_storm_category":
                x_stat[..., j] = torch.randint(0, 7, (n, 1), generator=g).float().expand(n, t)
        for prefix in ("cat_basin_", "cat_nature_"):
            columns = [j for j, name in enumerate(names) if name.startswith(prefix)]
            pick = torch.randint(0, len(columns), (n,), generator=g)
            x_stat[..., columns] = torch.nn.functional.one_hot(pick, len(columns)).float().unsqueeze(1).expand(n, t, len(columns))
        x_viz = torch.randn(n, t, 9, 25, 25, generator=g)
        position = torch.stack([torch.rand(n, generator=g) * 30 + 10, torch.rand(n, generator=g) * 60 - 100], dim=1)
        if self.target == "intensity":
            y = 30 + 100 * torch.rand(n, generator=g)
        else:
            y = (position + torch.randn(n, 2, generator=g)).unsqueeze(1)
        years = [2016 + (i % 4) for i in range(n)]
        train_end, val_end = _split_bounds(n)
        splits = {}
        for name, (a, b) in {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, n)}.items():
            splits[name] = DataSplit({"x_stat": x_stat[a:b], "x_viz": x_viz[a:b], "position": position[a:b]}, y[a:b], metadata={"groups": years[a:b]})
        metadata = {"dataset": self.name, "source_dataset": self.name, "lead_hours": [24], "window_size": t, "synthetic": True}
        if self.target == "intensity":
            metadata.update({"hazard_task": "tc.intensity", "units": {"wind": "kt"}})
        else:
            metadata.update({"hazard_task": "tc.track_intensity", "target_variables": ["lat", "lon"], "units": {}})
        return DataBundle(
            splits=splits,
            feature_spec=FeatureSpec(input_dim=len(names), description="Synthetic Hurricast statistics (8 x 30) and ERA5-like maps (8 x 9 x 25 x 25)."),
            label_spec=LabelSpec(num_targets=1 if self.target == "intensity" else 2, task_type="regression", description="Synthetic 24-hour intensity (kt) or position."),
            metadata=metadata,
        )


class SyntheticTCIFFusionDataset(Dataset):
    """Random inputs in the TCIF-fusion layout (channels last): ``u``, ``v``, ``w``
    ``(samples, grid, grid, times, levels)``, ``sst`` ``(samples, grid, grid, times, 1)``, ``all``
    ``(samples, grid, grid, 3 * times * levels + times)``, ``his`` ``(samples, 30)``, ``ir``
    ``(samples, ir_size, ir_size, 5)``; targets uniform 10-70 m/s. The defaults are the paper's sizes;
    smoke configs use smaller grids together with a small model.
    """

    name = "tcif_fusion_synthetic"

    def __init__(self, cache_dir: str | None = None, samples: int = 8, micro: bool = False, grid_size: int = 25, time_steps: int = 5, levels: int = 4, his_dim: int = 30, ir_size: int = 224, ir_channels: int = 5, seed: int = 0):
        super().__init__(cache_dir=cache_dir)
        self.samples = 4 if micro else int(samples)
        self.grid_size, self.time_steps, self.levels = int(grid_size), int(time_steps), int(levels)
        self.his_dim, self.ir_size, self.ir_channels = int(his_dim), int(ir_size), int(ir_channels)
        self.seed = int(seed)

    def _load(self) -> DataBundle:
        g = torch.Generator().manual_seed(self.seed)
        n, s, t, z = self.samples, self.grid_size, self.time_steps, self.levels
        inputs = {
            "u": torch.randn(n, s, s, t, z, generator=g),
            "v": torch.randn(n, s, s, t, z, generator=g),
            "w": torch.randn(n, s, s, t, z, generator=g),
            "sst": torch.rand(n, s, s, t, 1, generator=g),
            "his": torch.rand(n, self.his_dim, generator=g),
            "ir": torch.rand(n, self.ir_size, self.ir_size, self.ir_channels, generator=g),
        }
        inputs["all"] = torch.cat([inputs[k].reshape(n, s, s, -1) for k in ("u", "v", "w", "sst")], dim=-1)
        y = 10 + 60 * torch.rand(n, generator=g)
        years = [2020 + (i % 2) for i in range(n)]
        train_end, val_end = _split_bounds(n)
        splits = {
            name: DataSplit({k: v[a:b] for k, v in inputs.items()}, y[a:b], metadata={"groups": years[a:b]})
            for name, (a, b) in {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, n)}.items()
        }
        return DataBundle(
            splits=splits,
            feature_spec=FeatureSpec(description="Synthetic TCIF-fusion inputs (ERA5 U, V, W, SST, ALL, history, IR)."),
            label_spec=LabelSpec(num_targets=1, task_type="regression", description="Synthetic 24-hour maximum sustained wind (m/s)."),
            metadata={
                "dataset": self.name,
                "source_dataset": self.name,
                "hazard_task": "tc.intensity",
                "lead_hours": [24],
                "units": {"wind": "m/s"},
                "synthetic": True,
            },
        )

__all__ = [
    "SyntheticHurricastDataset",
    "SyntheticSAFNetDataset",
    "SyntheticSHIPSDataset",
    "SyntheticTCIFFusionDataset",
    "SyntheticTCNDDataset",
    "SyntheticTropicalCycloneDataset",
    "synthetic_tcnd_batch",
]
