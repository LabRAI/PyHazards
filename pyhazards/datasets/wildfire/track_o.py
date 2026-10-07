"""Track-O wildfire occurrence datasets: daily fire / no-fire grids with weather and fuel covariates.

Track-O ("occurrence") is the real-data wildfire setting of PyHazards PR #33 by runyangxu: on a daily
latitude-longitude grid, predict whether a cell has at least one satellite fire detection on day
``t`` from the weather of day ``t`` (and, for the temporal layout, of the days before it) and a
static fuel layer. The datasets here read a cache written by
:func:`pyhazards.datasets.wildfire.track_o_cache.build_track_o_cache` (CLI:
``scripts/build_wildfire_track_o_cache.py``)::

    <cache_dir>/
      metadata/lat.npy, lon.npy      cell-centre coordinates, both ascending (row 0 = southernmost)
      metadata/vars.json             {"weather_vars": [...]}
      met/<YYYY-MM-DD>.npy           (n_weather_vars, n_lat, n_lon) float32 daily means
      labels/<YYYY-MM-DD>.npy        (n_lat, n_lon) 1.0 where a fire detection fell in the cell
      static/fuel.npy, fuel_mask.npy optional fuel-model codes on the grid and their valid mask
      splits/{train,val,test}_dates.txt

Three layouts read the same cache:

- ``wildfire_track_o_raster``: one day per sample, ``(N, C, H, W)`` -> ``(N, 1, H, W)``;
- ``wildfire_track_o_temporal``: ``history`` consecutive days ending on the target day,
  ``(N, history, C, H, W)`` -> ``(N, 1, H, W)``; windows that would span a missing day are skipped;
- ``wildfire_track_o_tabular``: one row per cell and day, ``(N, F)`` -> ``(N,)`` int64 labels.

Weather channels are standardised with the mean and standard deviation of the training split
(NaN-aware; NaNs, e.g. MERRA-2 land variables over the ocean, become 0 after standardisation). The
fuel layer enters as two channels: the fuel-model code divided by 100 (a numeric stand-in for a
categorical code, kept from PR #33) and the valid-fuel mask. ``downsample_factor`` keeps every
``f``-th row and column of the native grid (a subset of native cells; nothing is averaged), and the
grid is then cropped to a multiple of ``spatial_multiple`` (dropping northern rows and eastern
columns) so that encoder-decoder models can pool it.

``micro=True`` replaces the cache with a small deterministic synthetic one (random fields, no
physical meaning) for smoke tests and examples.
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..base import DataBundle, DataSplit, Dataset, FeatureSpec, LabelSpec

SPLIT_NAMES = ("train", "val", "test")

# The 14 MERRA-2 surface variables PR #33 read from its Prithvi-WxC output files.
DEFAULT_WEATHER_VARS = (
    "T2M",
    "QV2M",
    "TQV",
    "U10M",
    "V10M",
    "GWETROOT",
    "TS",
    "LAI",
    "EFLUX",
    "HFLUX",
    "SWGNT",
    "SWTNT",
    "LWGAB",
    "LWGEM",
)

# PR #33's 2024 split: train January-September, validation October, test November-December.
DEFAULT_SPLITS_2024 = {
    "train": ("2024-01-01", "2024-09-30"),
    "val": ("2024-10-01", "2024-10-31"),
    "test": ("2024-11-01", "2024-12-31"),
}

STATIC_FEATURE_NAMES = ("fuel_code_div100", "fuel_valid_mask")


def _parse_date(text: str) -> date:
    return date.fromisoformat(str(text))


def _read_lines(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Track-O cache is missing the split file {path}.")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _limit(dates: Sequence[str], limit: Optional[int]) -> List[str]:
    if limit is None or int(limit) <= 0:
        return list(dates)
    return list(dates[: int(limit)])


class _CacheSource:
    """Reads a Track-O cache directory."""

    def __init__(self, root: Path):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Track-O cache directory not found: {self.root}. Build one with "
                "scripts/build_wildfire_track_o_cache.py, or pass micro=True for synthetic data."
            )
        vars_path = self.root / "metadata" / "vars.json"
        if not vars_path.exists():
            raise FileNotFoundError(f"{self.root} is not a Track-O cache: {vars_path} is missing.")
        self.weather_vars = list(json.loads(vars_path.read_text(encoding="utf-8"))["weather_vars"])
        self.lat = np.asarray(np.load(self.root / "metadata" / "lat.npy"), dtype=np.float64)
        self.lon = np.asarray(np.load(self.root / "metadata" / "lon.npy"), dtype=np.float64)

    def split_dates(self) -> Dict[str, List[str]]:
        return {name: _read_lines(self.root / "splits" / f"{name}_dates.txt") for name in SPLIT_NAMES}

    def met(self, day: str) -> np.ndarray:
        return np.load(self.root / "met" / f"{day}.npy")

    def label(self, day: str) -> np.ndarray:
        return np.load(self.root / "labels" / f"{day}.npy")

    def static(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        fuel_path = self.root / "static" / "fuel.npy"
        if not fuel_path.exists():
            return None, None
        fuel = np.load(fuel_path)
        mask_path = self.root / "static" / "fuel_mask.npy"
        mask = np.load(mask_path) if mask_path.exists() else (fuel > 0)
        return fuel, mask


class _MicroSource:
    """Deterministic synthetic stand-in for a cache (random fields; no physical meaning)."""

    weather_vars = ["T2M", "QV2M", "GWETROOT"]
    n_lat, n_lon = 64, 96
    n_days = {"train": 24, "val": 8, "test": 8}

    def __init__(self, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.lat = np.linspace(-79.0, 79.0, self.n_lat)
        self.lon = np.linspace(-180.0, 176.25, self.n_lon)
        start = date(2024, 1, 1)
        days = [str(start + timedelta(days=i)) for i in range(sum(self.n_days.values()))]
        train_end = self.n_days["train"]
        val_end = train_end + self.n_days["val"]
        self._dates = {"train": days[:train_end], "val": days[train_end:val_end], "test": days[val_end:]}
        rows = np.linspace(0.0, 1.0, self.n_lat)[:, None]
        cols = np.linspace(0.0, 1.0, self.n_lon)[None, :]
        self._met: Dict[str, np.ndarray] = {}
        self._labels: Dict[str, np.ndarray] = {}
        for index, day in enumerate(days):
            warm = np.sin(2 * np.pi * (rows + 0.05 * index)) * np.cos(2 * np.pi * cols) * 8.0 + 290.0
            fields = np.stack(
                [
                    warm + rng.normal(0.0, 1.0, (self.n_lat, self.n_lon)),
                    0.01 + 0.002 * rng.standard_normal((self.n_lat, self.n_lon)),
                    np.clip(0.5 + 0.2 * rng.standard_normal((self.n_lat, self.n_lon)), 0.0, 1.0),
                ]
            ).astype(np.float32)
            fields[2, :, :8] = np.nan  # a land-only variable undefined over an "ocean" strip
            logit = 0.8 * (fields[0] - 290.0) - 3.0
            self._labels[day] = (rng.random((self.n_lat, self.n_lon)) < 1.0 / (1.0 + np.exp(-logit))).astype(
                np.float32
            )
            self._met[day] = fields
        self._fuel = np.zeros((self.n_lat, self.n_lon), dtype=np.int16)
        self._fuel[16:48, 24:72] = rng.integers(1, 14, size=(32, 48))
        self._fuel_mask = (self._fuel > 0).astype(np.uint8)

    def split_dates(self) -> Dict[str, List[str]]:
        return {name: list(days) for name, days in self._dates.items()}

    def met(self, day: str) -> np.ndarray:
        return self._met[day]

    def label(self, day: str) -> np.ndarray:
        return self._labels[day]

    def static(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self._fuel, self._fuel_mask


class _TrackOBase(Dataset):
    """Shared loading logic of the three Track-O layouts."""

    name = "wildfire_track_o_base"
    default_downsample = 1

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        *,
        downsample_factor: Optional[int] = None,
        spatial_multiple: int = 4,
        train_limit_days: Optional[int] = None,
        val_limit_days: Optional[int] = None,
        test_limit_days: Optional[int] = None,
        micro: bool = False,
        seed: int = 7,
    ):
        super().__init__(cache_dir=cache_dir)
        if not micro and cache_dir is None:
            raise ValueError(
                f"{self.name} needs cache_dir (a cache built by scripts/build_wildfire_track_o_cache.py), "
                "or micro=True for the synthetic smoke-test data."
            )
        factor = self.default_downsample if downsample_factor is None else int(downsample_factor)
        if factor < 1:
            raise ValueError(f"downsample_factor must be >= 1, got {downsample_factor}")
        if int(spatial_multiple) < 1:
            raise ValueError(f"spatial_multiple must be >= 1, got {spatial_multiple}")
        self.downsample_factor = factor
        self.spatial_multiple = int(spatial_multiple)
        self.limits = {"train": train_limit_days, "val": val_limit_days, "test": test_limit_days}
        self.micro = bool(micro)
        self.seed = int(seed)

    # -- source and grid helpers -------------------------------------------------------------
    def _source(self):
        if self.micro:
            return _MicroSource(seed=self.seed)
        return _CacheSource(Path(self.cache_dir))

    def _split_dates(self, source) -> Dict[str, List[str]]:
        dates = source.split_dates()
        out = {name: _limit(dates[name], self.limits[name]) for name in SPLIT_NAMES}
        for name, days in out.items():
            if not days:
                raise ValueError(f"{self.name}: split '{name}' has no dates.")
        return out

    def _grid_shape(self, n_lat: int, n_lon: int) -> Tuple[int, int]:
        f, m = self.downsample_factor, self.spatial_multiple
        h, w = -(-n_lat // f), -(-n_lon // f)
        if h < m or w < m:
            raise ValueError(
                f"{self.name}: the {n_lat} x {n_lon} grid downsampled by {f} is {h} x {w}, smaller than "
                f"spatial_multiple={m}; lower downsample_factor or spatial_multiple."
            )
        return h - h % m, w - w % m

    def _resample(self, array: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        f = self.downsample_factor
        h, w = shape
        if array.ndim == 2:
            return np.asarray(array[::f, ::f][:h, :w], dtype=np.float32)
        if array.ndim == 3:
            return np.asarray(array[:, ::f, ::f][:, :h, :w], dtype=np.float32)
        raise ValueError(f"expected a 2-D or 3-D grid array, got shape {array.shape}")

    def _prepare(self):
        """Read the source once: per-split standardised weather, labels and the static block."""
        source = self._source()
        split_dates = self._split_dates(source)
        lat, lon = np.asarray(source.lat), np.asarray(source.lon)
        shape = self._grid_shape(lat.size, lon.size)
        weather_vars = list(source.weather_vars)

        raw = {
            name: {day: self._resample(source.met(day), shape) for day in days}
            for name, days in split_dates.items()
        }
        for name, arrays in raw.items():
            for day, array in arrays.items():
                if array.shape[0] != len(weather_vars):
                    raise ValueError(
                        f"{self.name}: met/{day}.npy has {array.shape[0]} channels, "
                        f"vars.json lists {len(weather_vars)}."
                    )

        train_stack = np.stack([raw["train"][day] for day in split_dates["train"]], axis=0).astype(np.float64)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(train_stack, axis=(0, 2, 3))
            std = np.nanstd(train_stack, axis=(0, 2, 3))
        mean = np.where(np.isfinite(mean), mean, 0.0)
        std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
        del train_stack

        met = {
            name: {
                day: np.nan_to_num(
                    (array - mean[:, None, None]) / std[:, None, None], nan=0.0, posinf=0.0, neginf=0.0
                ).astype(np.float32)
                for day, array in arrays.items()
            }
            for name, arrays in raw.items()
        }
        labels = {
            name: {day: (self._resample(source.label(day), shape) > 0.5).astype(np.float32) for day in days}
            for name, days in split_dates.items()
        }

        fuel, fuel_mask = source.static()
        static = None
        if fuel is not None:
            mask = self._resample(np.asarray(fuel_mask, dtype=np.float32), shape) > 0.5
            code = self._resample(np.asarray(fuel, dtype=np.float32), shape)
            static = np.stack([np.where(mask, code / 100.0, 0.0), mask.astype(np.float32)]).astype(np.float32)

        grid = {
            "lat": np.asarray(lat[:: self.downsample_factor][: shape[0]], dtype=np.float32),
            "lon": np.asarray(lon[:: self.downsample_factor][: shape[1]], dtype=np.float32),
        }
        normalization = {"mean": mean.tolist(), "std": std.tolist(), "fit_split": "train"}
        return split_dates, met, labels, static, weather_vars, grid, normalization

    def _metadata(self, weather_vars, grid, normalization, static, splits) -> Dict[str, object]:
        return {
            "dataset": self.name,
            "source_dataset": "micro_synthetic" if self.micro else "track_o_cache",
            "hazard_task": "wildfire.danger",
            "cache_root": None if self.micro else str(self.cache_dir),
            "micro": self.micro,
            "weather_vars": list(weather_vars),
            "static_feature_names": list(STATIC_FEATURE_NAMES) if static is not None else [],
            "has_static_fuel": static is not None,
            "downsample_factor": self.downsample_factor,
            "lat": grid["lat"].tolist(),
            "lon": grid["lon"].tolist(),
            "normalization": normalization,
            "splits": {name: int(split.inputs.shape[0]) for name, split in splits.items()},
        }


class WildfireTrackORasterDataset(_TrackOBase):
    """One day per sample: ``(N, C, H, W)`` covariates and ``(N, 1, H, W)`` fire-occurrence masks."""

    name = "wildfire_track_o_raster"
    default_downsample = 4

    def _load(self) -> DataBundle:
        split_dates, met, labels, static, weather_vars, grid, normalization = self._prepare()
        splits: Dict[str, DataSplit] = {}
        for name, days in split_dates.items():
            frames = [met[name][day] if static is None else np.concatenate([met[name][day], static]) for day in days]
            splits[name] = DataSplit(
                inputs=torch.from_numpy(np.stack(frames).astype(np.float32)),
                targets=torch.from_numpy(np.stack([labels[name][day][None] for day in days])),
                metadata={"dates": list(days)},
            )
        shape = splits["train"].inputs.shape
        return DataBundle(
            splits=splits,
            feature_spec=FeatureSpec(
                channels=int(shape[1]),
                description="Standardised daily-mean weather plus fuel channels on the Track-O grid.",
                extra={"height": int(shape[2]), "width": int(shape[3]), "weather_vars": weather_vars},
            ),
            label_spec=LabelSpec(
                num_targets=1,
                task_type="segmentation",
                description="1 where the cell has at least one fire detection on the same day, else 0.",
            ),
            metadata=self._metadata(weather_vars, grid, normalization, static, splits),
        )


class WildfireTrackOTemporalDataset(_TrackOBase):
    """``history`` consecutive days per sample: ``(N, history, C, H, W)`` -> ``(N, 1, H, W)``."""

    name = "wildfire_track_o_temporal"
    default_downsample = 8

    def __init__(self, cache_dir: Optional[str] = None, *, history: int = 6, **kwargs):
        super().__init__(cache_dir=cache_dir, **kwargs)
        if int(history) < 1:
            raise ValueError(f"history must be >= 1, got {history}")
        self.history = int(history)

    def _load(self) -> DataBundle:
        split_dates, met, labels, static, weather_vars, grid, normalization = self._prepare()
        splits: Dict[str, DataSplit] = {}
        for name, days in split_dates.items():
            available = set(days)
            frames: Dict[str, np.ndarray] = {
                day: met[name][day] if static is None else np.concatenate([met[name][day], static]) for day in days
            }
            inputs, targets, used = [], [], []
            for day in days:
                window = [str(_parse_date(day) - timedelta(days=lag)) for lag in range(self.history - 1, -1, -1)]
                if not all(item in available for item in window):
                    continue  # the window would reach before the split start or across a missing day
                inputs.append(np.stack([frames[item] for item in window]))
                targets.append(labels[name][day][None])
                used.append(day)
            if not inputs:
                raise ValueError(
                    f"{self.name}: split '{name}' has no run of {self.history} consecutive days "
                    f"({len(days)} dates)."
                )
            splits[name] = DataSplit(
                inputs=torch.from_numpy(np.stack(inputs).astype(np.float32)),
                targets=torch.from_numpy(np.stack(targets).astype(np.float32)),
                metadata={"dates": used, "history": self.history},
            )
        shape = splits["train"].inputs.shape
        return DataBundle(
            splits=splits,
            feature_spec=FeatureSpec(
                channels=int(shape[2]),
                description="Windows of standardised daily weather plus fuel channels on the Track-O grid.",
                extra={
                    "history": self.history,
                    "height": int(shape[3]),
                    "width": int(shape[4]),
                    "weather_vars": weather_vars,
                },
            ),
            label_spec=LabelSpec(
                num_targets=1,
                task_type="segmentation",
                description="Fire-occurrence mask of the last day of each window (1 = at least one detection).",
            ),
            metadata=self._metadata(weather_vars, grid, normalization, static, splits),
        )


def _day_of_year_features(day: str) -> Tuple[float, float]:
    parsed = _parse_date(day)
    days_in_year = 366 if parsed.year % 4 == 0 and (parsed.year % 100 != 0 or parsed.year % 400 == 0) else 365
    angle = 2.0 * math.pi * (parsed.timetuple().tm_yday - 1) / days_in_year
    return math.sin(angle), math.cos(angle)


class WildfireTrackOTabularDataset(_TrackOBase):
    """One row per grid cell and day: ``(N, F)`` features and ``(N,)`` int64 labels (1 = fire)."""

    name = "wildfire_track_o_tabular"
    default_downsample = 8

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        *,
        include_coords: bool = True,
        include_day_of_year: bool = True,
        **kwargs,
    ):
        super().__init__(cache_dir=cache_dir, **kwargs)
        self.include_coords = bool(include_coords)
        self.include_day_of_year = bool(include_day_of_year)

    def _load(self) -> DataBundle:
        split_dates, met, labels, static, weather_vars, grid, normalization = self._prepare()
        lat_grid, lon_grid = np.meshgrid(grid["lat"], grid["lon"], indexing="ij")
        coords = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=1).astype(np.float64)
        coords = ((coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-6)).astype(np.float32)
        n_cells = coords.shape[0]
        static_cols = static.reshape(static.shape[0], -1).T if static is not None else None

        splits: Dict[str, DataSplit] = {}
        for name, days in split_dates.items():
            rows, targets = [], []
            for day in days:
                blocks = [met[name][day].reshape(met[name][day].shape[0], -1).T]
                if self.include_coords:
                    blocks.append(coords)
                if self.include_day_of_year:
                    blocks.append(np.tile(np.asarray(_day_of_year_features(day), dtype=np.float32), (n_cells, 1)))
                if static_cols is not None:
                    blocks.append(static_cols)
                rows.append(np.concatenate(blocks, axis=1))
                targets.append(labels[name][day].reshape(-1))
            splits[name] = DataSplit(
                inputs=torch.from_numpy(np.concatenate(rows).astype(np.float32)),
                targets=torch.from_numpy(np.concatenate(targets).astype(np.int64)),
                metadata={"dates": list(days), "cells_per_day": n_cells},
            )

        feature_names = list(weather_vars)
        if self.include_coords:
            feature_names += ["lat", "lon"]
        if self.include_day_of_year:
            feature_names += ["sin_day_of_year", "cos_day_of_year"]
        if static is not None:
            feature_names += list(STATIC_FEATURE_NAMES)
        return DataBundle(
            splits=splits,
            feature_spec=FeatureSpec(
                input_dim=len(feature_names),
                description="Per-cell, per-day Track-O features (rows are day-major, cells in row-major order).",
                extra={"feature_names": feature_names, "height": int(lat_grid.shape[0]), "width": int(lat_grid.shape[1])},
            ),
            label_spec=LabelSpec(
                num_targets=2,
                task_type="classification",
                description="Whether the cell has at least one fire detection that day (0 = no, 1 = yes).",
            ),
            metadata=self._metadata(weather_vars, grid, normalization, static, splits),
        )


__all__ = [
    "DEFAULT_SPLITS_2024",
    "DEFAULT_WEATHER_VARS",
    "WildfireTrackORasterDataset",
    "WildfireTrackOTabularDataset",
    "WildfireTrackOTemporalDataset",
]
