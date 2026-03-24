from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ..base import DataBundle, DataSplit, Dataset, FeatureSpec, LabelSpec


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Expected split file not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_weather_vars(cache_root: Path) -> list[str]:
    payload = json.loads((cache_root / "metadata" / "vars.json").read_text(encoding="utf-8"))
    return list(payload["weather_vars"])


def _load_lat_lon(cache_root: Path) -> tuple[np.ndarray, np.ndarray]:
    lat = np.load(cache_root / "metadata" / "lat.npy")
    lon = np.load(cache_root / "metadata" / "lon.npy")
    return np.asarray(lat, dtype=np.float32), np.asarray(lon, dtype=np.float32)


def _subset_dates(dates: Sequence[str], limit: int | None) -> list[str]:
    if limit is None or int(limit) <= 0:
        return list(dates)
    return list(dates[: int(limit)])


def _crop_hw_to_multiple(arr: np.ndarray, multiple: int = 4) -> np.ndarray:
    if arr.ndim == 3:
        _, h, w = arr.shape
        h2 = h - (h % multiple)
        w2 = w - (w % multiple)
        return arr[:, : max(h2, multiple), : max(w2, multiple)]
    if arr.ndim == 2:
        h, w = arr.shape
        h2 = h - (h % multiple)
        w2 = w - (w % multiple)
        return arr[: max(h2, multiple), : max(w2, multiple)]
    raise ValueError(f"Unsupported array rank for cropping: {arr.ndim}")


def _spatial_downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        out = np.asarray(arr, dtype=np.float32)
    elif arr.ndim == 3:
        out = np.asarray(arr[:, ::factor, ::factor], dtype=np.float32)
    elif arr.ndim == 2:
        out = np.asarray(arr[::factor, ::factor], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported array rank for downsampling: {arr.ndim}")
    return np.asarray(_crop_hw_to_multiple(out, multiple=4), dtype=np.float32)


def _compute_channel_stats(cache_root: Path, dates: Sequence[str], downsample_factor: int) -> tuple[np.ndarray, np.ndarray]:
    weather_dir = cache_root / "met"
    example = _spatial_downsample(np.load(weather_dir / f"{dates[0]}.npy"), downsample_factor)
    channels = int(example.shape[0])
    sums = np.zeros((channels,), dtype=np.float64)
    sums_sq = np.zeros((channels,), dtype=np.float64)
    count = 0
    for date in dates:
        arr = _spatial_downsample(np.load(weather_dir / f"{date}.npy"), downsample_factor)
        flat = arr.reshape(channels, -1).astype(np.float64, copy=False)
        sums += flat.sum(axis=1)
        sums_sq += np.square(flat).sum(axis=1)
        count += flat.shape[1]
    mean = sums / max(count, 1)
    var = np.maximum(sums_sq / max(count, 1) - np.square(mean), 1e-12)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def _normalize_weather(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((arr - mean[:, None, None]) / std[:, None, None]).astype(np.float32)


def _load_static_fuel(cache_root: Path, downsample_factor: int) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    fuel_path = cache_root / "static" / "fuel.npy"
    fuel_mask_path = cache_root / "static" / "fuel_mask.npy"
    if not fuel_path.exists():
        return None, None
    fuel = np.load(fuel_path)
    fuel_mask = np.load(fuel_mask_path) if fuel_mask_path.exists() else (fuel > 0).astype(np.uint8)
    fuel = _spatial_downsample(fuel.astype(np.float32), downsample_factor)
    fuel_mask = _spatial_downsample(fuel_mask.astype(np.float32), downsample_factor)
    fuel_mask = (fuel_mask > 0.5).astype(np.float32)
    fuel = np.where(fuel_mask > 0, fuel / 100.0, 0.0).astype(np.float32)
    return fuel, fuel_mask


def _date_to_cyclical_features(date_text: str) -> tuple[float, float]:
    month = int(date_text[5:7])
    day = int(date_text[8:10])
    day_of_year = (month - 1) * 31 + day
    angle = 2.0 * math.pi * (float(day_of_year) / 366.0)
    return float(math.sin(angle)), float(math.cos(angle))


@dataclass(frozen=True)
class TrackOSplitConfig:
    train_limit_days: int | None = None
    val_limit_days: int | None = None
    test_limit_days: int | None = None


class _WildfireTrackOBase(Dataset):
    name = "wildfire_track_o_2024_base"

    def __init__(
        self,
        cache_dir: str | None = None,
        *,
        downsample_factor: int = 1,
        train_limit_days: int | None = None,
        val_limit_days: int | None = None,
        test_limit_days: int | None = None,
    ):
        super().__init__(cache_dir=cache_dir)
        self.cache_root = Path(cache_dir or "/home/runyang/my-copy/data_cache/wildfire_2024_v1")
        self.downsample_factor = max(1, int(downsample_factor))
        self.split_cfg = TrackOSplitConfig(
            train_limit_days=train_limit_days,
            val_limit_days=val_limit_days,
            test_limit_days=test_limit_days,
        )

    def _load_split_dates(self) -> dict[str, list[str]]:
        split_root = self.cache_root / "splits"
        return {
            "train": _subset_dates(_read_lines(split_root / "train_dates.txt"), self.split_cfg.train_limit_days),
            "val": _subset_dates(_read_lines(split_root / "val_dates.txt"), self.split_cfg.val_limit_days),
            "test": _subset_dates(_read_lines(split_root / "test_dates.txt"), self.split_cfg.test_limit_days),
        }


class WildfireTrackO2024RasterDataset(_WildfireTrackOBase):
    name = "wildfire_track_o_2024_raster"

    def __init__(
        self,
        cache_dir: str | None = None,
        *,
        downsample_factor: int = 4,
        train_limit_days: int | None = None,
        val_limit_days: int | None = None,
        test_limit_days: int | None = None,
    ):
        super().__init__(
            cache_dir=cache_dir,
            downsample_factor=downsample_factor,
            train_limit_days=train_limit_days,
            val_limit_days=val_limit_days,
            test_limit_days=test_limit_days,
        )

    def _load(self) -> DataBundle:
        split_dates = self._load_split_dates()
        weather_vars = _load_weather_vars(self.cache_root)
        mean, std = _compute_channel_stats(self.cache_root, split_dates["train"], self.downsample_factor)
        fuel, fuel_mask = _load_static_fuel(self.cache_root, self.downsample_factor)

        splits: dict[str, DataSplit] = {}
        for split_name, dates in split_dates.items():
            x_rows: list[np.ndarray] = []
            y_rows: list[np.ndarray] = []
            for date in dates:
                x = _spatial_downsample(np.load(self.cache_root / "met" / f"{date}.npy"), self.downsample_factor)
                x = _normalize_weather(x, mean, std)
                if fuel is not None and fuel_mask is not None:
                    x = np.concatenate([x, fuel[None, :, :], fuel_mask[None, :, :]], axis=0)
                x_rows.append(x.astype(np.float32))
                y = _spatial_downsample(np.load(self.cache_root / "labels" / f"{date}.npy"), self.downsample_factor)
                y_rows.append(y[None, :, :].astype(np.float32))

            x_np = np.stack(x_rows, axis=0).astype(np.float32)
            y_np = np.stack(y_rows, axis=0).astype(np.float32)
            splits[split_name] = DataSplit(
                inputs=torch.from_numpy(x_np),
                targets=torch.from_numpy(y_np),
                metadata={"dates": list(dates)},
            )

        sample_shape = splits["train"].inputs.shape
        return DataBundle(
            splits=splits,
            feature_spec=FeatureSpec(
                channels=int(sample_shape[1]),
                description="Daily gridded wildfire covariates from the 2024 Prithvi-WxC weather cache.",
                extra={
                    "height": int(sample_shape[2]),
                    "width": int(sample_shape[3]),
                    "downsample_factor": self.downsample_factor,
                    "weather_vars": weather_vars,
                    "static_feature_names": ["fuel_class_scaled", "fuel_valid_mask"] if fuel is not None else [],
                },
            ),
            label_spec=LabelSpec(
                num_targets=1,
                task_type="segmentation",
                description="Binary daily wildfire occurrence grid aligned to the benchmark cache.",
            ),
            metadata={
                "dataset": self.name,
                "cache_root": str(self.cache_root),
                "has_static_fuel": fuel is not None,
                "normalization": {
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                    "fit_split": "train",
                },
                "splits": {k: len(v) for k, v in split_dates.items()},
            },
        )


class WildfireTrackO2024TemporalDataset(_WildfireTrackOBase):
    name = "wildfire_track_o_2024_temporal"

    def __init__(
        self,
        cache_dir: str | None = None,
        *,
        history: int = 6,
        downsample_factor: int = 8,
        train_limit_days: int | None = None,
        val_limit_days: int | None = None,
        test_limit_days: int | None = None,
    ):
        super().__init__(
            cache_dir=cache_dir,
            downsample_factor=downsample_factor,
            train_limit_days=train_limit_days,
            val_limit_days=val_limit_days,
            test_limit_days=test_limit_days,
        )
        self.history = int(history)

    def _load(self) -> DataBundle:
        split_dates = self._load_split_dates()
        weather_vars = _load_weather_vars(self.cache_root)
        mean, std = _compute_channel_stats(self.cache_root, split_dates["train"], self.downsample_factor)
        fuel, fuel_mask = _load_static_fuel(self.cache_root, self.downsample_factor)
        static_channels = None
        if fuel is not None and fuel_mask is not None:
            static_channels = np.stack([fuel, fuel_mask], axis=0).astype(np.float32)

        splits: dict[str, DataSplit] = {}
        for split_name, dates in split_dates.items():
            x_rows: list[np.ndarray] = []
            y_rows: list[np.ndarray] = []
            used_dates: list[str] = []

            for idx in range(self.history - 1, len(dates)):
                seq_dates = dates[idx - self.history + 1 : idx + 1]
                seq_arrays = []
                for date in seq_dates:
                    x = _spatial_downsample(np.load(self.cache_root / "met" / f"{date}.npy"), self.downsample_factor)
                    x = _normalize_weather(x, mean, std)
                    if static_channels is not None:
                        x = np.concatenate([x, static_channels], axis=0)
                    seq_arrays.append(x.astype(np.float32))
                x_rows.append(np.stack(seq_arrays, axis=0).astype(np.float32))
                target = _spatial_downsample(np.load(self.cache_root / "labels" / f"{dates[idx]}.npy"), self.downsample_factor)
                y_rows.append(target[None, :, :].astype(np.float32))
                used_dates.append(dates[idx])

            if not x_rows:
                raise ValueError(
                    f"Temporal split '{split_name}' has no usable samples. Need at least history={self.history} dates, got {len(dates)}."
                )

            x_np = np.stack(x_rows, axis=0).astype(np.float32)
            y_np = np.stack(y_rows, axis=0).astype(np.float32)
            splits[split_name] = DataSplit(
                inputs=torch.from_numpy(x_np),
                targets=torch.from_numpy(y_np),
                metadata={"dates": used_dates, "history": self.history},
            )

        sample_shape = splits["train"].inputs.shape
        return DataBundle(
            splits=splits,
            feature_spec=FeatureSpec(
                channels=int(sample_shape[2]),
                description="Temporal weather histories for wildfire occurrence prediction.",
                extra={
                    "history": self.history,
                    "height": int(sample_shape[3]),
                    "width": int(sample_shape[4]),
                    "downsample_factor": self.downsample_factor,
                    "weather_vars": weather_vars,
                    "static_feature_names": ["fuel_class_scaled", "fuel_valid_mask"] if static_channels is not None else [],
                },
            ),
            label_spec=LabelSpec(
                num_targets=1,
                task_type="segmentation",
                description="Binary daily wildfire occurrence grid for the last frame in each history window.",
            ),
            metadata={
                "dataset": self.name,
                "cache_root": str(self.cache_root),
                "has_static_fuel": fuel is not None,
                "normalization": {
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                    "fit_split": "train",
                },
                "splits": {k: int(v.inputs.shape[0]) for k, v in splits.items()},
            },
        )


class WildfireTrackO2024TabularDataset(_WildfireTrackOBase):
    name = "wildfire_track_o_2024_tabular"

    def __init__(
        self,
        cache_dir: str | None = None,
        *,
        downsample_factor: int = 8,
        include_coords: bool = True,
        include_day_of_year: bool = True,
        train_limit_days: int | None = None,
        val_limit_days: int | None = None,
        test_limit_days: int | None = None,
    ):
        super().__init__(
            cache_dir=cache_dir,
            downsample_factor=downsample_factor,
            train_limit_days=train_limit_days,
            val_limit_days=val_limit_days,
            test_limit_days=test_limit_days,
        )
        self.include_coords = bool(include_coords)
        self.include_day_of_year = bool(include_day_of_year)

    def _load(self) -> DataBundle:
        split_dates = self._load_split_dates()
        weather_vars = _load_weather_vars(self.cache_root)
        lat, lon = _load_lat_lon(self.cache_root)
        mean, std = _compute_channel_stats(self.cache_root, split_dates["train"], self.downsample_factor)
        fuel, fuel_mask = _load_static_fuel(self.cache_root, self.downsample_factor)

        sample_met = _spatial_downsample(np.load(self.cache_root / "met" / f"{split_dates['train'][0]}.npy"), self.downsample_factor)
        lat = lat[:: self.downsample_factor][: sample_met.shape[1]]
        lon = lon[:: self.downsample_factor][: sample_met.shape[2]]
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
        coord_block = np.stack([lat_grid, lon_grid], axis=-1).reshape(-1, 2).astype(np.float32)
        coord_mean = coord_block.mean(axis=0, keepdims=True)
        coord_std = coord_block.std(axis=0, keepdims=True) + 1e-6
        coord_block = (coord_block - coord_mean) / coord_std

        splits: dict[str, DataSplit] = {}
        for split_name, dates in split_dates.items():
            x_rows: list[np.ndarray] = []
            y_rows: list[np.ndarray] = []
            row_dates: list[str] = []

            for date in dates:
                met = _spatial_downsample(np.load(self.cache_root / "met" / f"{date}.npy"), self.downsample_factor)
                met = _normalize_weather(met, mean, std)
                features = met.reshape(met.shape[0], -1).T.astype(np.float32)

                extras: list[np.ndarray] = []
                if self.include_coords:
                    extras.append(coord_block)
                if self.include_day_of_year:
                    sin_doy, cos_doy = _date_to_cyclical_features(date)
                    extras.append(
                        np.repeat(
                            np.asarray([[sin_doy, cos_doy]], dtype=np.float32),
                            repeats=features.shape[0],
                            axis=0,
                        )
                    )
                if extras:
                    features = np.concatenate([features, *extras], axis=1)
                if fuel is not None and fuel_mask is not None:
                    fuel_cols = fuel.reshape(-1, 1).astype(np.float32)
                    fuel_mask_cols = fuel_mask.reshape(-1, 1).astype(np.float32)
                    features = np.concatenate([features, fuel_cols, fuel_mask_cols], axis=1)

                label = _spatial_downsample(np.load(self.cache_root / "labels" / f"{date}.npy"), self.downsample_factor)
                labels = label.reshape(-1).astype(np.float32)

                x_rows.append(features)
                y_rows.append(labels)
                row_dates.extend([date] * features.shape[0])

            x_np = np.concatenate(x_rows, axis=0).astype(np.float32)
            y_np = np.concatenate(y_rows, axis=0).astype(np.float32)
            splits[split_name] = DataSplit(
                inputs=torch.from_numpy(x_np),
                targets=torch.from_numpy(y_np),
                metadata={"row_dates": row_dates},
            )

        feature_names = list(weather_vars)
        if self.include_coords:
            feature_names.extend(["lat", "lon"])
        if self.include_day_of_year:
            feature_names.extend(["sin_doy", "cos_doy"])
        if fuel is not None and fuel_mask is not None:
            feature_names.extend(["fuel_class_scaled", "fuel_valid_mask"])

        return DataBundle(
            splits=splits,
            feature_spec=FeatureSpec(
                input_dim=int(splits["train"].inputs.shape[1]),
                description="Tabularized wildfire occurrence features from daily gridded cache values.",
                extra={
                    "downsample_factor": self.downsample_factor,
                    "feature_names": feature_names,
                },
            ),
            label_spec=LabelSpec(
                num_targets=1,
                task_type="classification",
                description="Binary wildfire occurrence for each grid cell and day in tabular form.",
            ),
            metadata={
                "dataset": self.name,
                "cache_root": str(self.cache_root),
                "has_static_fuel": fuel is not None,
                "normalization": {
                    "weather_mean": mean.tolist(),
                    "weather_std": std.tolist(),
                    "fit_split": "train",
                },
                "splits": {k: int(v.inputs.shape[0]) for k, v in splits.items()},
            },
        )


__all__ = [
    "TrackOSplitConfig",
    "WildfireTrackO2024RasterDataset",
    "WildfireTrackO2024TemporalDataset",
    "WildfireTrackO2024TabularDataset",
]
