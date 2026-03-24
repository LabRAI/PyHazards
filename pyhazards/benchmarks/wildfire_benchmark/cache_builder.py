from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

import numpy as np
import pandas as pd
import xarray as xr
import yaml

_DATE_RE = re.compile(r"pred_(\d{8})_\d{2}\.nc$")


@dataclass
class CacheBuildSummary:
    cache_root: Path
    n_label_days: int
    n_met_days: int
    n_shared_days: int
    weather_vars: List[str]
    train_days: int
    val_days: int
    test_days: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cache_root": str(self.cache_root),
            "n_label_days": int(self.n_label_days),
            "n_met_days": int(self.n_met_days),
            "n_shared_days": int(self.n_shared_days),
            "weather_vars": list(self.weather_vars),
            "train_days": int(self.train_days),
            "val_days": int(self.val_days),
            "test_days": int(self.test_days),
        }


def _read_yaml(path: str | Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _extract_pred_date(path: Path) -> str | None:
    match = _DATE_RE.search(path.name)
    if not match:
        return None
    stamp = match.group(1)
    return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"


def _load_grid(sample_nc: Path) -> tuple[np.ndarray, np.ndarray]:
    ds = xr.open_dataset(sample_nc)
    try:
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
        lon = np.asarray(ds["lon"].values, dtype=np.float64)
    finally:
        ds.close()
    return lat, lon


def _select_weather_vars(ds: xr.Dataset, weather_vars: Sequence[str]) -> xr.Dataset:
    missing = [name for name in weather_vars if name not in ds.data_vars]
    if missing:
        raise KeyError(f"Missing weather variables in dataset: {missing}")
    return ds[list(weather_vars)]


def _discover_weather_groups(weather_dir: Path, weather_glob: str) -> Dict[str, List[Path]]:
    grouped: Dict[str, List[Path]] = {}
    for path in sorted(weather_dir.glob(weather_glob)):
        date = _extract_pred_date(path)
        if date is None:
            continue
        grouped.setdefault(date, []).append(path)
    return grouped


def _discover_label_paths(firms_dir: Path, year: int) -> Dict[str, Path]:
    return {path.stem: path for path in sorted(firms_dir.glob(f"{year}-*.csv"))}


def _daily_weather_arrays(
    weather_groups: Dict[str, List[Path]],
    weather_vars: Sequence[str],
    *,
    allowed_dates: Set[str] | None = None,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for date in sorted(weather_groups):
        if allowed_dates is not None and date not in allowed_dates:
            continue
        stacks: List[np.ndarray] = []
        for path in weather_groups[date]:
            ds = xr.open_dataset(path)
            try:
                picked = _select_weather_vars(ds, weather_vars)
                arr = np.stack([np.asarray(picked[var].values, dtype=np.float32) for var in weather_vars], axis=0)
                if arr.ndim == 4 and arr.shape[1] == 1:
                    arr = arr[:, 0, :, :]
                stacks.append(arr)
            finally:
                ds.close()
        if stacks:
            out[date] = np.mean(np.stack(stacks, axis=0), axis=0).astype(np.float32)
    return out


def _read_firms_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _nearest_index(sorted_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(sorted_values, values)
    idx = np.clip(idx, 0, len(sorted_values) - 1)
    left = np.clip(idx - 1, 0, len(sorted_values) - 1)
    choose_left = np.abs(sorted_values[left] - values) <= np.abs(sorted_values[idx] - values)
    return np.where(choose_left, left, idx)


def _firms_to_binary_grid(df: pd.DataFrame, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    label = np.zeros((lat.size, lon.size), dtype=np.float32)
    if df.empty:
        return label

    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise KeyError("FIRMS CSV must include 'latitude' and 'longitude' columns.")

    lat_vals = df["latitude"].to_numpy(dtype=np.float64, copy=False)
    lon_vals = df["longitude"].to_numpy(dtype=np.float64, copy=False)
    valid = np.isfinite(lat_vals) & np.isfinite(lon_vals)
    lat_vals = lat_vals[valid]
    lon_vals = lon_vals[valid]
    if lat_vals.size == 0:
        return label

    lat_idx = _nearest_index(lat, lat_vals)
    lon_idx = _nearest_index(lon, lon_vals)
    label[lat_idx, lon_idx] = 1.0
    return label


def _daily_label_arrays(
    label_paths: Dict[str, Path],
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    allowed_dates: Set[str] | None = None,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for date in sorted(label_paths):
        if allowed_dates is not None and date not in allowed_dates:
            continue
        df = _read_firms_csv(label_paths[date])
        out[date] = _firms_to_binary_grid(df, lat=lat, lon=lon)
    return out


def _write_lines(path: Path, items: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(items), encoding="utf-8")


def _date_in_range(date: str, start: str, end: str) -> bool:
    return start <= date <= end


def _write_split_files(cache_root: Path, dates: Sequence[str], split_cfg: Dict[str, Sequence[str]]) -> Dict[str, int]:
    split_root = cache_root / "splits"
    counts: Dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        start, end = split_cfg[split_name]
        split_dates = [d for d in dates if _date_in_range(d, str(start), str(end))]
        _write_lines(split_root / f"{split_name}_dates.txt", split_dates)
        counts[split_name] = len(split_dates)
    return counts


def build_cache(config_path: str | Path, *, limit_days: int = 0) -> CacheBuildSummary:
    cfg = _read_yaml(config_path)

    cache_root = Path(cfg["cache"]["root"])
    labels_dir = cache_root / "labels"
    met_dir = cache_root / "met"
    static_dir = cache_root / "static"
    metadata_dir = cache_root / "metadata"
    for path in (labels_dir, met_dir, static_dir, metadata_dir):
        path.mkdir(parents=True, exist_ok=True)

    weather_dir = Path(cfg["data"]["weather_dir"])
    weather_glob = str(cfg["data"].get("weather_glob", "pred_2024*.nc"))
    weather_vars = list(cfg["data"]["weather_vars"])
    sample_nc = weather_dir / str(cfg["data"].get("sample_nc", "pred_20240101_18.nc"))
    firms_dir = Path(cfg["data"]["firms_daily_dir"])
    landfire_tif = Path(cfg["data"]["landfire_tif"])
    year = int(cfg["data"]["year"])

    lat, lon = _load_grid(sample_nc)
    np.save(metadata_dir / "lat.npy", lat.astype(np.float32))
    np.save(metadata_dir / "lon.npy", lon.astype(np.float32))
    (metadata_dir / "grid.json").write_text(
        json.dumps(
            {
                "sample_nc": str(sample_nc),
                "lat_size": int(lat.size),
                "lon_size": int(lon.size),
                "lat_min": float(lat.min()),
                "lat_max": float(lat.max()),
                "lon_min": float(lon.min()),
                "lon_max": float(lon.max()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (metadata_dir / "vars.json").write_text(json.dumps({"weather_vars": weather_vars}, indent=2), encoding="utf-8")

    weather_groups = _discover_weather_groups(weather_dir, weather_glob)
    label_paths = _discover_label_paths(firms_dir, year)

    candidate_shared_dates = sorted(set(weather_groups) & set(label_paths))
    if limit_days > 0:
        candidate_shared_dates = candidate_shared_dates[:limit_days]
    allowed_dates = set(candidate_shared_dates)

    met_arrays = _daily_weather_arrays(weather_groups, weather_vars, allowed_dates=allowed_dates)
    label_arrays = _daily_label_arrays(label_paths, lat=lat, lon=lon, allowed_dates=allowed_dates)

    shared_dates = sorted(set(met_arrays) & set(label_arrays))

    for date in shared_dates:
        np.save(met_dir / f"{date}.npy", met_arrays[date])
        np.save(labels_dir / f"{date}.npy", label_arrays[date])

    _write_lines(cache_root / "dates.txt", shared_dates)

    static_manifest = {
        "fuel_source": str(landfire_tif),
        "status": "source_registered_only",
        "message": "Static fuel reprojection to the benchmark grid is deferred until rasterio/rioxarray are available.",
        "expected_output_path": str(static_dir / "fuel.npy"),
    }
    (static_dir / "fuel_manifest.json").write_text(json.dumps(static_manifest, indent=2), encoding="utf-8")

    split_counts = _write_split_files(cache_root, shared_dates, cfg["splits"])

    summary = CacheBuildSummary(
        cache_root=cache_root,
        n_label_days=len(label_paths),
        n_met_days=len(weather_groups),
        n_shared_days=len(shared_dates),
        weather_vars=weather_vars,
        train_days=split_counts["train"],
        val_days=split_counts["val"],
        test_days=split_counts["test"],
    )
    (cache_root / "cache_summary.json").write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    return summary


def align_static_fuel_to_cache(
    cache_root: str | Path,
    *,
    landfire_tif: str | Path | None = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    import tifffile as tf

    cache_root = Path(cache_root)
    static_dir = cache_root / "static"
    metadata_dir = cache_root / "metadata"
    static_dir.mkdir(parents=True, exist_ok=True)

    lat = np.load(metadata_dir / "lat.npy")
    lon = np.load(metadata_dir / "lon.npy")
    manifest_path = static_dir / "fuel_manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    source_path = Path(landfire_tif or manifest.get("fuel_source") or "")
    if not str(source_path):
        raise ValueError("LANDFIRE source path is required to align static fuel to the cache grid.")
    if not source_path.exists():
        raise FileNotFoundError(f"LANDFIRE source not found: {source_path}")

    fuel_npy_path = static_dir / "fuel.npy"
    fuel_mask_path = static_dir / "fuel_mask.npy"
    aligned_tif_path = static_dir / "fuel_aligned_benchmark_grid.tif"
    if fuel_npy_path.exists() and fuel_mask_path.exists() and not overwrite:
        payload = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        return payload

    width = int(lon.size)
    height = int(lat.size)
    cmd = [
        "gdalwarp",
        "-overwrite",
        "-multi",
        "-wo",
        "NUM_THREADS=ALL_CPUS",
        "-t_srs",
        "EPSG:4326",
        "-te",
        "-180",
        "-90",
        "180",
        "90",
        "-ts",
        str(width),
        str(height),
        "-r",
        "mode",
        "-srcnodata",
        "32767",
        "-dstnodata",
        "-9999",
        str(source_path),
        str(aligned_tif_path),
    ]
    subprocess.run(cmd, check=True)

    raw = tf.imread(aligned_tif_path)
    if raw.shape != (height, width):
        raise ValueError(f"Aligned fuel raster shape mismatch: expected {(height, width)}, got {raw.shape}")

    valid_mask = raw >= 0
    fuel = np.where(valid_mask, raw, 0).astype(np.int16)
    fuel_mask = valid_mask.astype(np.uint8)
    np.save(fuel_npy_path, fuel)
    np.save(fuel_mask_path, fuel_mask)

    unique_valid = np.unique(fuel[valid_mask]) if np.any(valid_mask) else np.asarray([], dtype=np.int16)
    payload = {
        "fuel_source": str(source_path),
        "status": "aligned_to_cache_grid",
        "grid_shape": [height, width],
        "warp": {
            "target_srs": "EPSG:4326",
            "target_extent": [-180, -90, 180, 90],
            "target_size": [width, height],
            "resampling": "mode",
            "dst_nodata": -9999,
        },
        "output_files": {
            "fuel": str(fuel_npy_path),
            "fuel_mask": str(fuel_mask_path),
            "aligned_tif": str(aligned_tif_path),
        },
        "valid_cells": int(valid_mask.sum()),
        "valid_fraction": float(valid_mask.mean()),
        "unique_valid_values_count": int(unique_valid.size),
        "unique_valid_values_sample": unique_valid[:32].astype(int).tolist(),
        "notes": [
            "Static fuel values were warped from LANDFIRE CONUS Albers to the benchmark's nominal global lat-lon grid.",
            "Negative values were treated as outside-domain/no-data and written as fuel=0 with fuel_mask=0.",
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary_path = cache_root / "cache_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["static_fuel"] = {
            "status": "aligned",
            "fuel_file": str(fuel_npy_path),
            "fuel_mask_file": str(fuel_mask_path),
            "valid_cells": int(valid_mask.sum()),
            "valid_fraction": float(valid_mask.mean()),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return payload


__all__ = ["CacheBuildSummary", "build_cache", "align_static_fuel_to_cache"]
