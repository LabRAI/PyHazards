"""Build a Track-O cache from daily weather NetCDF files, NASA FIRMS CSVs and a LANDFIRE fuel raster.

Adapted from the cache builder of PyHazards PR #33 by runyangxu
(``pyhazards/benchmarks/wildfire_benchmark/cache_builder.py``), with these changes:

- weather files: any NetCDF files holding the requested variables on a regular latitude-longitude
  grid, with the date read from the file name. Variables may be spread over several files of the
  same day (e.g. the MERRA-2 ``tavg1_2d_slv/flx/rad/lnd`` collections) and every non-spatial
  dimension (time, a length-1 level) is averaged, so a day of hourly files becomes a daily mean.
  PR #33 read the outputs of its own Prithvi-WxC runs (``pred_YYYYMMDD_HH.nc``), which are not
  public; the MERRA-2 surface files written by ``python -m pyhazards.datasets.merra2.inspection``
  (``MERRA2_sfc_YYYYMMDD.nc``) hold the same 14 variables;
- the grid is stored south-to-north and west-to-east whatever the order in the files, and every
  weather file must share it;
- FIRMS CSVs: the archive / NRT downloads are read with the standard library ``csv`` module and
  grouped by their ``acq_date`` column (UTC); a file without that column must be named after its
  day. Rows whose FIRMS ``type`` is not in ``firms_types`` (by default only 0, presumed vegetation
  fire, which drops volcanoes, other static land sources and offshore detections) are skipped when
  the column exists. Detections are snapped to the nearest cell centre; detections more than half a
  cell outside the grid are dropped instead of being piled onto the border cells;
- LANDFIRE fuel: reprojected with rasterio (optional extra ``pyhazards[geo]``) onto the cell
  edges implied by the grid's cell centres, with mode resampling, and flipped to the cache's
  south-to-north row order. PR #33 called ``gdalwarp -te -180 -90 180 90`` and stored the north-up
  result unflipped, which on a south-to-north latitude axis such as MERRA-2's puts the fuel layer
  upside down (CONUS fuels land at 22-52 degrees south) and shifts it by half a cell. LANDFIRE's
  -9999 fill value is excluded from the mode even when the file's nodata tag says 32767;
- days are processed one at a time instead of holding the whole year in memory.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .track_o import DEFAULT_SPLITS_2024, DEFAULT_WEATHER_VARS, SPLIT_NAMES

PathLike = Union[str, Path]

DEFAULT_DATE_PATTERN = r"(?<!\d)(\d{4}-?\d{2}-?\d{2})(?!\d)"
FUEL_NODATA_OUT = -9999
# LANDFIRE fuel GeoTIFFs mark missing cells with -9999 ("Fill-NoData"), while some files carry a
# GDAL nodata tag of 32767; both are treated as missing.
LANDFIRE_NODATA = -9999
LANDFIRE_INVALID_CODES = (-9999, 32767)


def _to_iso(stamp: str) -> str:
    digits = stamp.replace("-", "")
    if len(digits) != 8 or not digits.isdigit():
        raise ValueError(f"not a YYYYMMDD / YYYY-MM-DD date: {stamp!r}")
    return str(date(int(digits[:4]), int(digits[4:6]), int(digits[6:8])))


def date_from_name(path: PathLike, pattern: str = DEFAULT_DATE_PATTERN) -> Optional[str]:
    """ISO date taken from the first group of ``pattern`` in the file name, or ``None``."""
    match = re.search(pattern, Path(path).name)
    if not match:
        return None
    try:
        return _to_iso(match.group(1))
    except ValueError:
        return None


# ---------------------------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------------------------
def _coord_name(ds, candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in ds.coords or name in ds.variables:
            return name
    raise KeyError(f"no coordinate among {list(candidates)} in a weather file (have {list(ds.coords)})")


def _daily_field(da, lat_name: str, lon_name: str) -> np.ndarray:
    """Average every dimension except latitude and longitude and return ``(lat, lon)``."""
    extra = [dim for dim in da.dims if dim not in (lat_name, lon_name)]
    if extra:
        da = da.mean(dim=extra, skipna=True)
    return np.asarray(da.transpose(lat_name, lon_name).values, dtype=np.float64)


def read_daily_weather(
    paths: Sequence[PathLike],
    weather_vars: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Daily mean of ``weather_vars`` over ``paths`` (the files of one day).

    Returns ``(fields, lat, lon)`` with ``fields`` of shape ``(len(weather_vars), n_lat, n_lon)``
    (float32), latitude and longitude ascending. A variable found in several files is averaged
    over them; a variable missing from all of them raises ``KeyError``.
    """
    import xarray as xr

    sums: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    grid: Optional[Tuple[np.ndarray, np.ndarray]] = None
    for path in paths:
        with xr.open_dataset(path) as ds:
            lat_name = _coord_name(ds, ("lat", "latitude"))
            lon_name = _coord_name(ds, ("lon", "longitude"))
            lat = np.asarray(ds[lat_name].values, dtype=np.float64)
            lon = np.asarray(ds[lon_name].values, dtype=np.float64)
            lat_order, lon_order = np.argsort(lat), np.argsort(lon)
            lat, lon = lat[lat_order], lon[lon_order]
            if grid is None:
                grid = (lat, lon)
            elif not (
                grid[0].shape == lat.shape
                and grid[1].shape == lon.shape
                and np.allclose(grid[0], lat)
                and np.allclose(grid[1], lon)
            ):
                raise ValueError(f"{path} is on a different latitude-longitude grid than the other files of its day.")
            for name in weather_vars:
                if name not in ds.data_vars:
                    continue
                field = _daily_field(ds[name], lat_name, lon_name)[np.ix_(lat_order, lon_order)]
                sums[name] = sums.get(name, 0.0) + field
                counts[name] = counts.get(name, 0) + 1
    missing = [name for name in weather_vars if name not in sums]
    if missing:
        raise KeyError(f"weather variables {missing} are in none of {[str(p) for p in paths]}")
    if grid is None:
        raise ValueError("read_daily_weather needs at least one file")
    fields = np.stack([sums[name] / counts[name] for name in weather_vars]).astype(np.float32)
    return fields, grid[0], grid[1]


# ---------------------------------------------------------------------------------------------
# FIRMS labels
# ---------------------------------------------------------------------------------------------
def read_firms_detections(
    csv_paths: Iterable[PathLike],
    *,
    firms_types: Optional[Sequence[int]] = (0,),
    date_pattern: str = DEFAULT_DATE_PATTERN,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, int]]:
    """Group FIRMS detections by day.

    Returns ``({iso_date: (latitudes, longitudes)}, stats)``. The day comes from ``acq_date`` when
    the file has that column, else from the file name. ``firms_types`` keeps rows whose ``type``
    is listed (when the column exists; ``None`` keeps everything).
    """
    by_day: Dict[str, Tuple[List[float], List[float]]] = {}
    stats = {"files": 0, "rows": 0, "kept": 0, "dropped_type": 0, "files_without_type": 0}
    allowed = None if firms_types is None else {int(value) for value in firms_types}
    for path in csv_paths:
        path = Path(path)
        stats["files"] += 1
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            if not {"latitude", "longitude"} <= columns:
                raise KeyError(f"FIRMS CSV {path} lacks 'latitude' / 'longitude' columns (has {sorted(columns)}).")
            file_day = None
            if "acq_date" not in columns:
                file_day = date_from_name(path, date_pattern)
                if file_day is None:
                    raise ValueError(f"FIRMS CSV {path} has no acq_date column and no date in its name.")
            if "type" not in columns:
                stats["files_without_type"] += 1
            for row in reader:
                stats["rows"] += 1
                if allowed is not None and "type" in columns and row.get("type", "") != "":
                    if int(float(row["type"])) not in allowed:
                        stats["dropped_type"] += 1
                        continue
                try:
                    lat, lon = float(row["latitude"]), float(row["longitude"])
                except (TypeError, ValueError):
                    continue
                if not (np.isfinite(lat) and np.isfinite(lon)):
                    continue
                day = file_day or _to_iso(row["acq_date"].strip())
                lats, lons = by_day.setdefault(day, ([], []))
                lats.append(lat)
                lons.append(lon)
                stats["kept"] += 1
    return {day: (np.asarray(v[0]), np.asarray(v[1])) for day, v in by_day.items()}, stats


def _regular_step(values: np.ndarray, name: str) -> float:
    steps = np.diff(values)
    if values.size < 2 or not np.allclose(steps, steps[0], rtol=1e-4, atol=1e-6) or steps[0] <= 0:
        raise ValueError(f"the {name} axis must be regular and ascending")
    return float(steps[0])


def rasterize_points(
    point_lat: np.ndarray,
    point_lon: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:
    """``(n_lat, n_lon)`` float32 grid with 1.0 in every cell whose centre is nearest to a point.

    ``lat`` and ``lon`` are ascending, regular cell centres. Longitudes are compared modulo 360
    (a grid on 0..360 accepts -180..180 points); a grid that spans the globe wraps around.
    Points more than half a cell outside the grid are ignored.
    """
    grid = np.zeros((lat.size, lon.size), dtype=np.float32)
    point_lat = np.asarray(point_lat, dtype=np.float64)
    point_lon = np.asarray(point_lon, dtype=np.float64)
    if point_lat.size == 0:
        return grid
    dlat, dlon = _regular_step(lat, "latitude"), _regular_step(lon, "longitude")
    rows = np.rint((point_lat - lat[0]) / dlat).astype(np.int64)
    offset = np.mod(point_lon - lon[0] + dlon / 2.0, 360.0) - dlon / 2.0
    cols = np.rint(offset / dlon).astype(np.int64)
    if abs(lon.size * dlon - 360.0) < 1e-6:
        cols = np.mod(cols, lon.size)
    keep = (rows >= 0) & (rows < lat.size) & (cols >= 0) & (cols < lon.size)
    grid[rows[keep], cols[keep]] = 1.0
    return grid


# ---------------------------------------------------------------------------------------------
# LANDFIRE fuel
# ---------------------------------------------------------------------------------------------
def grid_bounds(lat: np.ndarray, lon: np.ndarray) -> Tuple[float, float, float, float]:
    """``(west, south, east, north)`` cell edges of an ascending, regular cell-centre grid."""
    dlat, dlon = _regular_step(lat, "latitude"), _regular_step(lon, "longitude")
    return (
        float(lon[0] - dlon / 2.0),
        float(lat[0] - dlat / 2.0),
        float(lon[-1] + dlon / 2.0),
        float(lat[-1] + dlat / 2.0),
    )


def _import_rasterio():
    try:
        import rasterio
        from rasterio.warp import Resampling, reproject
    except ImportError as exc:
        raise ImportError(
            "Aligning a LANDFIRE fuel raster needs the optional rasterio package: pip install 'pyhazards[geo]'."
        ) from exc
    return rasterio, reproject, Resampling


def align_fuel_to_grid(
    raster_path: PathLike,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    src_nodata: Optional[float] = LANDFIRE_NODATA,
    invalid_codes: Sequence[int] = LANDFIRE_INVALID_CODES,
    num_threads: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mode-resample a categorical fuel raster onto the cache grid.

    ``lat`` / ``lon`` are the ascending cell centres of the cache. Returns ``(fuel, mask)`` as
    ``(n_lat, n_lon)`` int32 codes (0 where invalid) and uint8 validity, rows south-to-north.
    Source pixels equal to ``src_nodata`` (LANDFIRE's -9999 by default; ``None`` uses the raster's
    nodata tag) are left out of the mode. A cell whose mode is negative or in ``invalid_codes`` is
    marked invalid. Only the cells overlapping the raster's footprint are warped (a CONUS raster on
    a global grid).
    """
    rasterio, reproject, Resampling = _import_rasterio()
    from rasterio.transform import from_origin
    from rasterio.warp import transform_bounds

    dlat, dlon = _regular_step(lat, "latitude"), _regular_step(lon, "longitude")
    fuel = np.full((lat.size, lon.size), FUEL_NODATA_OUT, dtype=np.int32)
    with rasterio.open(raster_path) as src:
        nodata = src.nodata if src_nodata is None else src_nodata
        left, bottom, right, top = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        rows = np.flatnonzero((lat + dlat / 2.0 > bottom) & (lat - dlat / 2.0 < top))
        cols = np.flatnonzero((lon + dlon / 2.0 > left) & (lon - dlon / 2.0 < right))
        if left > right:  # footprint crosses the antimeridian: warp every column
            cols = np.arange(lon.size)
        if rows.size and cols.size:
            r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
            window = np.full((r1 - r0, c1 - c0), FUEL_NODATA_OUT, dtype=np.int32)
            reproject(
                source=rasterio.band(src, 1),
                destination=window,
                src_nodata=nodata,
                # north-up rows: the window's top edge is the northern edge of its last cache row
                dst_transform=from_origin(lon[c0] - dlon / 2.0, lat[r1 - 1] + dlat / 2.0, dlon, dlat),
                dst_crs="EPSG:4326",
                dst_nodata=FUEL_NODATA_OUT,
                resampling=Resampling.mode,
                num_threads=int(num_threads),
            )
            fuel[r0:r1, c0:c1] = window[::-1]  # north-up rows -> the cache's south-to-north rows
    mask = (fuel >= 0) & ~np.isin(fuel, np.asarray(list(invalid_codes), dtype=np.int64))
    return np.where(mask, fuel, 0).astype(np.int32), mask.astype(np.uint8)


# ---------------------------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------------------------
def _date_range(start: str, end: str) -> List[str]:
    first, last = date.fromisoformat(str(start)), date.fromisoformat(str(end))
    return [str(first + timedelta(days=offset)) for offset in range((last - first).days + 1)]


def build_track_o_cache(
    cache_dir: PathLike,
    weather_files: Sequence[PathLike],
    firms_files: Sequence[PathLike],
    *,
    weather_vars: Sequence[str] = DEFAULT_WEATHER_VARS,
    splits: Mapping[str, Sequence[str]] = DEFAULT_SPLITS_2024,
    weather_date_pattern: str = DEFAULT_DATE_PATTERN,
    firms_types: Optional[Sequence[int]] = (0,),
    fuel_raster: Optional[PathLike] = None,
    fuel_nodata: Optional[float] = LANDFIRE_NODATA,
    limit_days: int = 0,
) -> Dict[str, object]:
    """Write a Track-O cache (layout in :mod:`pyhazards.datasets.wildfire.track_o`) and return a summary.

    A day inside the split ranges is written when it has weather files and FIRMS coverage: a CSV
    named after the day, or a date between the first and last detection of the CSVs (archive
    downloads span many days). A covered day without detections gets an all-zero label.
    ``fuel_raster`` adds the static fuel layer (needs ``pyhazards[geo]``); it can also be added
    later with :func:`add_fuel_to_cache`.
    """
    for name in SPLIT_NAMES:
        if name not in splits:
            raise ValueError(f"splits must define {SPLIT_NAMES}, missing '{name}'")
    wanted = {day: name for name in SPLIT_NAMES for day in _date_range(*splits[name])}
    if len(wanted) != sum(len(_date_range(*splits[name])) for name in SPLIT_NAMES):
        raise ValueError("split date ranges overlap")

    weather_by_day: Dict[str, List[Path]] = {}
    for path in sorted(Path(p) for p in weather_files):
        day = date_from_name(path, weather_date_pattern)
        if day is not None and day in wanted:
            weather_by_day.setdefault(day, []).append(path)

    firms_paths = sorted(Path(p) for p in firms_files)
    detections, firms_stats = read_firms_detections(firms_paths, firms_types=firms_types)
    per_day_files = {date_from_name(path) for path in firms_paths}
    covered = set(detections) | {day for day in per_day_files if day is not None}
    if detections:
        # acq_date-grouped downloads: every day between the first and last detection is covered.
        covered |= set(_date_range(min(detections), max(detections)))

    days = sorted(day for day in weather_by_day if day in covered)
    if limit_days > 0:
        days = days[: int(limit_days)]
    if not days:
        raise ValueError("no day has both weather files and FIRMS coverage inside the split ranges")

    root = Path(cache_dir)
    for sub in ("met", "labels", "static", "metadata", "splits"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    lat = lon = None
    positives = 0
    for day in days:
        fields, day_lat, day_lon = read_daily_weather(weather_by_day[day], weather_vars)
        if lat is None:
            lat, lon = day_lat, day_lon
            _regular_step(lat, "latitude")
            _regular_step(lon, "longitude")
        elif not (day_lat.shape == lat.shape and day_lon.shape == lon.shape and np.allclose(day_lat, lat) and np.allclose(day_lon, lon)):
            raise ValueError(f"weather files of {day} are on a different grid than {days[0]}")
        point_lat, point_lon = detections.get(day, (np.empty(0), np.empty(0)))
        label = rasterize_points(point_lat, point_lon, lat, lon)
        positives += int(label.sum())
        np.save(root / "met" / f"{day}.npy", fields)
        np.save(root / "labels" / f"{day}.npy", label)

    np.save(root / "metadata" / "lat.npy", lat.astype(np.float32))
    np.save(root / "metadata" / "lon.npy", lon.astype(np.float32))
    (root / "metadata" / "vars.json").write_text(json.dumps({"weather_vars": list(weather_vars)}, indent=2), encoding="utf-8")
    west, south, east, north = grid_bounds(lat, lon)
    (root / "metadata" / "grid.json").write_text(
        json.dumps(
            {
                "n_lat": int(lat.size),
                "n_lon": int(lon.size),
                "cell_centres": {"lat": [float(lat[0]), float(lat[-1])], "lon": [float(lon[0]), float(lon[-1])]},
                "bounds": {"west": west, "south": south, "east": east, "north": north},
                "row_order": "south_to_north",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    split_counts = {}
    for name in SPLIT_NAMES:
        split_days = [day for day in days if wanted[day] == name]
        (root / "splits" / f"{name}_dates.txt").write_text("\n".join(split_days), encoding="utf-8")
        split_counts[name] = len(split_days)

    summary: Dict[str, object] = {
        "cache_dir": str(root),
        "weather_vars": list(weather_vars),
        "days": len(days),
        "first_day": days[0],
        "last_day": days[-1],
        "splits": split_counts,
        "label_positive_cells": positives,
        "label_positive_fraction": positives / float(len(days) * lat.size * lon.size),
        "firms": firms_stats,
        "firms_types_kept": None if firms_types is None else [int(v) for v in firms_types],
        "static_fuel": None,
    }
    (root / "cache_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if fuel_raster is not None:
        summary["static_fuel"] = add_fuel_to_cache(root, fuel_raster, src_nodata=fuel_nodata)
    return summary


def add_fuel_to_cache(
    cache_dir: PathLike,
    fuel_raster: PathLike,
    *,
    src_nodata: Optional[float] = LANDFIRE_NODATA,
) -> Dict[str, object]:
    """Align ``fuel_raster`` to an existing cache's grid and write ``static/fuel.npy`` and ``fuel_mask.npy``."""
    root = Path(cache_dir)
    lat = np.load(root / "metadata" / "lat.npy").astype(np.float64)
    lon = np.load(root / "metadata" / "lon.npy").astype(np.float64)
    fuel, mask = align_fuel_to_grid(fuel_raster, lat, lon, src_nodata=src_nodata)
    (root / "static").mkdir(parents=True, exist_ok=True)
    np.save(root / "static" / "fuel.npy", fuel)
    np.save(root / "static" / "fuel_mask.npy", mask)
    info: Dict[str, object] = {
        "source": str(fuel_raster),
        "resampling": "mode",
        "valid_cells": int(mask.sum()),
        "valid_fraction": float(mask.mean()),
        "codes": sorted(int(code) for code in np.unique(fuel[mask > 0])),
    }
    summary_path = root / "cache_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["static_fuel"] = info
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return info


__all__ = [
    "DEFAULT_DATE_PATTERN",
    "add_fuel_to_cache",
    "align_fuel_to_grid",
    "build_track_o_cache",
    "date_from_name",
    "grid_bounds",
    "rasterize_points",
    "read_daily_weather",
    "read_firms_detections",
]
