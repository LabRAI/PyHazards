"""Hurricast inputs: IBTrACS statistical features and ERA5 reanalysis maps (Boussioux et al. 2022).

Hurricast (Weather and Forecasting 37(6), 2022, doi:10.1175/WAF-D-21-0091.1) forecasts the 24-hour
intensity and displacement of a tropical cyclone from its last 8 three-hourly time steps (t-21 h ... t):

* **Statistical data** (paper Section 2a, Table 2) from IBTrACS: per step the 30 features of
  :data:`pyhazards.models.hurricast.HURRICAST_STAT_FEATURES` (position, WMO wind and pressure, distance
  to land, translation speed, cosine / sine of the day of year, direction, latitude and longitude,
  Saffir-Simpson category value, one-hot basin and storm nature, and the latitude / longitude change
  since the previous step). WMO wind and pressure, which IBTrACS reports only at synoptic times, are
  interpolated linearly to every 3-hour step, and 10-minute winds are converted to 1-minute winds by
  dividing by 0.93 (samples outside the North Atlantic and Eastern Pacific, as in the official notebooks).
* **Reanalysis maps** (Section 2b) from ERA5: u, v and geopotential z at 225, 500 and 700 hPa on a
  25 x 25 grid of 1 degree centred on the storm, per step.

Storms are selected as in the official ``sort_storm`` (src/utils/data_processing.py of leobix/hurricast;
read as a test oracle only, the repository has no licence): a storm is kept when more than ``min_steps``
of its rows have a WMO wind of at least ``min_wind`` knots (34 kt and 20 rows, i.e. 60 h, in the paper),
and its rows from the first time it reaches ``min_wind`` on (at most ``max_steps``, 120) form its track.
Every window of ``window_size`` steps followed by ``predict_at`` steps gives one sample; as the official
``remove_zeros``, windows whose 24-hour target period contains a step with zero latitude and longitude
change are dropped. Targets: WMO wind ``predict_at`` steps after the last input step (24 h), or the
latitude / longitude 24 h ahead (the model predicts the displacement from the current position).

PyHazards reads IBTrACS v04 (:func:`pyhazards.datasets.tc.read_ibtracs`) and ERA5 pressure-level data
the user has (a netCDF / zarr file or an ``xarray.Dataset`` with u, v, z at 225, 500, 700 hPa, e.g. a
Copernicus Climate Data Store download or the NCAR RDA copy); nothing is downloaded or redistributed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ...models.hurricast import HURRICAST_MAP_CHANNELS, HURRICAST_STAT_FEATURES
from ..base import DataBundle, DataSplit, Dataset, FeatureSpec, LabelSpec
from .ibtracs import read_ibtracs

ERA5_LEVELS: Tuple[int, ...] = (225, 500, 700)
ERA5_VARIABLES: Tuple[str, ...] = ("u", "v", "z")
# Paper Section 4b: train 1980-2011, validation 2012-2015, test 2016-2019 (test: NA and EP only).
PAPER_SEASONS = {"train": tuple(range(1980, 2012)), "val": tuple(range(2012, 2016)), "test": tuple(range(2016, 2020))}
PAPER_TEST_BASINS = ("NA", "EP")
BASIN_COLUMNS = (("NA", "cat_basin_AN"), ("EP", "cat_basin_EP"), ("NI", "cat_basin_NI"), ("SA", "cat_basin_SA"), ("SI", "cat_basin_SI"), ("SP", "cat_basin_SP"), ("WP", "cat_basin_WP"))
NATURE_COLUMNS = tuple((code, f"cat_nature_{code}") for code in ("DS", "ET", "MX", "NR", "SS", "TS"))
# Table 2: these numerical features are standardised; cyclic encodings and categories are not.
STANDARDISED_FEATURES = ("LAT", "LON", "WMO_WIND", "WMO_PRES", "DIST2LAND", "STORM_SPEED", "STORM_DISPLACEMENT_X", "STORM_DISPLACEMENT_Y")
TEN_MINUTE_TO_ONE_MINUTE = 0.93  # Harper et al. (2010), as in the paper

_ERA5_NAMES = {
    "u": ("u", "U", "u_component_of_wind"),
    "v": ("v", "V", "v_component_of_wind"),
    "z": ("z", "Z", "geopotential"),
}
_DIM_NAMES = {
    "time": ("time", "valid_time"),
    "level": ("level", "pressure_level", "isobaricInhPa", "plev"),
    "latitude": ("latitude", "lat"),
    "longitude": ("longitude", "lon"),
}


def wind_category(wind: Union[float, np.ndarray]) -> np.ndarray:
    """Saffir-Simpson category value of the official ``sust_wind_to_cat_val`` (knots): TD 0, TS 1, H1-H5 2-6, missing 7."""
    wind = np.asarray(wind, dtype=np.float64)
    edges = np.array([33.0, 63.0, 82.0, 95.0, 112.0, 136.0])
    category = np.searchsorted(edges, wind, side="left").astype(np.float64)
    return np.where(np.isnan(wind), 7.0, category)


def _storm_features(storm) -> "Any":
    """The 30 features of one storm's rows (pandas DataFrame in, DataFrame out)."""
    import pandas as pd

    out = pd.DataFrame(index=storm.index)
    out["LAT"] = storm["LAT"].astype(float)
    out["LON"] = storm["LON"].astype(float)
    out["WMO_WIND"] = storm["WMO_WIND"].astype(float)
    out["WMO_PRES"] = storm["WMO_PRES"].astype(float)
    out["DIST2LAND"] = storm["DIST2LAND"].astype(float)
    out["STORM_SPEED"] = storm["STORM_SPEED"].astype(float)
    day = storm["ISO_TIME"].dt.dayofyear.to_numpy(dtype=np.float64)
    out["cat_cos_day"] = np.cos(2 * np.pi * day / 365)
    out["cat_sign_day"] = np.sin(2 * np.pi * day / 365)
    direction = np.deg2rad(storm["STORM_DIR"].to_numpy(dtype=np.float64))
    out["COS_STORM_DIR"] = np.cos(direction)
    out["SIN_STORM_DIR"] = np.sin(direction)
    lat, lon = np.deg2rad(out["LAT"].to_numpy()), np.deg2rad(out["LON"].to_numpy())
    out["COS_LAT"], out["SIN_LAT"] = np.cos(lat), np.sin(lat)
    out["COS_LON"], out["SIN_LON"] = np.cos(lon), np.sin(lon)
    out["cat_storm_category"] = wind_category(out["WMO_WIND"].to_numpy())
    basin = storm["BASIN"].astype(str).to_numpy()
    for code, column in BASIN_COLUMNS:
        out[column] = (basin == code).astype(np.float64)
    nature = storm["NATURE"].astype(str).to_numpy() if "NATURE" in storm else np.full(len(storm), "")
    for code, column in NATURE_COLUMNS:
        out[column] = (nature == code).astype(np.float64)
    # Change since the previous step; 0 for the first row of the (cut) track, as the official
    # add_displacement_lat_lon2.
    out["STORM_DISPLACEMENT_X"] = out["LAT"].diff().fillna(0.0)
    out["STORM_DISPLACEMENT_Y"] = out["LON"].diff().fillna(0.0)
    return out[list(HURRICAST_STAT_FEATURES)]


def hurricast_storms(
    table,
    min_wind: float = 34.0,
    min_steps: int = 20,
    max_steps: int = 120,
    include_spur: bool = False,
) -> List[Dict[str, Any]]:
    """Select and cut storms like the official ``sort_storm`` and compute their statistical features.

    ``table`` is an IBTrACS table (:func:`pyhazards.datasets.tc.read_ibtracs`). Rows off the 3-hour
    grid (e.g. landfall records at 10:15) are dropped; WMO wind and pressure are interpolated linearly
    within each storm (values after the last report repeat it). Returns one dict per kept storm with
    ``sid``, ``season``, ``basin`` (first row), ``times`` and ``features`` ``(rows, 30)``.
    """
    required = ("SID", "SEASON", "BASIN", "ISO_TIME", "LAT", "LON", "WMO_WIND", "WMO_PRES", "DIST2LAND", "STORM_SPEED", "STORM_DIR")
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"the IBTrACS table lacks column(s) {missing} needed by Hurricast")
    frame = table
    times = frame["ISO_TIME"]
    frame = frame[(times.dt.hour % 3 == 0) & (times.dt.minute == 0) & (times.dt.second == 0)]
    if not include_spur and "TRACK_TYPE" in frame.columns:
        frame = frame[~frame["TRACK_TYPE"].astype(str).str.contains("spur", case=False, na=False)]
    storms = []
    for sid, storm in frame.groupby("SID", sort=True):
        storm = storm.sort_values("ISO_TIME", kind="stable").reset_index(drop=True)
        storm = storm.copy()
        for column in ("WMO_WIND", "WMO_PRES"):
            storm[column] = storm[column].astype(float).interpolate(method="linear")
        strong = np.flatnonzero(storm["WMO_WIND"].to_numpy() >= min_wind)
        if len(strong) <= min_steps:
            continue
        cut = storm.iloc[strong[0] : strong[0] + int(max_steps)].reset_index(drop=True)
        storms.append(
            {
                "sid": sid,
                "season": int(cut["SEASON"].iloc[0]),
                "basin": str(cut["BASIN"].iloc[0]),
                "times": cut["ISO_TIME"].to_numpy(dtype="datetime64[ns]"),
                "features": _storm_features(cut).to_numpy(dtype=np.float64),
            }
        )
    return storms


def hurricast_windows(
    storms: Sequence[Mapping[str, Any]],
    window_size: int = 8,
    predict_at: int = 8,
    one_minute_winds: bool = True,
) -> Dict[str, Any]:
    """Forecast samples from :func:`hurricast_storms`.

    Returns ``x_stat`` ``(N, window_size, 30)``, ``position`` ``(N, 2)`` (latitude, longitude at the
    last input step), ``intensity`` ``(N,)`` (WMO wind ``predict_at`` steps later), ``displacement``
    ``(N, 2)`` (sum of the latitude / longitude changes over those steps), and per-sample ``sid``,
    ``season``, ``basin``, ``iso_time`` and ``rows`` (index of each input step in its storm).
    """
    names = list(HURRICAST_STAT_FEATURES)
    i_lat, i_lon, i_wind = names.index("LAT"), names.index("LON"), names.index("WMO_WIND")
    i_dx, i_dy = names.index("STORM_DISPLACEMENT_X"), names.index("STORM_DISPLACEMENT_Y")
    i_na, i_ep = names.index("cat_basin_AN"), names.index("cat_basin_EP")
    out: Dict[str, List[Any]] = {key: [] for key in ("x_stat", "position", "intensity", "displacement", "sid", "season", "basin", "iso_time", "storm", "rows")}
    for k, storm in enumerate(storms):
        values = storm["features"]
        for start in range(0, len(values) - window_size - predict_at + 1):
            last = start + window_size - 1
            future = values[last + 1 : last + 1 + predict_at]
            if np.any((future[:, i_dx] == 0) & (future[:, i_dy] == 0)):
                continue  # official remove_zeros
            x = values[start : last + 1].copy()
            target = values[last + predict_at, i_wind]
            if one_minute_winds and x[0, i_na] + x[0, i_ep] < 1:
                x[:, i_wind] = x[:, i_wind] / TEN_MINUTE_TO_ONE_MINUTE
                target = target / TEN_MINUTE_TO_ONE_MINUTE
            out["x_stat"].append(x)
            out["position"].append(values[last, [i_lat, i_lon]])
            out["intensity"].append(target)
            out["displacement"].append([future[:, i_dx].sum(), future[:, i_dy].sum()])
            out["sid"].append(storm["sid"])
            out["season"].append(storm["season"])
            out["basin"].append(storm["basin"])
            out["iso_time"].append(str(np.datetime_as_string(storm["times"][last], unit="s")))
            out["storm"].append(k)
            out["rows"].append(list(range(start, last + 1)))
    width = len(names)
    return {
        "x_stat": np.asarray(out["x_stat"], dtype=np.float64).reshape(-1, window_size, width),
        "position": np.asarray(out["position"], dtype=np.float64).reshape(-1, 2),
        "intensity": np.asarray(out["intensity"], dtype=np.float64),
        "displacement": np.asarray(out["displacement"], dtype=np.float64).reshape(-1, 2),
        **{key: out[key] for key in ("sid", "season", "basin", "iso_time", "storm", "rows")},
    }


# ---------------------------------------------------------------------------------------------------
# ERA5


def _rename(ds):
    """Rename an ERA5 dataset to dims (time, level, latitude, longitude) and variables u, v, z."""
    renames = {}
    for target, options in _DIM_NAMES.items():
        found = next((name for name in options if name in ds.dims or name in ds.coords), None)
        if found is None:
            raise ValueError(f"ERA5 data has no {target} coordinate (looked for {options})")
        if found != target:
            renames[found] = target
    for target, options in _ERA5_NAMES.items():
        found = next((name for name in options if name in ds.data_vars), None)
        if found is None:
            raise ValueError(f"ERA5 data has no {target} variable (looked for {options})")
        if found != target:
            renames[found] = target
    return ds.rename(renames) if renames else ds


def open_era5(source: Any):
    """Open ERA5 pressure-level data (netCDF file, zarr store or ``xarray.Dataset``) with u, v, z."""
    import xarray as xr

    if isinstance(source, xr.Dataset):
        ds = source
    else:
        path = Path(source).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"ERA5 data {path} not found")
        ds = xr.open_zarr(path) if path.suffix == ".zarr" or (path / ".zgroup").exists() else xr.open_dataset(path)
    return _rename(ds)


def era5_maps(
    ds,
    times: Sequence[Any],
    lats: Sequence[float],
    lons: Sequence[float],
    size: int = 25,
    resolution: float = 1.0,
    levels: Sequence[int] = ERA5_LEVELS,
) -> np.ndarray:
    """Storm-centred ERA5 maps ``(n, 9, size, size)``: u, v, z at ``levels``, north to south, west to east.

    The centre is the grid point (at ``resolution`` degrees) nearest to the storm; the maps take every
    grid point within ``size // 2`` steps of it, so a 0.25-degree source is sampled at whole degrees.
    """
    ds = _rename(ds)
    lon_coord = ds["longitude"].values
    east_360 = float(np.nanmax(lon_coord)) > 180.0
    offsets = (np.arange(size) - size // 2) * resolution
    maps = np.empty((len(times), len(ERA5_VARIABLES) * len(levels), size, size), dtype=np.float32)
    tol = resolution / 100.0
    for i, (time, lat, lon) in enumerate(zip(times, lats, lons)):
        centre_lat = math.floor(float(lat) / resolution + 0.5) * resolution
        centre_lon = math.floor(float(lon) / resolution + 0.5) * resolution
        wanted_lat = centre_lat - offsets  # north to south
        wanted_lon = centre_lon + offsets
        wanted_lon = np.mod(wanted_lon, 360.0) if east_360 else np.mod(wanted_lon + 180.0, 360.0) - 180.0
        try:
            step = ds[list(ERA5_VARIABLES)].sel(time=np.datetime64(time, "ns"), level=list(levels))
            step = step.sel(latitude=wanted_lat, longitude=wanted_lon, method="nearest", tolerance=tol)
        except KeyError as exc:
            raise KeyError(f"ERA5 data lacks the maps around ({lat}, {lon}) at {time}: {exc}") from exc
        for v, name in enumerate(ERA5_VARIABLES):
            block = step[name].transpose("level", "latitude", "longitude").values
            maps[i, v * len(levels) : (v + 1) * len(levels)] = block
    if not np.isfinite(maps).all():
        raise ValueError("ERA5 maps contain missing values (outside the data's area or time range?)")
    return maps


# ---------------------------------------------------------------------------------------------------
# Dataset


class HurricastDataset(Dataset):
    """24-hour Hurricast samples from an IBTrACS file and (optionally) ERA5 pressure-level data.

    Inputs per split: ``{"x_stat": (N, 8, 30), "x_viz": (N, 8, 9, 25, 25), "position": (N, 2)}`` (no
    ``x_viz`` without ``era5``; HUML-(stat, xgb) needs none). With ``standardize`` the numerical
    statistical features of Table 2 and every map channel are standardised with the training split's
    mean and standard deviation (over samples and steps, and pixels for maps); ``position`` stays in
    degrees. Targets: ``target="intensity"`` the WMO wind 24 h ahead ``(N,)`` in 1-minute knots
    (``tc.intensity``); ``target="displacement"`` the latitude / longitude 24 h ahead ``(N, 1, 2)``
    (``tc.track_intensity``, lead 24 h). Splits are by season (paper: 1980-2011 / 2012-2015 /
    2016-2019); the test split keeps only ``test_basins`` (paper: NA and EP); ``basins`` restricts every
    split. ERA5 maps are read only for samples of a split.
    """

    name = "hurricast_ibtracs_era5"

    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        era5: Any = None,
        cache_dir: Optional[str] = None,
        target: str = "intensity",
        window_size: int = 8,
        predict_at: int = 8,
        min_wind: float = 34.0,
        min_steps: int = 20,
        max_steps: int = 120,
        train_seasons: Optional[Sequence[int]] = PAPER_SEASONS["train"],
        val_seasons: Optional[Sequence[int]] = PAPER_SEASONS["val"],
        test_seasons: Optional[Sequence[int]] = PAPER_SEASONS["test"],
        test_basins: Optional[Sequence[str]] = PAPER_TEST_BASINS,
        basins: Optional[Sequence[str]] = None,
        standardize: bool = True,
        one_minute_winds: bool = True,
        include_spur: bool = False,
    ):
        super().__init__(cache_dir=cache_dir)
        if path is None:
            raise ValueError("hurricast_ibtracs_era5 needs path= to an IBTrACS v04 CSV or netCDF file (see ibtracs_tracks)")
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} not found")
        if target not in ("intensity", "displacement"):
            raise ValueError("target must be 'intensity' or 'displacement'")
        if window_size < 1 or predict_at < 1:
            raise ValueError("window_size and predict_at must be positive")
        self.era5 = era5
        self.target = target
        self.window_size, self.predict_at = int(window_size), int(predict_at)
        self.min_wind, self.min_steps, self.max_steps = float(min_wind), int(min_steps), int(max_steps)
        self.seasons = {"train": train_seasons, "val": val_seasons, "test": test_seasons}
        self.test_basins = None if test_basins is None else tuple(b.upper() for b in test_basins)
        self.basins = None if basins is None else tuple(b.upper() for b in basins)
        self.standardize = standardize
        self.one_minute_winds = one_minute_winds
        self.include_spur = include_spur

    def _load(self) -> DataBundle:
        columns = ("NATURE", "TRACK_TYPE", "WMO_WIND", "WMO_PRES", "DIST2LAND", "STORM_SPEED", "STORM_DIR")
        storms = hurricast_storms(read_ibtracs(self.path, columns), self.min_wind, self.min_steps, self.max_steps, self.include_spur)
        samples = hurricast_windows(storms, self.window_size, self.predict_at, self.one_minute_winds)
        if len(samples["intensity"]) == 0:
            raise ValueError(f"no Hurricast samples in {self.path} (min_wind={self.min_wind}, min_steps={self.min_steps})")
        seasons = np.asarray(samples["season"])
        basins = np.asarray(samples["basin"])
        keep_basin = np.ones(len(seasons), dtype=bool) if self.basins is None else np.isin(basins, self.basins)
        masks = {}
        for split, years in self.seasons.items():
            mask = np.zeros(len(seasons), dtype=bool) if years is None else np.isin(seasons, list(years))
            mask &= keep_basin
            if split == "test" and self.test_basins is not None:
                mask &= np.isin(basins, self.test_basins)
            masks[split] = mask
        if not any(mask.any() for mask in masks.values()):
            raise ValueError("no Hurricast sample falls into the train / val / test seasons and basins")

        x_stat = samples["x_stat"]
        x_viz = None
        if self.era5 is not None:
            # Maps are read only for samples that belong to a split.
            ds = open_era5(self.era5)
            wanted = np.flatnonzero(np.logical_or.reduce(list(masks.values())))
            used = sorted({(samples["storm"][i], r) for i in wanted for r in samples["rows"][i]})
            i_lat, i_lon = HURRICAST_STAT_FEATURES.index("LAT"), HURRICAST_STAT_FEATURES.index("LON")
            crops = era5_maps(
                ds,
                [storms[k]["times"][r] for k, r in used],
                [storms[k]["features"][r, i_lat] for k, r in used],
                [storms[k]["features"][r, i_lon] for k, r in used],
            )
            where = {key: i for i, key in enumerate(used)}
            blank = np.zeros_like(crops[0])  # samples outside every split are never returned
            x_viz = np.stack([[crops[where[(k, r)]] if (k, r) in where else blank for r in rows] for k, rows in zip(samples["storm"], samples["rows"])])

        stat_mean = np.zeros(x_stat.shape[-1])
        stat_std = np.ones(x_stat.shape[-1])
        viz_mean = viz_std = None
        train = masks["train"]
        if self.standardize:
            if not train.any():
                raise ValueError("standardize=True needs training samples (train_seasons)")
            for name in STANDARDISED_FEATURES:
                j = HURRICAST_STAT_FEATURES.index(name)
                values = torch.as_tensor(x_stat[train][:, :, j])
                stat_mean[j], stat_std[j] = float(values.mean()), float(values.std()) or 1.0
            x_stat = (x_stat - stat_mean) / stat_std
            if x_viz is not None:
                maps = torch.as_tensor(x_viz[train], dtype=torch.float32)
                viz_mean = maps.mean(dim=(0, 1, 3, 4)).numpy()
                viz_std = maps.std(dim=(0, 1, 3, 4)).numpy()
                viz_std = np.where(viz_std > 0, viz_std, 1.0)
                x_viz = ((x_viz - viz_mean[None, None, :, None, None]) / viz_std[None, None, :, None, None]).astype(np.float32)

        position = samples["position"]
        if self.target == "intensity":
            targets = samples["intensity"]
        else:
            targets = (position + samples["displacement"])[:, None, :]
        splits = {}
        for split, mask in masks.items():
            index = np.flatnonzero(mask)
            inputs = {"x_stat": torch.as_tensor(x_stat[index], dtype=torch.float32), "position": torch.as_tensor(position[index], dtype=torch.float32)}
            if x_viz is not None:
                inputs["x_viz"] = torch.as_tensor(x_viz[index])
            splits[split] = DataSplit(
                inputs,
                torch.as_tensor(targets[index], dtype=torch.float32),
                metadata={
                    "groups": [samples["season"][i] for i in index],
                    "sid": [samples["sid"][i] for i in index],
                    "basin": [samples["basin"][i] for i in index],
                    "iso_time": [samples["iso_time"][i] for i in index],
                },
            )
        metadata = {
            "dataset": self.name,
            "source_dataset": "IBTrACS v04" + (" + ERA5" if x_viz is not None else ""),
            "lead_hours": [3 * self.predict_at],
            "window_size": self.window_size,
            "stat_features": list(HURRICAST_STAT_FEATURES),
            "map_channels": list(HURRICAST_MAP_CHANNELS) if x_viz is not None else [],
            "standardization": {
                "stat_mean": stat_mean.tolist(),
                "stat_std": stat_std.tolist(),
                "map_mean": None if viz_mean is None else viz_mean.tolist(),
                "map_std": None if viz_std is None else viz_std.tolist(),
            },
            "source_file": str(self.path),
            "synthetic": False,
        }
        if self.target == "intensity":
            metadata.update({"hazard_task": "tc.intensity", "units": {"wind": "kt"}, "intensity_target": "1-minute WMO wind 24 h ahead"})
            label = LabelSpec(num_targets=1, task_type="regression", description="Maximum sustained wind 24 hours ahead (1-minute, kt).")
        else:
            metadata.update({"hazard_task": "tc.track_intensity", "target_variables": ["lat", "lon"], "units": {}})
            label = LabelSpec(num_targets=2, task_type="regression", description="Latitude and longitude 24 hours ahead (degrees).")
        return DataBundle(
            splits=splits,
            feature_spec=FeatureSpec(
                input_dim=len(HURRICAST_STAT_FEATURES),
                description=f"{self.window_size} three-hourly steps of 30 IBTrACS statistical features" + (" and 9 ERA5 maps (25 x 25)" if x_viz is not None else ""),
                extra={"window_size": self.window_size},
            ),
            label_spec=label,
            metadata=metadata,
        )


__all__ = [
    "ERA5_LEVELS",
    "ERA5_VARIABLES",
    "HurricastDataset",
    "PAPER_SEASONS",
    "STANDARDISED_FEATURES",
    "era5_maps",
    "hurricast_storms",
    "hurricast_windows",
    "open_era5",
    "wind_category",
]
