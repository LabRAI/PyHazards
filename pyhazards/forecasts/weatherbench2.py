"""Reader for the GraphCast and Pangu-Weather forecasts published by WeatherBench 2.

WeatherBench 2 (Rasp et al., JAMES 2024, doi:10.1029/2023MS004019) hosts global 0.25-degree
forecasts on Google Cloud Storage (public bucket ``gs://weatherbench2``, zarr v2):

* GraphCast (Lam et al. 2023), ERA5 initial conditions at 00/12 UTC: 2018 forecasts from the model
  trained on 1979-2017 (the paper's test-year model) and 2020 forecasts (trained to 2019);
  ``graphcast_hres_init`` holds GraphCast-operational 2020 forecasts from HRES analyses.
* Pangu-Weather (Bi et al. 2023), run by the WeatherBench 2 team with the official code and weights:
  2018-2022 from ERA5 (``pangu``) and 2020-2022 from HRES (``pangu_hres_init``).

Lead times are 0-234 h every 6 h. WeatherBench 2 does not host FourCastNet forecasts. The WeatherBench
2 data guide asks users to check each dataset's licence ("some datasets allow commercial use, others
only permit research use"); no licence file was found next to these forecasts, and the Pangu-Weather
forecasts derive from weights licensed CC BY-NC-SA 4.0, so treat them as non-commercial. PyHazards
reads them at run time and never redistributes them.

Each zarr chunk holds one initial time, one lead time and the whole globe (all pressure levels for
upper-air variables), so a pressure-level variable costs one full chunk per lead time even when only
one level is used (about 54 MB uncompressed for Pangu's 13 levels, 154 MB for GraphCast's 37). Reading
needs the optional packages ``zarr`` and ``gcsfs`` (``pip install pyhazards[weather]``).
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["WEATHERBENCH2_FORECASTS", "read_weatherbench2_forecast", "weatherbench2_url"]

_BUCKET = "gs://weatherbench2/datasets"

WEATHERBENCH2_FORECASTS: Dict[str, Dict[Union[int, str], str]] = {
    "graphcast": {
        2018: f"{_BUCKET}/graphcast/2018/date_range_2017-11-16_2019-02-01_12_hours.zarr",
        2020: f"{_BUCKET}/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours.zarr",
    },
    "graphcast_hres_init": {
        2020: f"{_BUCKET}/graphcast_hres_init/2020/date_range_2019-11-16_2021-02-01_12_hours.zarr",
    },
    "pangu": {"all": f"{_BUCKET}/pangu/2018-2022_0012_0p25.zarr"},
    "pangu_hres_init": {year: f"{_BUCKET}/pangu_hres_init/{year}_0012_0p25.zarr" for year in (2020, 2021, 2022)},
}
"""0.25-degree forecast stores by model and year (``all`` = one store for every year)."""

_SURFACE = {
    "msl": "mean_sea_level_pressure",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "t2m": "2m_temperature",
}
_LEVEL = {"u": "u_component_of_wind", "v": "v_component_of_wind", "z": "geopotential", "t": "temperature", "q": "specific_humidity"}


def weatherbench2_url(model: str, init_time) -> str:
    """Store holding ``model``'s forecast initialised at ``init_time``."""
    import pandas as pd

    if model not in WEATHERBENCH2_FORECASTS:
        raise ValueError(f"unknown WeatherBench 2 model {model!r}; choose from {sorted(WEATHERBENCH2_FORECASTS)}")
    stores = WEATHERBENCH2_FORECASTS[model]
    if "all" in stores:
        return stores["all"]
    stamp = pd.Timestamp(init_time)
    # The GraphCast stores span mid-November of the previous year to the end of January of the next.
    for year in (stamp.year, stamp.year + 1, stamp.year - 1):
        if year in stores:
            url = stores[year]
            if "date_range_" in url:
                start, end = url.split("date_range_")[1].split("_")[:2]
                if pd.Timestamp(start) <= stamp <= pd.Timestamp(end):
                    return url
            elif year == stamp.year:
                return url
    raise ValueError(f"no WeatherBench 2 {model} store covers {stamp}")


def _source_name(name: str) -> Tuple[str, Optional[int]]:
    if name in _SURFACE:
        return _SURFACE[name], None
    for prefix, long_name in _LEVEL.items():
        rest = name[len(prefix):]
        if name.startswith(prefix) and rest.isdigit():
            return long_name, int(rest)
    raise ValueError(f"cannot map {name!r} to a WeatherBench 2 variable (msl, u10, v10, t2m, u/v/z/t/q<hPa>)")


def _lead_hours(coord) -> list:
    """Lead times in hours whether or not xarray decoded ``prediction_timedelta`` (units: hours)."""
    values = np.asarray(coord.values)
    if np.issubdtype(values.dtype, np.timedelta64):
        return [int(v) for v in values / np.timedelta64(1, "h")]
    units = str(coord.attrs.get("units", "hours"))
    if units != "hours":
        raise ValueError(f"unexpected prediction_timedelta units {units!r}")
    return [int(v) for v in values]


def read_weatherbench2_forecast(
    model: str,
    init_time,
    variables: Sequence[str] = ("msl", "u10", "v10", "z300", "z500"),
    lead_hours: Optional[Iterable[int]] = None,
    lat_bounds: Optional[Tuple[float, float]] = None,
    lon_bounds: Optional[Tuple[float, float]] = None,
    url: Optional[str] = None,
    storage_options: Optional[dict] = None,
):
    """One forecast as an ``xarray.Dataset`` in the tracker conventions (``time`` = valid times).

    ``lead_hours`` defaults to the available 6-hourly leads up to 120 h (the Pangu-Weather stores start
    at 6 h, the GraphCast stores at 0 h). ``lat_bounds`` / ``lon_bounds`` (degrees, longitudes
    0-360 and possibly wrapping, e.g. ``(300, 30)``) crop the result after reading; they do not reduce
    the download (see the module docstring). ``url`` overrides the store (any path xarray's zarr
    backend opens, e.g. a local copy); ``gs://`` stores are read anonymously.
    """
    import pandas as pd
    import xarray as xr

    try:
        import zarr  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("reading WeatherBench 2 needs zarr (and gcsfs for gs:// stores): pip install pyhazards[weather]") from exc
    init = pd.Timestamp(init_time)
    store = url or weatherbench2_url(model, init)
    options = storage_options
    if options is None and str(store).startswith("gs://"):
        options = {"token": "anon"}
    ds = xr.open_zarr(store, consolidated=True, chunks=None, storage_options=options)
    ds = ds.rename({k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items() if k in ds.dims})
    if init not in pd.to_datetime(ds["time"].values):
        raise KeyError(f"{store} has no forecast initialised at {init}")
    available = _lead_hours(ds["prediction_timedelta"])
    if lead_hours is None:
        leads = [h for h in available if 0 <= h <= 120 and h % 6 == 0]
    else:
        leads = [int(h) for h in lead_hours]
        absent = sorted(set(leads) - set(available))
        if absent:
            raise KeyError(f"{store} has no lead times {absent} h (available: {available[:3]}...{available[-1]})")
    ds = ds.sel(time=init.to_datetime64()).isel(prediction_timedelta=[available.index(h) for h in leads])
    lat_sel = slice(None)
    lat = ds["lat"].values
    if lat_bounds is not None:
        lo, hi = sorted(lat_bounds)
        lat_sel = np.flatnonzero((lat >= lo) & (lat <= hi))
    lon_sel = slice(None)
    lon = ds["lon"].values % 360.0
    if lon_bounds is not None:
        west, east = (b % 360.0 for b in lon_bounds)
        mask = (lon >= west) & (lon <= east) if west <= east else (lon >= west) | (lon <= east)
        lon_sel = np.flatnonzero(mask)
    data = {}
    for name in variables:
        source, level = _source_name(name)
        if source not in ds:
            raise KeyError(f"{store} has no variable {source!r}")
        array = ds[source]
        if level is not None:
            array = array.sel(level=level)
        array = array.transpose("prediction_timedelta", "lat", "lon").isel(lat=lat_sel, lon=lon_sel)
        data[name] = (("time", "lat", "lon"), np.asarray(array.values, dtype=np.float32))
    out_lat = lat if isinstance(lat_sel, slice) else lat[lat_sel]
    out_lon = ds["lon"].values if isinstance(lon_sel, slice) else ds["lon"].values[lon_sel]
    times = [init + pd.Timedelta(hours=h) for h in leads]
    result = xr.Dataset(data, coords={"time": times, "lat": out_lat.astype(np.float64), "lon": out_lon.astype(np.float64)})
    result.attrs.update({"source": str(store), "model": model, "init_time": str(init)})
    return result
