"""Forecast-field conventions shared by the cyclone trackers and the forecast sources.

Trackers take an ``xarray.Dataset`` with dimensions ``(time, lat, lon)`` (``time`` = valid times,
``lat`` / ``lon`` in degrees, any orientation; longitudes are treated as periodic when they cover the
globe) and variables with these canonical names and SI units:

========================  =============================================  ==========
name                      quantity                                       unit
========================  =============================================  ==========
``msl``                   mean sea-level pressure                        Pa
``u10``, ``v10``          10 m wind components                           m s-1
``u<p>``, ``v<p>``        wind components at ``p`` hPa (e.g. ``u850``)   m s-1
``z<p>``                  geopotential at ``p`` hPa (e.g. ``z500``)      m2 s-2
``t<p>``                  temperature at ``p`` hPa                       K
``lsm``                   land-sea mask (1 = land), optional, static     1
========================  =============================================  ==========

:func:`standardize_fields` renames the variable names used by WeatherBench 2, earth2studio, ECMWF
``ai-models`` / TCBench outputs and ERA5 to these names and splits pressure-level variables.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "EARTH_RADIUS_KM",
    "SURFACE_ALIASES",
    "LEVEL_ALIASES",
    "TRACKER_VARIABLES",
    "great_circle_km",
    "relative_vorticity",
    "standardize_fields",
]

EARTH_RADIUS_KM = 6371.0
"""Sphere radius used for track distances (the PyHazards cyclone benchmark and TCBench use 6371 km)."""

SURFACE_ALIASES: Dict[str, str] = {
    # WeatherBench 2 / ERA5 long names
    "mean_sea_level_pressure": "msl",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "land_sea_mask": "lsm",
    # earth2studio
    "u10m": "u10",
    "v10m": "v10",
    # CF / ECMWF short names
    "MSL": "msl",
    "u10": "u10",
    "v10": "v10",
    "msl": "msl",
    "lsm": "lsm",
}

LEVEL_ALIASES: Dict[str, str] = {
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "geopotential": "z",
    "temperature": "t",
    "u": "u",
    "v": "v",
    "z": "z",
    "t": "t",
}

TRACKER_VARIABLES = {
    "tempest": ("msl", "u10", "v10", "z300", "z500"),
    "pangu": ("msl", "u10", "v10", "u850", "v850", "z200", "z850"),
    "graphcast": ("msl", "u10", "v10", "u200", "v200", "u500", "v500", "u700", "v700", "u850", "v850", "z200", "z850"),
}
"""Fields each tracker reads (``tempest`` = the TCBench TempestExtremes configuration)."""


def great_circle_km(lat1, lon1, lat2, lon2, radius_km: float = EARTH_RADIUS_KM) -> np.ndarray:
    """Haversine distance in km between points in degrees (numpy, broadcasting)."""
    lat1, lon1, lat2, lon2 = (np.deg2rad(np.asarray(v, dtype=np.float64)) for v in (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * radius_km * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def relative_vorticity(u: np.ndarray, v: np.ndarray, lat: Sequence[float], lon: Sequence[float], periodic: Optional[bool] = None) -> np.ndarray:
    """Relative vorticity (s-1) of ``(..., lat, lon)`` winds on a regular lat-lon grid.

    zeta = (dv/dlambda) / (a cos phi) - (du/dphi) / a + u tan(phi) / a with second-order centred
    differences (one-sided at the edges), a = 6371 km. Longitudes are periodic when they span the
    globe (or when ``periodic=True``). Values at the poles are set to NaN.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    if u.shape != v.shape or u.shape[-2:] != (lat.size, lon.size):
        raise ValueError(f"u and v must be shaped (..., {lat.size}, {lon.size}), got {u.shape} and {v.shape}")
    a = EARTH_RADIUS_KM * 1000.0
    phi = np.deg2rad(lat)[:, None]
    lam = np.deg2rad(lon)
    if periodic is None:
        step = np.median(np.abs(np.diff(lon))) if lon.size > 1 else 0.0
        periodic = lon.size > 2 and abs(lon.size * step - 360.0) < 1e-6
    if periodic:
        dlam = np.deg2rad(np.median(np.diff(lon)))
        dv_dlam = (np.roll(v, -1, axis=-1) - np.roll(v, 1, axis=-1)) / (2 * dlam)
    else:
        dv_dlam = np.gradient(v, lam, axis=-1)
    du_dphi = np.gradient(u, np.deg2rad(lat), axis=-2)
    with np.errstate(divide="ignore", invalid="ignore"):
        zeta = dv_dlam / (a * np.cos(phi)) - du_dphi / a + u * np.tan(phi) / a
    poles = np.isclose(np.abs(lat), 90.0)
    zeta[..., poles, :] = np.nan
    return zeta


def _level_name(prefix: str, level) -> str:
    value = float(level)
    if value.is_integer():
        value = int(value)
    return f"{prefix}{value}"


def standardize_fields(
    dataset,
    variables: Optional[Iterable[str]] = None,
    rename: Optional[Mapping[str, str]] = None,
    time_dim: Optional[str] = None,
):
    """Return an ``xarray.Dataset`` in the tracker conventions (see the module docstring).

    Accepts WeatherBench 2 (``latitude`` / ``longitude``, ``mean_sea_level_pressure``,
    ``u_component_of_wind`` with a ``level`` dimension), earth2studio (``u10m``, ``u850``) and ECMWF /
    TCBench short names (``msl``, ``u10``, ``z`` with ``level``). ``rename`` adds explicit
    source-to-canonical mappings, ``variables`` keeps only the listed canonical variables and
    ``time_dim`` names the valid-time dimension when it is not ``time`` / ``valid_time``.
    """
    import xarray as xr

    ds = dataset
    coord_names = {"latitude": "lat", "longitude": "lon"}
    ds = ds.rename({k: v for k, v in coord_names.items() if k in ds.dims or k in ds.coords})
    if time_dim is not None and time_dim != "time":
        ds = ds.rename({time_dim: "time"})
    elif "time" not in ds.dims and "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    if "lat" not in ds.dims or "lon" not in ds.dims:
        raise ValueError(f"dataset needs lat/lon (or latitude/longitude) dimensions, has {dict(ds.sizes)}")
    explicit = dict(rename or {})
    wanted = set(variables) if variables is not None else None
    out: Dict[str, "xr.DataArray"] = {}
    for name, array in ds.data_vars.items():
        if name in explicit:
            out[explicit[name]] = array
            continue
        level_dim = next((d for d in ("level", "pressure_level", "isobaricInhPa", "plev") if d in array.dims), None)
        if level_dim is not None and name in LEVEL_ALIASES:
            prefix = LEVEL_ALIASES[name]
            levels = array[level_dim].values
            if level_dim == "plev" and np.nanmax(levels) > 2000:  # Pa
                levels = levels / 100.0
            for k, level in enumerate(levels):
                key = _level_name(prefix, level)
                if wanted is None or key in wanted:
                    out[key] = array.isel({level_dim: k}).drop_vars(level_dim, errors="ignore")
            continue
        key = SURFACE_ALIASES.get(name, name)
        out[key] = array
    if wanted is not None:
        missing = sorted(wanted - set(out))
        if missing:
            raise KeyError(f"dataset lacks variables {missing} (found {sorted(out)})")
        out = {k: out[k] for k in sorted(wanted)}
    result = xr.Dataset(out)
    order = [d for d in ("time", "lat", "lon") if d in result.dims]
    return result.transpose(*order, ...)
