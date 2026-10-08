"""ECMWF-style "following" cyclone tracker, as described in the Pangu-Weather and GraphCast papers.

Both papers track a known cyclone through a deterministic forecast, starting from its observed
position at the initial time, with the rules of ECMWF's tracker (White 2005, ECMWF Newsletter 102;
van der Grijn 2002, ECMWF Tech. Memo. 386). Neither paper released its tracker code, and ECMWF's
operational tracker is not public, so this is a re-implementation of the published descriptions:

* Bi et al., "Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather
  Forecast", arXiv:2211.02556, Sec. 4.2.2 (the Nature paper, doi:10.1038/s41586-023-06185-3, cites it
  for the details): from the cyclone's position, look 6 h later for a local minimum of mean sea-level
  pressure (MSLP) within 445 km such that (1) there is a maximum of 850 hPa relative vorticity larger
  than 5e-5 s-1 within 278 km (a minimum smaller than -5e-5 s-1 in the Southern Hemisphere; the
  report prints "smaller than 5e-5", a dropped minus sign), (2) there is a maximum of the 850-200 hPa
  thickness within 278 km when the cyclone is extratropical, (3) the maximum 10 m wind speed within
  278 km exceeds 8 m/s when the cyclone is on land. Tracking stops when no minimum qualifies.
* Lam et al., "Learning skillful medium-range global weather forecasting", Science 382:1416 (2023),
  supplementary Sec. 8.1.4: the same three checks (radii 278 km), candidates are all MSLP local minima
  within 445 km of a first guess, the closest qualifying candidate is taken, and the first guess moves
  the current position by the average of the last displacement (linear extrapolation) and the wind
  steering (u, v averaged over 200, 500, 700 and 850 hPa at the current position). GraphCast's
  modified tracker, chosen by a hyper-parameter search (selected values printed in bold in the
  supplement): candidate radius 445 x 0.5 = 222.5 km, check radius 278 x 0.75 = 208.5 km, steering
  weight 0.5, candidates must not turn the track by 90 degrees or more, no clipping of the first-guess
  displacement. When a cyclone disappears the track stops (ECMWF's brief-disappearance rule is not
  used).

Presets: :data:`PANGU_TRACKER` (search around the current position, no first guess),
:data:`ECMWF_TRACKER` (GraphCast's description of the unmodified ECMWF rules) and
:data:`GRAPHCAST_TRACKER` (GraphCast's modified tracker).

What the papers leave open, and what this module does (each is a parameter):

* "Extratropical" is not defined: the thickness check is applied only when
  ``extratropical_latitude`` is set, poleward of that absolute latitude (default: never). Thickness
  is ``z200 - z850`` (GraphCast's text writes ``z850 - z200``, whose maximum would be a cold core) and
  "a maximum" means a grid-point local maximum (3 x 3 neighbourhood) inside the radius.
* "On land" needs a land-sea mask: pass an ``lsm`` field (land where ``lsm >= 0.5``); without it every
  candidate is treated as over sea.
* MSLP local minima are grid points not larger than any of their 8 neighbours; positions are grid
  points (no sub-grid interpolation); steering winds are read from the fields at the current time, at
  the grid point nearest the current position (for the first step from a start time that is not in
  the fields, from the first forecast step); vorticity uses centred differences on the sphere
  (:func:`~pyhazards.forecasts.fields.relative_vorticity`). Among the candidates, the one closest to
  the first guess (Pangu-Weather: to the current position) that passes the checks is taken.
* The first step has no previous displacement unless ``previous_position`` is given (e.g. the
  best-track position 6 h before the initial time); the first guess then uses the steering alone.
* If the fields include the initial time, ``relocate_start`` moves the start to the qualifying
  minimum nearest the observed position (ECMWF verifies the cyclone in the analysis); Pangu's
  description starts from the observed position itself.

Intensity outputs, which neither paper scores, are the MSLP at the centre and the maximum 10 m wind
speed within ``intensity_radius_km`` (default: the check radius).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .fields import EARTH_RADIUS_KM, great_circle_km, relative_vorticity

__all__ = [
    "ECMWF_TRACKER",
    "FollowingTrackerConfig",
    "GRAPHCAST_TRACKER",
    "PANGU_TRACKER",
    "follow_cyclone",
    "required_variables",
    "with_overrides",
]

KM_PER_DEGREE = math.pi * EARTH_RADIUS_KM / 180.0


@dataclass(frozen=True)
class FollowingTrackerConfig:
    """Parameters of :func:`follow_cyclone` (distances in km, vorticity in s-1, winds in m/s)."""

    name: str = "ecmwf"
    search_radius_km: float = 445.0
    check_radius_km: float = 278.0
    vorticity_threshold: float = 5e-5
    vorticity_level: int = 850
    land_wind_threshold: float = 8.0
    use_first_guess: bool = True
    steering_weight: float = 0.5
    steering_levels: Tuple[int, ...] = (200, 500, 700, 850)
    max_turn_deg: Optional[float] = None
    max_first_guess_km: Optional[float] = None
    extratropical_latitude: Optional[float] = None
    thickness_levels: Tuple[int, int] = (200, 850)
    relocate_start: bool = True
    intensity_radius_km: Optional[float] = None

    def __post_init__(self) -> None:
        if self.search_radius_km <= 0 or self.check_radius_km <= 0:
            raise ValueError("search and check radii must be positive")
        if not 0.0 <= self.steering_weight <= 1.0:
            raise ValueError("steering_weight must be in [0, 1]")
        if self.max_turn_deg is not None and not 0 < self.max_turn_deg <= 180:
            raise ValueError("max_turn_deg must be in (0, 180]")


PANGU_TRACKER = FollowingTrackerConfig(
    name="pangu", use_first_guess=False, steering_weight=0.0, steering_levels=(), relocate_start=False
)
"""Bi et al. (2022) Sec. 4.2.2: minima within 445 km of the current position, checks within 278 km."""

ECMWF_TRACKER = FollowingTrackerConfig(name="ecmwf")
"""ECMWF rules as summarised in the GraphCast supplement (445 / 278 km, 50-50 first guess)."""

GRAPHCAST_TRACKER = FollowingTrackerConfig(
    name="graphcast", search_radius_km=445.0 * 0.5, check_radius_km=278.0 * 0.75, steering_weight=0.5, max_turn_deg=90.0
)
"""GraphCast's modified ECMWF tracker (Science 2023 supplement, Sec. 8.1.4, bold values)."""


def required_variables(config: FollowingTrackerConfig) -> Tuple[str, ...]:
    """Fields :func:`follow_cyclone` reads for ``config`` (``lsm`` is optional and not listed)."""
    names = ["msl", "u10", "v10", f"u{config.vorticity_level}", f"v{config.vorticity_level}"]
    if config.use_first_guess and config.steering_weight > 0:
        for level in config.steering_levels:
            names += [f"u{level}", f"v{level}"]
    if config.extratropical_latitude is not None:
        names += [f"z{level}" for level in config.thickness_levels]
    return tuple(dict.fromkeys(names))


class _Frame:
    """One time step of the fields with cached derived quantities."""

    def __init__(self, ds, t: int, lat: np.ndarray, lon: np.ndarray, config: FollowingTrackerConfig, periodic: bool):
        self.ds, self.t, self.lat, self.lon, self.config, self.periodic = ds, t, lat, lon, config, periodic
        self._cache: Dict[str, np.ndarray] = {}

    def field(self, name: str) -> np.ndarray:
        if name not in self._cache:
            array = self.ds[name]
            if "time" in array.dims:
                array = array.isel(time=self.t)
            self._cache[name] = np.asarray(array.values, dtype=np.float64)
        return self._cache[name]

    def has(self, name: str) -> bool:
        return name in self.ds.data_vars

    def vorticity(self) -> np.ndarray:
        if "vort" not in self._cache:
            level = self.config.vorticity_level
            self._cache["vort"] = relative_vorticity(self.field(f"u{level}"), self.field(f"v{level}"), self.lat, self.lon, self.periodic)
        return self._cache["vort"]

    def wind10(self) -> np.ndarray:
        if "wind10" not in self._cache:
            self._cache["wind10"] = np.hypot(self.field("u10"), self.field("v10"))
        return self._cache["wind10"]

    def msl_minima(self) -> np.ndarray:
        if "minima" not in self._cache:
            self._cache["minima"] = _local_extrema(self.field("msl"), minimum=True, periodic=self.periodic)
        return self._cache["minima"]

    def thickness_maxima(self) -> np.ndarray:
        if "thick_max" not in self._cache:
            top, bottom = self.config.thickness_levels
            thickness = self.field(f"z{top}") - self.field(f"z{bottom}")
            self._cache["thick_max"] = _local_extrema(thickness, minimum=False, periodic=self.periodic)
        return self._cache["thick_max"]


def _local_extrema(values: np.ndarray, minimum: bool, periodic: bool) -> np.ndarray:
    """Boolean mask of grid points not larger (smaller for maxima) than their 8 neighbours."""
    from scipy.ndimage import maximum_filter, minimum_filter

    data = np.where(np.isnan(values), np.inf if minimum else -np.inf, values)
    mode = ("nearest", "wrap" if periodic else "nearest")
    filtered = minimum_filter(data, size=3, mode=mode) if minimum else maximum_filter(data, size=3, mode=mode)
    return (data == filtered) & ~np.isnan(values)


def _disk(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float, radius_km: float):
    """Row indices, column indices and distances (km) of grid points within ``radius_km``."""
    band = np.flatnonzero(np.abs(lat - lat0) <= radius_km / KM_PER_DEGREE + 1e-9)
    if band.size == 0:
        return band, band, np.empty(0)
    dist = great_circle_km(lat[band][:, None], lon[None, :], lat0, lon0)
    rows, cols = np.nonzero(dist <= radius_km)
    return band[rows], cols, dist[rows, cols]


def _move(lat: float, lon: float, east_km: float, north_km: float) -> Tuple[float, float]:
    new_lat = lat + north_km / KM_PER_DEGREE
    new_lat = max(-89.999, min(89.999, new_lat))
    coslat = max(math.cos(math.radians(lat)), 1e-6)
    return new_lat, lon + east_km / (KM_PER_DEGREE * coslat)


def _displacement_km(lat0: float, lon0: float, lat1: float, lon1: float) -> Tuple[float, float]:
    """East and north components (km) of the move from point 0 to point 1 (local tangent plane)."""
    dlon = (lon1 - lon0 + 180.0) % 360.0 - 180.0
    mean_lat = math.radians(0.5 * (lat0 + lat1))
    return dlon * KM_PER_DEGREE * math.cos(mean_lat), (lat1 - lat0) * KM_PER_DEGREE


def _nearest_index(values: np.ndarray, target: float, periodic: bool = False) -> int:
    diff = np.abs(values - target)
    if periodic:
        diff = np.minimum(diff, 360.0 - diff % 360.0)
    return int(np.argmin(diff))


def _checks(frame: _Frame, j: int, i: int, config: FollowingTrackerConfig) -> Optional[str]:
    """None if the candidate passes the checks, else the name of the failed check."""
    lat0, lon0 = float(frame.lat[j]), float(frame.lon[i])
    rows, cols, _ = _disk(frame.lat, frame.lon, lat0, lon0, config.check_radius_km)
    vort = frame.vorticity()[rows, cols]
    vort = vort[~np.isnan(vort)]
    if vort.size == 0:
        return "vorticity"
    if lat0 >= 0:
        if not vort.max() > config.vorticity_threshold:
            return "vorticity"
    elif not vort.min() < -config.vorticity_threshold:
        return "vorticity"
    if frame.has("lsm") and float(frame.field("lsm")[j, i]) >= 0.5:
        if not frame.wind10()[rows, cols].max() > config.land_wind_threshold:
            return "land_wind"
    if config.extratropical_latitude is not None and abs(lat0) > config.extratropical_latitude:
        if not frame.thickness_maxima()[rows, cols].any():
            return "thickness"
    return None


def _steering_km(frame: _Frame, lat0: float, lon0: float, levels: Sequence[int], hours: float) -> Optional[Tuple[float, float]]:
    if not levels or not all(frame.has(f"u{l}") and frame.has(f"v{l}") for l in levels):
        return None
    j = _nearest_index(frame.lat, lat0)
    i = _nearest_index(frame.lon % 360.0, lon0 % 360.0, periodic=True)
    u = float(np.mean([frame.field(f"u{l}")[j, i] for l in levels]))
    v = float(np.mean([frame.field(f"v{l}")[j, i] for l in levels]))
    seconds = hours * 3600.0
    return u * seconds / 1000.0, v * seconds / 1000.0


def _turn_angle(prev: Tuple[float, float], new: Tuple[float, float]) -> float:
    norm = math.hypot(*prev) * math.hypot(*new)
    if norm == 0.0:
        return 0.0
    cosine = (prev[0] * new[0] + prev[1] * new[1]) / norm
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _select(frame: _Frame, guess: Tuple[float, float], current: Optional[Tuple[float, float]], prev_disp, config: FollowingTrackerConfig):
    rows, cols, dist = _disk(frame.lat, frame.lon, guess[0], guess[1], config.search_radius_km)
    minima = frame.msl_minima()[rows, cols]
    rows, cols, dist = rows[minima], cols[minima], dist[minima]
    order = np.argsort(dist, kind="stable")
    for k in order:
        j, i = int(rows[k]), int(cols[k])
        if config.max_turn_deg is not None and prev_disp is not None and current is not None:
            new_disp = _displacement_km(current[0], current[1], float(frame.lat[j]), float(frame.lon[i]))
            if math.hypot(*new_disp) > 0 and _turn_angle(prev_disp, new_disp) >= config.max_turn_deg:
                continue
        if _checks(frame, j, i, config) is None:
            return j, i
    return None


def _point(frame: _Frame, j: int, i: int, config: FollowingTrackerConfig) -> Dict[str, float]:
    radius = config.intensity_radius_km or config.check_radius_km
    lat0, lon0 = float(frame.lat[j]), float(frame.lon[i])
    rows, cols, _ = _disk(frame.lat, frame.lon, lat0, lon0, radius)
    return {
        "lat": lat0,
        "lon": lon0,
        "msl": float(frame.field("msl")[j, i]),
        "wind_max": float(np.nanmax(frame.wind10()[rows, cols])),
    }


def follow_cyclone(
    fields,
    start_lat: float,
    start_lon: float,
    start_time,
    config: FollowingTrackerConfig = ECMWF_TRACKER,
    previous_position: Optional[Tuple[float, float]] = None,
    max_lead_hours: Optional[float] = None,
):
    """Track one cyclone through forecast ``fields`` from its position at ``start_time``.

    ``fields`` is an ``xarray.Dataset`` in the conventions of :mod:`pyhazards.forecasts.fields`
    (dimensions ``time``, ``lat``, ``lon``; valid times after ``start_time`` are tracked in order).
    Returns a DataFrame with columns ``time``, ``lead_hours``, ``lat``, ``lon``, ``msl`` (Pa) and
    ``wind_max`` (m/s); ``attrs['stop_reason']`` says why tracking ended.
    """
    import pandas as pd

    missing = [name for name in required_variables(config) if name not in fields.data_vars]
    if missing:
        raise KeyError(f"fields lack {missing} needed by the {config.name} tracker")
    if set(("time", "lat", "lon")) - set(fields.dims):
        raise ValueError(f"fields must have dimensions time, lat, lon; got {dict(fields.sizes)}")
    lat = np.asarray(fields["lat"].values, dtype=np.float64)
    lon = np.asarray(fields["lon"].values, dtype=np.float64)
    times = pd.to_datetime(fields["time"].values)
    start = pd.Timestamp(start_time)
    step = np.median(np.diff(lon)) if lon.size > 1 else 0.0
    periodic = lon.size > 2 and abs(lon.size * abs(step) - 360.0) < 1e-6

    rows: List[Dict[str, object]] = []
    current = (float(start_lat), float(start_lon))
    current_time = start
    prev_disp = None
    if previous_position is not None:
        prev_disp = _displacement_km(previous_position[0], previous_position[1], current[0], current[1])
    stop_reason = "end of forecast"

    start_index = np.flatnonzero(times == start)
    if start_index.size and config.relocate_start:
        frame = _Frame(fields, int(start_index[0]), lat, lon, config, periodic)
        found = _select(frame, current, None, None, config)
        if found is None:
            out = pd.DataFrame(rows, columns=["time", "lead_hours", "lat", "lon", "msl", "wind_max"])
            out.attrs["stop_reason"] = "no qualifying minimum at the initial time"
            return out
        point = _point(frame, *found, config)
        rows.append({"time": start, "lead_hours": 0.0, **point})
        lon_value = point["lon"]
        current = (point["lat"], lon_value)

    future = [k for k, time in enumerate(times) if time > start]
    if max_lead_hours is not None:
        future = [k for k in future if (times[k] - start).total_seconds() / 3600.0 <= max_lead_hours + 1e-9]
    previous_frame = _Frame(fields, int(start_index[0]), lat, lon, config, periodic) if start_index.size else None
    for k in future:
        frame = _Frame(fields, k, lat, lon, config, periodic)
        hours = (times[k] - current_time).total_seconds() / 3600.0
        guess = current
        if config.use_first_guess:
            steering = None
            if config.steering_weight > 0:
                steering = _steering_km(previous_frame if previous_frame is not None else frame, current[0], current[1], config.steering_levels, hours)
            weight = config.steering_weight
            if prev_disp is None and steering is None:
                east = north = 0.0
            elif prev_disp is None:
                east, north = steering
            elif steering is None:
                east, north = prev_disp
            else:
                east = (1.0 - weight) * prev_disp[0] + weight * steering[0]
                north = (1.0 - weight) * prev_disp[1] + weight * steering[1]
            if config.max_first_guess_km is not None:
                length = math.hypot(east, north)
                if length > config.max_first_guess_km:
                    east, north = east * config.max_first_guess_km / length, north * config.max_first_guess_km / length
            guess = _move(current[0], current[1], east, north)
        found = _select(frame, guess, current, prev_disp, config)
        if found is None:
            stop_reason = f"no qualifying minimum at {times[k]}"
            break
        point = _point(frame, *found, config)
        prev_disp = _displacement_km(current[0], current[1], point["lat"], point["lon"])
        rows.append({"time": times[k], "lead_hours": (times[k] - start).total_seconds() / 3600.0, **point})
        current = (point["lat"], point["lon"])
        current_time = times[k]
        previous_frame = frame
    out = pd.DataFrame(rows, columns=["time", "lead_hours", "lat", "lon", "msl", "wind_max"])
    out.attrs["stop_reason"] = stop_reason
    out.attrs["tracker"] = config.name
    return out


def with_overrides(config: FollowingTrackerConfig, **changes) -> FollowingTrackerConfig:
    """Copy of ``config`` with some parameters changed (e.g. ``extratropical_latitude=35``)."""
    return replace(config, **changes)
