"""Synthetic global fields with idealised cyclones on known tracks (for tests and examples only).

Nothing here describes real weather. Each vortex has a Gaussian sea-level-pressure deficit, a
cyclonic (counter-clockwise in the Northern Hemisphere, clockwise in the Southern) tangential wind
of the form ``v(r) = vmax (r / rmax) exp((1 - (r / rmax)^2) / 2)`` at 10 m and 850 hPa, a warm core
(a Gaussian bump of the 300-500 hPa and 200-850 hPa thicknesses) and moves at a constant velocity,
which is also the uniform steering wind at 200-850 hPa. The background pressure varies smoothly so
that it has no flat plateaus.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .fields import EARTH_RADIUS_KM, great_circle_km

__all__ = ["SyntheticVortex", "synthetic_cyclone_fields", "vortex_track"]

_KM_PER_DEG = math.pi * EARTH_RADIUS_KM / 180.0


@dataclass(frozen=True)
class SyntheticVortex:
    """An idealised cyclone; ``u`` / ``v`` (m/s) is its motion and the steering flow around it."""

    lat: float
    lon: float
    u: float = -5.0
    v: float = 2.0
    pressure_drop: float = 3000.0  # Pa
    radius_km: float = 250.0  # e-folding radius of the pressure deficit
    vmax: float = 30.0  # m/s
    rmax_km: float = 100.0
    warm_core: float = 300.0  # m2 s-2, amplitude of the thickness anomaly
    warm_core_radius_km: float = 300.0
    cyclonic: bool = True


def vortex_track(vortex: SyntheticVortex, hours: Sequence[float]) -> List[Tuple[float, float]]:
    """Centre positions (lat, lon in degrees) after ``hours`` of constant-velocity motion."""
    out = []
    for h in hours:
        north = vortex.v * h * 3.6
        east = vortex.u * h * 3.6
        lat = vortex.lat + north / _KM_PER_DEG
        mean_lat = math.radians(0.5 * (vortex.lat + lat))
        lon = vortex.lon + east / (_KM_PER_DEG * math.cos(mean_lat))
        out.append((lat, lon % 360.0))
    return out


def _local_offsets(lat, lon, lat0, lon0):
    """East / north offsets (km) of grid points from a centre on the local tangent plane."""
    dlon = (lon - lon0 + 180.0) % 360.0 - 180.0
    east = dlon * _KM_PER_DEG * np.cos(np.deg2rad(lat))
    north = (lat - lat0) * _KM_PER_DEG
    return east, north


def synthetic_cyclone_fields(
    vortices: Sequence[SyntheticVortex],
    hours: Sequence[float] = tuple(range(0, 121, 6)),
    resolution: float = 0.5,
    init_time="2020-01-01T00:00",
    lat_descending: bool = True,
    land_mask: bool = False,
):
    """Global fields (``xarray.Dataset`` in the tracker conventions) with the given vortices.

    Variables: ``msl``, ``u10``, ``v10``, ``u/v`` at 200, 500, 700 and 850 hPa, ``z200``, ``z300``,
    ``z500``, ``z850`` and, with ``land_mask``, an ``lsm`` that is land east of 160 W and west of 20 W
    in the Northern Hemisphere poleward of 20 N. ``attrs['tracks']`` holds each vortex's true centres.
    """
    import pandas as pd
    import xarray as xr

    lat = np.arange(-90.0, 90.0 + resolution / 2, resolution)
    if lat_descending:
        lat = lat[::-1]
    lon = np.arange(0.0, 360.0, resolution)
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    phi, lam = np.deg2rad(lat2), np.deg2rad(lon2)
    background = 101325.0 + 900.0 * np.sin(phi) ** 2 + 250.0 * np.cos(phi) * np.cos(2 * lam + 0.3) + 120.0 * np.sin(3 * phi + 0.2) * np.sin(lam)
    nt = len(hours)
    shape = (nt,) + lat2.shape
    fields = {name: np.zeros(shape) for name in ("msl", "u10", "v10", "z200", "z300", "z500", "z850")}
    for level in (200, 500, 700, 850):
        fields[f"u{level}"] = np.zeros(shape)
        fields[f"v{level}"] = np.zeros(shape)
    fields["msl"][:] = background
    fields["z850"][:] = 14000.0 + 300.0 * np.cos(phi) ** 2
    fields["z500"][:] = 55000.0 + 1500.0 * np.cos(phi) ** 2
    fields["z300"][:] = 90000.0 + 2500.0 * np.cos(phi) ** 2
    fields["z200"][:] = 115000.0 + 3500.0 * np.cos(phi) ** 2
    tracks = []
    for vortex in vortices:
        centres = vortex_track(vortex, hours)
        tracks.append(centres)
        for t, (clat, clon) in enumerate(centres):
            r = great_circle_km(lat2, lon2, clat, clon)
            east, north = _local_offsets(lat2, lon2, clat, clon)
            fields["msl"][t] -= vortex.pressure_drop * np.exp(-((r / vortex.radius_km) ** 2))
            x = r / vortex.rmax_km
            speed = vortex.vmax * x * np.exp(0.5 * (1.0 - x**2))
            with np.errstate(invalid="ignore", divide="ignore"):
                sense = (1.0 if clat >= 0 else -1.0) * (1.0 if vortex.cyclonic else -1.0)
                tu = np.where(r > 0, -sense * speed * north / r, 0.0)
                tv = np.where(r > 0, sense * speed * east / r, 0.0)
            near = np.exp(-((r / (8 * vortex.radius_km)) ** 2))  # confine the steering flow to the storm area
            fields["u10"][t] += 0.9 * tu + vortex.u * near
            fields["v10"][t] += 0.9 * tv + vortex.v * near
            fields["u850"][t] += 0.8 * tu
            fields["v850"][t] += 0.8 * tv
            for level in (200, 500, 700, 850):
                fields[f"u{level}"][t] += vortex.u * near
                fields[f"v{level}"][t] += vortex.v * near
            bump = vortex.warm_core * np.exp(-((r / vortex.warm_core_radius_km) ** 2))
            fields["z300"][t] += bump
            fields["z200"][t] += 2.0 * bump
    times = [pd.Timestamp(init_time) + pd.Timedelta(hours=float(h)) for h in hours]
    data = {name: (("time", "lat", "lon"), value.astype(np.float32)) for name, value in fields.items()}
    if land_mask:
        land = ((lon2 > 200.0) & (lon2 < 340.0) & (lat2 > 20.0)).astype(np.float32)
        data["lsm"] = (("lat", "lon"), land)
    ds = xr.Dataset(data, coords={"time": times, "lat": lat, "lon": lon})
    ds.attrs["tracks"] = repr(tracks)
    ds.attrs["synthetic"] = 1
    return ds
