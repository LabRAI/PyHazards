"""Assign best-track storm IDs to tracks detected in a forecast.

TCBench matches the TempestExtremes tracks of each forecast to IBTrACS with HuracanPy's
``huracanpy.assess.match`` at its defaults (``dev/track_matcher.py``). :func:`match_tracks` implements
that rule (HuracanPy 1.4.0, ``assess/_match.py``, MIT; written from its documented behaviour, not
copied): a detected track and a best track match when, at one or more common times, their positions
are at most ``max_dist_km`` apart (haversine; HuracanPy's ``haversine`` package uses the mean Earth
radius 6371.0088 km). Every matching pair is kept, so one detected track can be assigned to several
storms and vice versa, as in TCBench. :func:`matched_forecast_tracks` then writes, for each pair, all
points of the detected track under the storm's ``SID`` (TCBench's ``matched_tracks`` layout).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .fields import great_circle_km

__all__ = ["HURACANPY_EARTH_RADIUS_KM", "match_tracks", "matched_forecast_tracks"]

HURACANPY_EARTH_RADIUS_KM = 6371.0088
"""Mean Earth radius of the ``haversine`` package used by HuracanPy's matcher."""


def match_tracks(
    tracks,
    reference,
    max_dist_km: float = 300.0,
    min_overlap: int = 0,
    radius_km: float = HURACANPY_EARTH_RADIUS_KM,
):
    """Pairs of (best track, detected track) sharing a time with positions within ``max_dist_km``.

    ``tracks`` has columns ``track_id``, ``time``, ``lat``, ``lon``; ``reference`` (e.g. an IBTrACS table
    from :func:`pyhazards.datasets.tc.read_ibtracs`) has ``SID``, ``ISO_TIME``, ``LAT``, ``LON``.
    Returns a DataFrame with ``SID``, ``track_id``, ``overlap`` (number of close common times) and
    ``mean_dist_km``. With ``min_overlap >= 2`` pairs need that many close common times (HuracanPy).
    """
    import pandas as pd

    left = pd.DataFrame(
        {
            "SID": reference["SID"].astype(str).to_numpy(),
            "time": pd.to_datetime(reference["ISO_TIME"]).to_numpy(),
            "lat_ref": pd.to_numeric(reference["LAT"], errors="coerce").to_numpy(dtype=float),
            "lon_ref": pd.to_numeric(reference["LON"], errors="coerce").to_numpy(dtype=float),
        }
    )
    right = pd.DataFrame(
        {
            "track_id": tracks["track_id"].to_numpy(),
            "time": pd.to_datetime(tracks["time"]).to_numpy(),
            "lat_det": tracks["lat"].to_numpy(dtype=float),
            "lon_det": tracks["lon"].to_numpy(dtype=float),
        }
    )
    merged = left.merge(right, on="time").drop_duplicates()
    columns = ["SID", "track_id", "overlap", "mean_dist_km"]
    if merged.empty:
        return pd.DataFrame(columns=columns)
    merged["dist"] = great_circle_km(merged["lat_ref"], merged["lon_ref"], merged["lat_det"], merged["lon_det"], radius_km)
    if max_dist_km is not None:
        merged = merged[merged["dist"] <= max_dist_km]
    grouped = merged.groupby(["SID", "track_id"], sort=True)["dist"]
    result = pd.DataFrame({"overlap": grouped.count(), "mean_dist_km": grouped.mean()}).reset_index()
    if min_overlap >= 2:
        result = result[result["overlap"] >= min_overlap]
    return result[columns].reset_index(drop=True)


def matched_forecast_tracks(
    tracks,
    reference,
    init_time,
    max_dist_km: float = 300.0,
    min_overlap: int = 0,
    wind_column: Optional[str] = "wind10",
    pressure_column: Optional[str] = "slp",
):
    """Forecast-track table (one row per storm and valid time) for the detected ``tracks`` of one forecast.

    Columns: ``SID``, ``init_time``, ``valid_time``, ``lead_hours``, ``lat``, ``lon``, ``wind_ms`` and
    ``pres_pa`` (taken from ``wind_column`` / ``pressure_column`` of ``tracks``, NaN when absent) and
    ``track_id``. This is the layout :func:`pyhazards.forecasts.scoring.score_forecast_tracks` scores.
    """
    import pandas as pd

    init = pd.Timestamp(init_time)
    pairs = match_tracks(tracks, reference, max_dist_km=max_dist_km, min_overlap=min_overlap)
    frames = []
    for sid, track_id in zip(pairs["SID"], pairs["track_id"]):
        points = tracks[tracks["track_id"] == track_id]
        valid = pd.to_datetime(points["time"])
        frames.append(
            pd.DataFrame(
                {
                    "SID": sid,
                    "init_time": init,
                    "valid_time": valid.to_numpy(),
                    "lead_hours": ((valid - init).dt.total_seconds() / 3600.0).to_numpy(),
                    "lat": points["lat"].to_numpy(dtype=float),
                    "lon": points["lon"].to_numpy(dtype=float),
                    "wind_ms": points[wind_column].to_numpy(dtype=float) if wind_column in points else np.nan,
                    "pres_pa": points[pressure_column].to_numpy(dtype=float) if pressure_column in points else np.nan,
                    "track_id": track_id,
                }
            )
        )
    columns = ["SID", "init_time", "valid_time", "lead_hours", "lat", "lon", "wind_ms", "pres_pa", "track_id"]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]
