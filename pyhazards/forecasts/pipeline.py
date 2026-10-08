"""Forecast fields -> cyclone tracks -> forecast-track table matched to IBTrACS storms.

Two routes, matching the two protocols in the literature:

* :func:`tempest_forecast_tracks` (TCBench): detect and stitch every cyclone in the forecast with the
  TempestExtremes rules (:mod:`.tempest`), then assign IBTrACS storm IDs with HuracanPy's matching
  rule (:mod:`.matching`).
* :func:`follow_forecast_tracks` (Pangu-Weather, GraphCast papers): for each IBTrACS storm present at
  the initial time, follow it from its observed position with the ECMWF-style tracker
  (:mod:`.following`).

Both return the forecast-track table scored by :func:`pyhazards.forecasts.scoring.score_forecast_tracks`.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .following import ECMWF_TRACKER, FollowingTrackerConfig, follow_cyclone
from .matching import matched_forecast_tracks
from .tempest import TCBENCH_DETECT_NODES, TCBENCH_STITCH_NODES, DetectNodesConfig, StitchNodesConfig, tempest_tracks

__all__ = ["follow_forecast_tracks", "storms_at", "tempest_forecast_tracks"]

_COLUMNS = ["SID", "init_time", "valid_time", "lead_hours", "lat", "lon", "wind_ms", "pres_pa"]


def tempest_forecast_tracks(
    fields,
    ibtracs,
    init_time,
    detect: DetectNodesConfig = TCBENCH_DETECT_NODES,
    stitch: StitchNodesConfig = TCBENCH_STITCH_NODES,
    max_dist_km: float = 300.0,
    return_tracks: bool = False,
):
    """TCBench route: TempestExtremes-style tracks of one forecast, matched to IBTrACS storms.

    ``fields`` is an ``xarray.Dataset`` with ``msl``, ``u10``, ``v10``, ``z300``, ``z500`` (for the
    default configuration) on ``(time, lat, lon)``. ``wind_ms`` / ``pres_pa`` come from the
    ``wind10`` / ``slp`` outputs (maximum 10 m wind within 2 degrees, MSLP at the node).
    With ``return_tracks`` the unmatched StitchNodes table is returned as well.
    """
    names = {detect.search_by_min}
    for criterion in detect.closed_contours + detect.no_closed_contours:
        names.update(_fields_in(criterion.variable))
    for output in detect.outputs:
        names.update(_fields_in(output.variable))
    arrays = {name: fields[name].transpose("time", "lat", "lon").values for name in sorted(names)}
    tracks = tempest_tracks(arrays, fields["lat"].values, fields["lon"].values, fields["time"].values, detect, stitch)
    columns = set(stitch.columns)
    table = matched_forecast_tracks(
        tracks,
        ibtracs,
        init_time,
        max_dist_km=max_dist_km,
        wind_column="wind10" if "wind10" in columns else None,
        pressure_column="slp" if "slp" in columns else None,
    )
    return (table, tracks) if return_tracks else table


def _fields_in(expression: str):
    import re

    return {token for token in re.split(r"[(),\s]+", expression) if token and not token.startswith("_")}


def storms_at(ibtracs, time, include_spur: bool = False):
    """IBTrACS storms with a record at ``time``: DataFrame of ``SID``, ``LAT``, ``LON`` (one row per storm)."""
    import pandas as pd

    stamp = pd.Timestamp(time)
    rows = ibtracs[pd.to_datetime(ibtracs["ISO_TIME"]) == stamp]
    if not include_spur and "TRACK_TYPE" in rows.columns:
        rows = rows[~rows["TRACK_TYPE"].astype(str).str.contains("spur", case=False, na=False)]
    rows = rows.dropna(subset=["LAT", "LON"])
    return rows.drop_duplicates(subset=["SID"])[["SID", "LAT", "LON"]].reset_index(drop=True)


def follow_forecast_tracks(
    fields,
    ibtracs,
    init_time,
    config: FollowingTrackerConfig = ECMWF_TRACKER,
    sids: Optional[Iterable[str]] = None,
    use_previous_position: bool = True,
    max_lead_hours: Optional[float] = None,
):
    """Paper route: follow each IBTrACS storm present at ``init_time`` through the forecast ``fields``.

    The start is the IBTrACS position at ``init_time``; with ``use_previous_position`` the position 6 h
    earlier gives the first displacement for the first guess. ``wind_ms`` is the maximum 10 m wind
    within the intensity radius and ``pres_pa`` the MSLP at the centre.
    """
    import pandas as pd

    init = pd.Timestamp(init_time)
    storms = storms_at(ibtracs, init)
    if sids is not None:
        wanted = {str(s) for s in sids}
        storms = storms[storms["SID"].astype(str).isin(wanted)]
    before = storms_at(ibtracs, init - pd.Timedelta(hours=6)).set_index("SID") if use_previous_position else None
    frames = []
    for sid, lat, lon in zip(storms["SID"], storms["LAT"], storms["LON"]):
        previous = None
        if before is not None and sid in before.index:
            previous = (float(before.loc[sid, "LAT"]), float(before.loc[sid, "LON"]))
        track = follow_cyclone(fields, float(lat), float(lon), init, config, previous_position=previous, max_lead_hours=max_lead_hours)
        track = track[track["lead_hours"] > 0]
        if track.empty:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "SID": str(sid),
                    "init_time": init,
                    "valid_time": pd.to_datetime(track["time"]).to_numpy(),
                    "lead_hours": track["lead_hours"].to_numpy(dtype=float),
                    "lat": track["lat"].to_numpy(dtype=float),
                    "lon": track["lon"].to_numpy(dtype=float),
                    "wind_ms": track["wind_max"].to_numpy(dtype=float),
                    "pres_pa": track["msl"].to_numpy(dtype=float),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=_COLUMNS)
    return pd.concat(frames, ignore_index=True)[_COLUMNS]
