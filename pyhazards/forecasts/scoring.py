"""Score forecast cyclone tracks against IBTrACS best tracks.

A *forecast-track table* has one row per storm, forecast and valid time, with columns ``SID``,
``init_time``, ``valid_time``, ``lead_hours``, ``lat``, ``lon`` (degrees), ``wind_ms`` (maximum 10 m
wind, m/s) and ``pres_pa`` (minimum sea-level pressure, Pa). The trackers' pipeline
(:mod:`pyhazards.forecasts.pipeline`) and the TCBench reader (:mod:`pyhazards.forecasts.tcbench`)
produce it.

Each row is compared with the IBTrACS record of the same storm at exactly the valid time (rows
without one are not scored). :func:`score_forecast_tracks` reports, per lead time and averaged over
leads, the great-circle track error in km (the ``tc`` benchmark's
:func:`~pyhazards.benchmarks.tc.track_intensity_metrics`, sphere of radius 6371 km), the maximum-wind
error in knots against an IBTrACS wind column (default ``USA_WIND``, 1-minute sustained) and the
central-pressure error in hPa (default ``USA_PRES``), plus the number of scored rows and the median
track error per lead (GraphCast reports medians).

``protocol="tcbench"`` reproduces TCBench's ``dev/evaluate_tracks.py`` and ``dev/metrics_test.py``
(TCBench_Alpha 93924483): forecasts initialised at 00 and 12 UTC only, duplicate (storm, initial
time, valid time) rows dropped keeping the first, wind converted with TCBench's factor 1.94384 kt per
m/s, and the direct position error computed as TCBench computes it, including a quirk that changes
its values by up to several km: TCBench reads IBTrACS ``LAT`` and ``LON`` as ``float16``
(``dev/utils/constants.py``), shifts negative longitudes by 360 in double precision, and evaluates
the haversine formula with the reference latitude still in ``float16`` (so ``np.radians`` and
``np.cos`` of it are half precision). :func:`tcbench_position_error_km` emulates this exactly; it
reproduces the ``DPE_GCD`` values printed in TCBench's ``dev/Getting_Started.ipynb``.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "KT_PER_MS",
    "TCBENCH_KT_PER_MS",
    "forecast_track_errors",
    "score_forecast_tracks",
    "tcbench_position_error_km",
]

KT_PER_MS = 1.0 / 0.514444
"""Knots per m/s (1 kt = 1852 m / 3600 s = 0.514444 m/s)."""

TCBENCH_KT_PER_MS = 1.94384
"""Conversion factor used by TCBench's ``track_matcher.py``."""

_TRACK_COLUMNS = ("SID", "init_time", "valid_time", "lat", "lon")


def tcbench_position_error_km(ref_lat, ref_lon, pred_lat, pred_lon) -> np.ndarray:
    """TCBench's ``toolbox.haversine`` on IBTrACS positions read as ``float16`` (see module docstring).

    ``ref_lat`` / ``ref_lon`` are the IBTrACS values in degrees as written in the CSV (longitudes in
    either convention); predictions are double precision. Radius 6371 km.
    """
    lat16 = np.asarray(ref_lat, dtype=np.float64).astype(np.float16)
    lon16 = np.asarray(ref_lon, dtype=np.float64).astype(np.float16)
    lon = np.where(lon16 < 0, lon16.astype(np.float64) + 360.0, lon16.astype(np.float64))
    latp = np.radians(lat16)  # float16, as in TCBench
    lonp = np.radians(lon)
    lat_list = np.radians(np.asarray(pred_lat, dtype=np.float64))
    lon_list = np.radians(np.asarray(pred_lon, dtype=np.float64))
    dlon = lonp - lon_list
    dlat = latp - lat_list
    a = np.power(np.sin(dlat / 2), 2) + np.cos(lat_list) * np.cos(latp) * np.power(np.sin(dlon / 2), 2)
    a = np.where(np.sqrt(a) <= 1, a, np.sign(a))
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def _reference_table(ibtracs, wind_column: Optional[str], pres_column: Optional[str]):
    import pandas as pd

    ref = pd.DataFrame(
        {
            "SID": ibtracs["SID"].astype(str).to_numpy(),
            "valid_time": pd.to_datetime(ibtracs["ISO_TIME"]).to_numpy(),
            "ref_lat": pd.to_numeric(ibtracs["LAT"], errors="coerce").to_numpy(dtype=float),
            "ref_lon": pd.to_numeric(ibtracs["LON"], errors="coerce").to_numpy(dtype=float),
        }
    )
    ref["ref_wind_kt"] = pd.to_numeric(ibtracs[wind_column], errors="coerce").to_numpy(dtype=float) if wind_column in ibtracs else np.nan
    ref["ref_pres_hpa"] = pd.to_numeric(ibtracs[pres_column], errors="coerce").to_numpy(dtype=float) if pres_column in ibtracs else np.nan
    return ref.drop_duplicates(subset=["SID", "valid_time"], keep="first")


def forecast_track_errors(
    tracks,
    ibtracs,
    protocol: str = "pyhazards",
    wind_column: Optional[str] = "USA_WIND",
    pres_column: Optional[str] = "USA_PRES",
    init_hours: Optional[Iterable[int]] = None,
):
    """Row-level errors of a forecast-track table against IBTrACS.

    Adds ``ref_lat``, ``ref_lon``, ``ref_wind_kt``, ``ref_pres_hpa``, ``wind_kt``, ``pres_hpa``,
    ``track_error_km``, ``wind_error_kt`` and ``pres_error_hpa`` (forecast minus best track). See the
    module docstring for ``protocol``; ``init_hours`` restricts initial times (TCBench: 0 and 12).
    """
    import pandas as pd

    if protocol not in {"pyhazards", "tcbench"}:
        raise ValueError("protocol must be 'pyhazards' or 'tcbench'")
    missing = [c for c in _TRACK_COLUMNS if c not in tracks.columns]
    if missing:
        raise ValueError(f"forecast-track table lacks columns {missing}")
    table = tracks.copy()
    table["init_time"] = pd.to_datetime(table["init_time"])
    table["valid_time"] = pd.to_datetime(table["valid_time"])
    table["SID"] = table["SID"].astype(str)
    if "lead_hours" not in table:
        table["lead_hours"] = (table["valid_time"] - table["init_time"]).dt.total_seconds() / 3600.0
    if protocol == "tcbench" and init_hours is None:
        init_hours = (0, 12)
    if init_hours is not None:
        table = table[table["init_time"].dt.hour.isin(list(init_hours))]
    ref = _reference_table(ibtracs, wind_column, pres_column)
    table = table[table["SID"].isin(set(ref["SID"]))]
    if "member" not in table.columns:
        table = table.drop_duplicates(subset=["init_time", "valid_time", "SID"], keep="first")
    table = table.merge(ref, on=["SID", "valid_time"], how="left")
    factor = TCBENCH_KT_PER_MS if protocol == "tcbench" else KT_PER_MS
    table["wind_kt"] = table["wind_ms"].astype(float) * factor if "wind_ms" in table else np.nan
    table["pres_hpa"] = table["pres_pa"].astype(float) / 100.0 if "pres_pa" in table else np.nan
    if protocol == "tcbench":
        distance = tcbench_position_error_km(table["ref_lat"], table["ref_lon"], table["lat"], table["lon"])
    else:
        from .fields import great_circle_km

        distance = great_circle_km(table["ref_lat"], table["ref_lon"], table["lat"], table["lon"])
    table["track_error_km"] = distance
    table["wind_error_kt"] = table["wind_kt"] - table["ref_wind_kt"]
    table["pres_error_hpa"] = table["pres_hpa"] - table["ref_pres_hpa"]
    return table.reset_index(drop=True)


def _lead_label(hours: float) -> str:
    return f"{int(hours)}h" if float(hours).is_integer() else f"{hours:g}h".replace(".", "p")


def score_forecast_tracks(
    tracks,
    ibtracs,
    lead_hours: Optional[Sequence[float]] = None,
    protocol: str = "pyhazards",
    wind_column: Optional[str] = "USA_WIND",
    pres_column: Optional[str] = "USA_PRES",
    init_hours: Optional[Iterable[int]] = None,
    source: Optional[str] = None,
):
    """Track and intensity errors per lead time as a :class:`~pyhazards.benchmarks.BenchmarkResult`.

    Metrics (``tc`` benchmark names): ``track_error_km``, ``intensity_mae`` (kt), ``pressure_mae``
    (hPa), each with ``_<h>h`` per lead, plus ``track_error_km_median_<h>h`` and ``count_<h>h``.
    ``lead_hours`` defaults to the multiples of 6 h present in the table. The sample is heterogeneous:
    each lead averages the rows that exist at that lead.
    """
    import pandas as pd
    import torch

    from ..benchmarks.schemas import BenchmarkResult
    from ..benchmarks.tc import track_intensity_metrics

    errors = forecast_track_errors(tracks, ibtracs, protocol, wind_column, pres_column, init_hours)
    errors = errors[np.isfinite(errors["ref_lat"].to_numpy(dtype=float))]
    if lead_hours is None:
        leads = sorted({float(h) for h in errors["lead_hours"] if float(h) > 0 and float(h) % 6 == 0})
    else:
        leads = [float(h) for h in lead_hours]
    if not leads:
        raise ValueError("no forecast rows match IBTrACS records at the requested lead times")
    errors = errors[errors["lead_hours"].isin(leads)]
    cases = errors[["SID", "init_time"]].drop_duplicates().reset_index(drop=True)
    case_index = {key: k for k, key in enumerate(zip(cases["SID"], cases["init_time"]))}
    lead_index = {h: k for k, h in enumerate(leads)}
    shape = (len(cases), len(leads))

    def grid(column: str) -> np.ndarray:
        out = np.full(shape, np.nan)
        for sid, init, lead, value in zip(errors["SID"], errors["init_time"], errors["lead_hours"], errors[column]):
            out[case_index[(sid, init)], lead_index[float(lead)]] = value
        return out

    metrics: Dict[str, float]
    if protocol == "tcbench":
        # TCBench's own distance (float16 quirk); averaged the same way as the tc benchmark.
        values = {"track_error_km": grid("track_error_km")}
        values["intensity_mae"] = np.abs(grid("wind_error_kt"))
        values["pressure_mae"] = np.abs(grid("pres_error_hpa"))
        metrics = {}
        for name, array in values.items():
            finite = array[np.isfinite(array)]
            metrics[name] = float(finite.mean()) if finite.size else float("nan")
            for k, hours in enumerate(leads):
                column = array[:, k]
                column = column[np.isfinite(column)]
                metrics[f"{name}_{_lead_label(hours)}"] = float(column.mean()) if column.size else float("nan")
    else:
        forecast = {"lat": torch.as_tensor(grid("lat")), "lon": torch.as_tensor(grid("lon"))}
        target = {"lat": torch.as_tensor(grid("ref_lat")), "lon": torch.as_tensor(grid("ref_lon"))}
        if errors["wind_kt"].notna().any():
            forecast["wind"], target["wind"] = torch.as_tensor(grid("wind_kt")), torch.as_tensor(grid("ref_wind_kt"))
        if errors["pres_hpa"].notna().any():
            forecast["pres"], target["pres"] = torch.as_tensor(grid("pres_hpa")), torch.as_tensor(grid("ref_pres_hpa"))
        metrics = track_intensity_metrics(forecast, target, leads)
    track = grid("track_error_km")
    for k, hours in enumerate(leads):
        column = track[:, k]
        column = column[np.isfinite(column)]
        metrics[f"count_{_lead_label(hours)}"] = float(column.size)
        metrics[f"track_error_km_median_{_lead_label(hours)}"] = float(np.median(column)) if column.size else float("nan")
    return BenchmarkResult(
        benchmark_name="tc",
        hazard_task="tc.track_intensity",
        metrics={key: value for key, value in metrics.items() if not (isinstance(value, float) and np.isnan(value))},
        metadata={
            "source": source,
            "protocol": protocol,
            "lead_hours": leads,
            "units": {"track_error_km": "km", "intensity_mae": "kt", "pressure_mae": "hPa"},
            "reference": {"wind": wind_column, "pressure": pres_column},
            "num_cases": len(cases),
            "num_storms": int(cases["SID"].nunique()),
            "sample": "heterogeneous (each lead averages the rows available at that lead)",
            "synthetic": False,
        },
    )
