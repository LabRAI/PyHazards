"""Hydrological evaluation metrics, defined as in NeuralHydrology.

Ported from NeuralHydrology, ``neuralhydrology/evaluation/metrics.py`` at commit ea94a40
(https://github.com/neuralhydrology/neuralhydrology, BSD-3-Clause, Copyright (c) 2021, NeuralHydrology).
The reference works on xarray DataArrays; this port works on 1-D NumPy arrays (plus an optional array of
dates for the peak-timing metrics) and keeps every definition, default and edge case of the reference:

- observations and simulations are masked to the time steps where both are finite;
- standard deviations are population standard deviations (``ddof=0``, as ``DataArray.std()``);
- the flow-duration-curve metrics set zero observations and non-positive simulations to 1e-6 before
  taking logarithms, and use ``np.round`` (round half to even) for the curve indices;
- the peak metrics find observed peaks with ``scipy.signal.find_peaks`` and skip peaks whose window
  runs past the series or across a gap in the dates.

``calculate_metrics`` returns NeuralHydrology's deterministic metric set for one basin under
snake-case names (``NSE`` -> ``nse``, ``Alpha-NSE`` -> ``alpha_nse``, ``Peak-Timing`` -> ``peak_timing``, ...).
``aggregate_basin_metrics`` summarises per-basin values the way Kratzert et al. (HESS 2019, Table 2)
report them: the median and the mean over basins, and the number of basins with NSE <= 0.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal, stats

# NeuralHydrology name -> PyHazards key, in the order of ``calculate_all_metrics``.
NEURALHYDROLOGY_METRIC_NAMES: Dict[str, str] = {
    "NSE": "nse",
    "MSE": "mse",
    "RMSE": "rmse",
    "KGE": "kge",
    "Alpha-NSE": "alpha_nse",
    "Beta-KGE": "beta_kge",
    "Beta-NSE": "beta_nse",
    "Pearson-r": "pearson_r",
    "FHV": "fhv",
    "FMS": "fms",
    "FLV": "flv",
    "Peak-Timing": "peak_timing",
    "Missed-Peaks": "missed_peaks",
    "Peak-MAPE": "peak_mape",
}
STREAMFLOW_METRICS: Tuple[str, ...] = tuple(NEURALHYDROLOGY_METRIC_NAMES.values())


def _as_series(obs, sim) -> Tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    if obs.shape != sim.shape:
        raise ValueError(f"Shapes of observations {obs.shape} and simulations {sim.shape} must match.")
    if obs.ndim != 1:
        raise ValueError(f"Metrics are only defined for 1-D time series, got ndim={obs.ndim}.")
    return obs, sim


def _mask_valid(obs: np.ndarray, sim: np.ndarray, dates: Optional[np.ndarray] = None):
    """Keep the time steps where neither series is NaN (``_mask_valid`` of the reference)."""
    idx = ~np.isnan(sim) & ~np.isnan(obs)
    if dates is None:
        return obs[idx], sim[idx]
    return obs[idx], sim[idx], dates[idx]


def _fdc(values: np.ndarray) -> np.ndarray:
    """Flow duration curve: values sorted in descending order."""
    return np.sort(values)[::-1].copy()


def nse(obs, sim) -> float:
    """Nash-Sutcliffe efficiency: 1 - sum((sim - obs)^2) / sum((obs - mean(obs))^2)."""
    obs, sim = _mask_valid(*_as_series(obs, sim))
    denominator = ((obs - obs.mean()) ** 2).sum()
    numerator = ((sim - obs) ** 2).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(1 - numerator / denominator)


def mse(obs, sim) -> float:
    obs, sim = _mask_valid(*_as_series(obs, sim))
    with np.errstate(invalid="ignore"):
        return float(((sim - obs) ** 2).mean()) if obs.size else float("nan")


def rmse(obs, sim) -> float:
    return float(np.sqrt(mse(obs, sim)))


def alpha_nse(obs, sim) -> float:
    """Ratio of the standard deviations, sigma_sim / sigma_obs (Gupta et al. 2009)."""
    obs, sim = _mask_valid(*_as_series(obs, sim))
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(sim.std() / obs.std()) if obs.size else float("nan")


def beta_nse(obs, sim) -> float:
    """(mean(sim) - mean(obs)) / sigma_obs (Gupta et al. 2009)."""
    obs, sim = _mask_valid(*_as_series(obs, sim))
    with np.errstate(divide="ignore", invalid="ignore"):
        return float((sim.mean() - obs.mean()) / obs.std()) if obs.size else float("nan")


def beta_kge(obs, sim) -> float:
    """mean(sim) / mean(obs), the bias term of the KGE."""
    obs, sim = _mask_valid(*_as_series(obs, sim))
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(sim.mean() / obs.mean()) if obs.size else float("nan")


def kge(obs, sim, weights: Sequence[float] = (1.0, 1.0, 1.0)) -> float:
    """Kling-Gupta efficiency 1 - sqrt((s_r (r-1))^2 + (s_a (alpha-1))^2 + (s_b (beta-1))^2).

    As in the reference, the weights multiply the squared terms.
    """
    if len(weights) != 3:
        raise ValueError("Weights of the KGE must be a list of three values")
    obs, sim = _mask_valid(*_as_series(obs, sim))
    if len(obs) < 2:
        return float("nan")
    r, _ = stats.pearsonr(obs, sim)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = sim.std() / obs.std()
        beta = sim.mean() / obs.mean()
        value = weights[0] * (r - 1) ** 2 + weights[1] * (alpha - 1) ** 2 + weights[2] * (beta - 1) ** 2
    return float(1 - np.sqrt(float(value)))


def pearson_r(obs, sim) -> float:
    obs, sim = _mask_valid(*_as_series(obs, sim))
    if len(obs) < 2:
        return float("nan")
    r, _ = stats.pearsonr(obs, sim)
    return float(r)


def fdc_fms(obs, sim, lower: float = 0.2, upper: float = 0.7) -> float:
    """Bias of the slope of the middle section of the flow duration curve, in percent (Yilmaz et al. 2008)."""
    obs, sim = _mask_valid(*_as_series(obs, sim))
    if len(obs) < 1:
        return float("nan")
    if any((x <= 0) or (x >= 1) for x in (upper, lower)):
        raise ValueError("upper and lower have to be in range ]0,1[")
    if lower >= upper:
        raise ValueError("The lower threshold has to be smaller than the upper.")
    obs, sim = _fdc(obs), _fdc(sim)
    sim[sim <= 0] = 1e-6
    obs[obs == 0] = 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        qsm_lower = np.log(sim[np.round(lower * len(sim)).astype(int)])
        qsm_upper = np.log(sim[np.round(upper * len(sim)).astype(int)])
        qom_lower = np.log(obs[np.round(lower * len(obs)).astype(int)])
        qom_upper = np.log(obs[np.round(upper * len(obs)).astype(int)])
        fms = ((qsm_lower - qsm_upper) - (qom_lower - qom_upper)) / (qom_lower - qom_upper + 1e-6)
    return float(fms * 100)


def fdc_fhv(obs, sim, h: float = 0.02) -> float:
    """Peak-flow bias of the top ``h`` fraction of the flow duration curve, in percent."""
    obs, sim = _mask_valid(*_as_series(obs, sim))
    if len(obs) < 1:
        return float("nan")
    if (h <= 0) or (h >= 1):
        raise ValueError("h has to be in range ]0,1[. Consider small values, e.g. 0.02 for 2% peak flows")
    obs, sim = _fdc(obs), _fdc(sim)
    obs = obs[: np.round(h * len(obs)).astype(int)]
    sim = sim[: np.round(h * len(sim)).astype(int)]
    with np.errstate(divide="ignore", invalid="ignore"):
        fhv = np.sum(sim - obs) / np.sum(obs)
    return float(fhv * 100)


def fdc_flv(obs, sim, l: float = 0.3) -> float:  # noqa: E741 (reference argument name)
    """Low-flow bias of the bottom ``l`` fraction of the flow duration curve (log space), in percent.

    As in the reference, a fraction that rounds to zero elements selects the whole curve
    (``array[-0:]``).
    """
    obs, sim = _mask_valid(*_as_series(obs, sim))
    if len(obs) < 1:
        return float("nan")
    if (l <= 0) or (l >= 1):
        raise ValueError("l has to be in range ]0,1[. Consider small values, e.g. 0.3 for 30% low flows")
    obs, sim = _fdc(obs), _fdc(sim)
    sim[sim <= 0] = 1e-6
    obs[obs == 0] = 1e-6
    obs = obs[-np.round(l * len(obs)).astype(int):]
    sim = sim[-np.round(l * len(sim)).astype(int):]
    with np.errstate(divide="ignore", invalid="ignore"):
        obs = np.log(obs)
        sim = np.log(sim)
        qsl = np.sum(sim - sim.min())
        qol = np.sum(obs - obs.min())
        flv = -1 * (qsl - qol) / (qol + 1e-6)
    return float(flv * 100)


def _resolution(resolution: str) -> pd.Timedelta:
    return pd.to_timedelta(resolution if resolution[0].isdigit() else "1" + resolution)


def _default_window(minimum: int, resolution: str) -> int:
    # NeuralHydrology: max(int(get_frequency_factor('12h', resolution)), minimum).
    return max(int(pd.Timedelta("12h") / _resolution(resolution)), minimum)


def _dates_or_index(dates, n: int, resolution: str) -> np.ndarray:
    if dates is None:
        return (np.datetime64("2000-01-01") + np.arange(n) * _resolution(resolution).to_timedelta64())
    dates = np.asarray(dates, dtype="datetime64[ns]")
    if dates.shape != (n,):
        raise ValueError(f"dates must have shape ({n},), got {dates.shape}.")
    return dates


def _window_has_gap(dates: np.ndarray, idx: int, window: int, step: pd.Timedelta) -> bool:
    # pd.date_range(dates[idx - window], dates[idx + window], freq=step).size != 2 * window + 1
    span = pd.Timedelta(dates[idx + window] - dates[idx - window])
    size = int(span // step) + 1 if span >= pd.Timedelta(0) else 0
    return size != 2 * window + 1


def mean_peak_timing(obs, sim, dates=None, window: Optional[int] = None, resolution: str = "1D") -> float:
    """Mean absolute timing error (in time steps) of the observed peaks (Kratzert et al. 2021, appendix).

    Observed peaks: ``find_peaks(obs, distance=100, prominence=std(obs))``. The simulated peak is the
    value at the observed peak when it is a local maximum, else the maximum within ``window`` steps
    on either side (default 3 for daily data). ``dates`` (one per time step) are needed to skip
    windows across gaps; without them the series is taken as gap-free.
    """
    obs, sim = _as_series(obs, sim)
    dates = _dates_or_index(dates, len(obs), resolution)
    obs, sim, dates = _mask_valid(obs, sim, dates)
    step = _resolution(resolution)
    peaks, _ = signal.find_peaks(obs, distance=100, prominence=np.std(obs))
    if window is None:
        window = _default_window(3, resolution)
    timing_errors = []
    for idx in peaks:
        if (idx - window < 0) or (idx + window >= len(obs)) or _window_has_gap(dates, idx, window, step):
            continue
        if (sim[idx] > sim[idx - 1]) and (sim[idx] > sim[idx + 1]):
            peak_sim = idx
        else:
            peak_sim = idx - window + int(np.argmax(sim[idx - window: idx + window + 1]))
        delta = pd.Timedelta(dates[idx] - dates[peak_sim])
        timing_errors.append(abs(delta / step))
    return float(np.mean(timing_errors)) if timing_errors else float("nan")


def missed_peaks(
    obs, sim, dates=None, window: Optional[int] = None, resolution: str = "1D", percentile: float = 80
) -> float:
    """Fraction of observed peaks above the ``percentile`` flow without a simulated peak nearby.

    Peaks of both series: ``find_peaks(distance=30, height=percentile)``; a simulated peak within
    ``window`` steps (default 1 for daily data) counts as a hit. As in the reference, the denominator
    counts all observed peaks, including those skipped at the edges or next to gaps.
    """
    obs, sim = _as_series(obs, sim)
    dates = _dates_or_index(dates, len(obs), resolution)
    obs, sim, dates = _mask_valid(obs, sim, dates)
    step = _resolution(resolution)
    min_obs_height = np.percentile(obs, percentile)
    min_sim_height = np.percentile(sim, percentile)
    peaks_obs_times, _ = signal.find_peaks(obs, distance=30, height=min_obs_height)
    peaks_sim_times, _ = signal.find_peaks(sim, distance=30, height=min_sim_height)
    if len(peaks_obs_times) == 0:
        return 0.0
    if window is None:
        window = _default_window(1, resolution)
    missed_events = 0
    for idx in peaks_obs_times:
        if (idx - window < 0) or (idx + window >= len(obs)) or _window_has_gap(dates, idx, window, step):
            continue
        if len(np.where(np.abs(peaks_sim_times - idx) <= window)[0]) == 0:
            missed_events += 1
    return missed_events / len(peaks_obs_times)


def mean_absolute_percentage_peak_error(obs, sim) -> float:
    """Mean absolute percentage error of the simulation at the observed peaks (``distance=100``,
    ``prominence=std(obs)``)."""
    obs, sim = _mask_valid(*_as_series(obs, sim))
    if obs.size == 0 or sim.size == 0:
        return float("nan")
    peaks, _ = signal.find_peaks(obs, distance=100, prominence=np.std(obs))
    if peaks.size == 0:
        return float("nan")
    obs, sim = obs[peaks], sim[peaks]
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.sum(np.abs((sim - obs) / obs)) / peaks.size * 100)


_METRIC_FUNCTIONS: Dict[str, Callable[..., float]] = {
    "nse": nse,
    "mse": mse,
    "rmse": rmse,
    "kge": kge,
    "alpha_nse": alpha_nse,
    "beta_kge": beta_kge,
    "beta_nse": beta_nse,
    "pearson_r": pearson_r,
    "fhv": fdc_fhv,
    "fms": fdc_fms,
    "flv": fdc_flv,
    "peak_timing": mean_peak_timing,
    "missed_peaks": missed_peaks,
    "peak_mape": mean_absolute_percentage_peak_error,
}
_DATED_METRICS = {"peak_timing", "missed_peaks"}


def calculate_metrics(
    obs,
    sim,
    dates=None,
    metrics: Optional[Iterable[str]] = None,
    resolution: str = "1D",
) -> Dict[str, float]:
    """Metrics of one basin's series; ``metrics`` defaults to all of ``STREAMFLOW_METRICS``.

    Like the NeuralHydrology tester, a basin whose observations or simulations are all NaN gets NaN
    for every metric (the reference raises ``AllNaNError`` and the tester then stores NaN).
    """
    obs, sim = _as_series(obs, sim)
    names = list(STREAMFLOW_METRICS if metrics is None else metrics)
    unknown = [name for name in names if name not in _METRIC_FUNCTIONS]
    if unknown:
        raise ValueError(f"Unknown streamflow metric(s) {unknown}; known: {list(STREAMFLOW_METRICS)}")
    if np.isnan(obs).all() or np.isnan(sim).all():
        return {name: float("nan") for name in names}
    values: Dict[str, float] = {}
    for name in names:
        if name in _DATED_METRICS:
            values[name] = float(_METRIC_FUNCTIONS[name](obs, sim, dates=dates, resolution=resolution))
        else:
            values[name] = float(_METRIC_FUNCTIONS[name](obs, sim))
    return values


def aggregate_basin_metrics(per_basin: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
    """Median (``<metric>``) and mean (``<metric>_mean``) over basins, ignoring NaN, plus counts.

    ``n_basins`` is the number of basins scored and ``n_basins_nse_le_0`` the number with NSE <= 0
    (Kratzert et al. 2019, Table 2).
    """
    names = []
    for values in per_basin.values():
        for name in values:
            if name not in names:
                names.append(name)
    summary: Dict[str, float] = {}
    for name in names:
        column = np.array([float(values.get(name, np.nan)) for values in per_basin.values()], dtype=np.float64)
        finite = column[~np.isnan(column)]
        summary[name] = float(np.median(finite)) if finite.size else float("nan")
        summary[f"{name}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
    summary["n_basins"] = float(len(per_basin))
    if "nse" in names:
        nse_values = np.array([float(values.get("nse", np.nan)) for values in per_basin.values()])
        summary["n_basins_nse_le_0"] = float(np.sum(nse_values <= 0))
    return summary


def streamflow_metric_names() -> list:
    """Every key ``aggregate_basin_metrics`` reports for the full metric set."""
    names = []
    for name in STREAMFLOW_METRICS:
        names.extend([name, f"{name}_mean"])
    return names + ["n_basins", "n_basins_nse_le_0"]


__all__ = [
    "NEURALHYDROLOGY_METRIC_NAMES",
    "STREAMFLOW_METRICS",
    "aggregate_basin_metrics",
    "alpha_nse",
    "beta_kge",
    "beta_nse",
    "calculate_metrics",
    "fdc_fhv",
    "fdc_flv",
    "fdc_fms",
    "kge",
    "mean_absolute_percentage_peak_error",
    "mean_peak_timing",
    "missed_peaks",
    "mse",
    "nse",
    "pearson_r",
    "rmse",
    "streamflow_metric_names",
]
