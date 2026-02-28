# pyhazards/metrics/forecasting.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


@dataclass(frozen=True)
class ForecastMetrics:
    rmse: float
    r2: float


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_forecast_buckets(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bucket_names: List[str],
) -> Dict[str, ForecastMetrics]:
    """
    y_true, y_pred: shape (T, K) where K = number of buckets (A,B,C,D,EFG).
    Returns metrics for each bucket + "all" (sum across buckets).
    STRICT: caller must pass TEST-ONLY arrays for the chosen region (US or CA).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    assert y_true.shape == y_pred.shape, "y_true and y_pred must match shape (T, K)."
    assert y_true.shape[1] == len(bucket_names), "K must equal len(bucket_names)."

    out: Dict[str, ForecastMetrics] = {}

    # per bucket
    for j, name in enumerate(bucket_names):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        out[name] = ForecastMetrics(rmse=_rmse(yt, yp), r2=float(r2_score(yt, yp)))

    # all = sum across buckets (paper-style)
    yt_all = y_true.sum(axis=1)
    yp_all = y_pred.sum(axis=1)
    out["all"] = ForecastMetrics(rmse=_rmse(yt_all, yp_all), r2=float(r2_score(yt_all, yp_all)))

    return out
