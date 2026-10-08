"""Seismic phase-pick extraction and pick metrics (earthquake.picking).

Pick extraction
    :func:`detect_peaks` is a port of ``detect_peaks`` by Marcos Duarte (version 1.0.6, MIT License,
    https://github.com/demotu/BMC), the copy that ships with PhaseNet (``phasenet/detect_peaks.py``) and,
    slightly modified, with EQTransformer (``EqT_utils._detect_peaks``). It returns peak indices only.
    :func:`trigger_onset` reimplements the documented behaviour of ObsPy's ``trigger_onset`` (ObsPy is
    LGPL; no code was copied). It was checked against ObsPy 1.5.1 on random inputs
    (``tests/oracle/test_picking_metrics_oracle.py``).

Scoring (:func:`score_picks`)
    Follows the official PhaseNet evaluation (``correct_picks`` and ``metrics`` in ``phasenet/util.py``,
    AI4EPS/PhaseNet, MIT License, Copyright (c) 2021 Weiqiang Zhu), which produced Table 1 of Zhu &
    Beroza (2019): every pair (predicted pick, manual pick) of the same phase whose residual is within
    ``tolerance`` counts as a true positive; precision = TP / number of predicted picks; recall =
    TP / number of manual picks; F1 = 2PR / (P + R). Residual statistics are taken over the pairs with
    ``|residual| < residual_window``. Residuals are ``predicted - manual`` and reported in seconds.

    Protocols (:data:`PICKING_PROTOCOLS`):

    - ``"phasenet"``: Zhu & Beroza (2019), GJI 216:261-273: a pick is correct when ``|dt| < 0.1 s``;
      mean and standard deviation of residuals over ``|dt| < 0.5 s``.
    - ``"eqtransformer"``: Mousavi et al. (2020), Nat. Commun. 11:3952: a pick is a true positive when
      its absolute distance from the ground truth is less than 0.5 s; mean, standard deviation and mean
      absolute error over the true positives.

    Undefined ratios (no predicted or no manual picks) are reported as 0; residual statistics without
    any residual are NaN. EQTransformer's MAPE is not computed: the paper does not define its
    denominator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PICKING_PROTOCOLS: Dict[str, Dict[str, float]] = {
    "phasenet": {"tolerance_s": 0.1, "residual_window_s": 0.5},
    "eqtransformer": {"tolerance_s": 0.5, "residual_window_s": 0.5},
}

# One trace's picks: phase name -> list of (sample index, probability).
TracePicks = Dict[str, List[Tuple[float, float]]]


def detect_peaks(
    x: Sequence[float],
    mph: Optional[float] = None,
    mpd: int = 1,
    threshold: float = 0.0,
    edge: Optional[str] = "rising",
    kpsh: bool = False,
    valley: bool = False,
) -> np.ndarray:
    """Indices of the peaks of ``x`` (port of Duarte's ``detect_peaks`` 1.0.6, MIT).

    ``mph``: minimum peak height; ``mpd``: minimum peak distance in samples (smaller peaks within
    ``mpd`` of a higher one are removed); ``threshold``: minimum height above both neighbours;
    ``edge``: which edge of a flat peak to keep (``"rising"``, ``"falling"``, ``"both"`` or ``None``);
    ``kpsh``: keep peaks of the same height even when closer than ``mpd``; ``valley``: detect minima.
    """
    x = np.atleast_1d(np.asarray(x, dtype="float64")).copy()
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
        if mph is not None:
            mph = -mph
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ["rising", "both"]:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ["falling", "both"]:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # NaN's and values next to NaN's cannot be peaks
    if ind.size and indnan.size:
        ind = ind[np.isin(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbours < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # remove small peaks closer than the minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # keep the current peak
        ind = np.sort(ind[~idel])
    return ind


def trigger_onset(charfct: Sequence[float], thres1: float, thres2: float) -> np.ndarray:
    """``(n, 2)`` array of ``[on, off]`` sample pairs of a two-threshold trigger.

    A trigger switches on at the first sample with ``charfct >= thres1`` and stays on while
    ``charfct >= thres2``; ``off`` is the last sample of that run (the last sample of ``charfct`` when
    the trigger is still on at the end). Same results as ObsPy's ``trigger_onset`` (without its
    ``max_len`` options) for ``thres2 <= thres1``.
    """
    if thres2 > thres1:
        raise ValueError(f"trigger_onset needs thres2 <= thres1, got thres1={thres1}, thres2={thres2}.")
    x = np.asarray(charfct, dtype="float64").ravel()
    on_mask, stay_mask = x >= thres1, x >= thres2
    triggers: List[Tuple[int, int]] = []
    start = 0
    while start < x.size:
        candidates = np.flatnonzero(on_mask[start:])
        if not candidates.size:
            break
        on = start + int(candidates[0])
        below = np.flatnonzero(~stay_mask[on:])
        off = on + int(below[0]) - 1 if below.size else x.size - 1
        triggers.append((on, off))
        start = off + 1
    return np.array(triggers, dtype=int).reshape(-1, 2)


def peak_picks(probability: Sequence[float], threshold: float, min_distance: int) -> List[Tuple[float, float]]:
    """``(sample, probability)`` of the peaks of a probability trace above ``threshold``.

    The PhaseNet rule (``detect_peaks(prob, mph=threshold, mpd=min_distance)``).
    """
    trace = np.asarray(probability, dtype="float64")
    indices = detect_peaks(trace, mph=threshold, mpd=int(min_distance))
    return [(float(i), float(trace[i])) for i in indices]


@dataclass
class PhaseScore:
    """Pick counts and residuals (in samples, ``predicted - manual``) of one phase."""

    n_true: int = 0
    n_pred: int = 0
    n_tp: int = 0
    residuals: List[float] = field(default_factory=list)

    def metrics(self, sampling_rate: float) -> Dict[str, float]:
        precision = self.n_tp / self.n_pred if self.n_pred else 0.0
        recall = self.n_tp / self.n_true if self.n_true else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        seconds = np.asarray(self.residuals, dtype="float64") / float(sampling_rate)
        if seconds.size:
            mean, std, mae = float(seconds.mean()), float(seconds.std()), float(np.abs(seconds).mean())
        else:
            mean = std = mae = math.nan
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "residual_mean": mean,
            "residual_std": std,
            "mae": mae,
        }


def score_picks(
    predicted: Iterable[Sequence[float]],
    manual: Iterable[Sequence[float]],
    tolerance: float,
    residual_window: float,
) -> PhaseScore:
    """Score one phase over traces (sample units).

    ``predicted[i]`` and ``manual[i]`` are the predicted and the manual pick samples of trace ``i``
    (NaN entries are ignored). Counting follows the official PhaseNet evaluation: TP = number of
    (prediction, manual pick) pairs with ``|residual| < tolerance``; residuals are kept for pairs
    with ``|residual| < residual_window``.
    """
    score = PhaseScore()
    for pred, true in zip(predicted, manual):
        pred = np.asarray([p for p in pred if not math.isnan(float(p))], dtype="float64")
        true = np.asarray([t for t in true if not math.isnan(float(t))], dtype="float64")
        score.n_true += int(true.size)
        score.n_pred += int(pred.size)
        if not pred.size or not true.size:
            continue
        diff = pred[None, :] - true[:, None]
        score.n_tp += int(np.sum(np.abs(diff) < tolerance))
        score.residuals.extend(diff[np.abs(diff) < residual_window].tolist())
    return score


def detection_scores(detected: Sequence[bool], is_event: Sequence[bool]) -> Dict[str, float]:
    """Trace-level detection precision, recall and F1 (EQTransformer Table 1)."""
    detected = np.asarray(detected, dtype=bool)
    is_event = np.asarray(is_event, dtype=bool)
    tp = int(np.sum(detected & is_event))
    n_pred, n_true = int(detected.sum()), int(is_event.sum())
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_true if n_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


__all__ = [
    "PICKING_PROTOCOLS",
    "PhaseScore",
    "TracePicks",
    "detect_peaks",
    "detection_scores",
    "peak_picks",
    "score_picks",
    "trigger_onset",
]
