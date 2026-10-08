"""Pick-extraction utilities checked against the code the original pickers use.

- ``detect_peaks`` against the copies shipped with PhaseNet (``phasenet/detect_peaks.py``, Duarte's
  version 1.0.6) and EQTransformer (``EqT_utils._detect_peaks``), both pinned in repos.yaml.
- ``trigger_onset`` (reimplemented from ObsPy's documentation, ObsPy being LGPL) against ObsPy 1.5.1,
  which EQTransformer and gpd_predict.py import.
- ``score_picks`` against the official PhaseNet evaluation, ``correct_picks`` + ``metrics`` in
  ``phasenet/util.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from oracle_utils import load_definitions, oracle_package, oracle_repo
from pyhazards.metrics.picking import detect_peaks, score_picks, trigger_onset


def _random_traces(seed: int, count: int = 400):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        n = int(rng.integers(0, 300))
        x = rng.random(n)
        if rng.random() < 0.3:
            x = np.round(x, 1)  # plateaus and ties
        if rng.random() < 0.1 and n > 5:
            x[rng.integers(0, n)] = np.nan
        yield x


def _eqt_detect_peaks():
    source = oracle_repo("EQTransformer") / "EQTransformer" / "core" / "EqT_utils.py"
    return load_definitions(source, ["_detect_peaks"], {"np": np})["_detect_peaks"]


def test_detect_peaks_matches_phasenet_copy():
    official = load_definitions(
        oracle_repo("PhaseNet") / "phasenet" / "detect_peaks.py", ["detect_peaks"], {"np": np, "warnings": warnings}
    )["detect_peaks"]
    rng = np.random.default_rng(0)
    for x in _random_traces(1):
        kwargs = dict(mph=float(rng.choice([0.3, 0.5, 0.9])), mpd=int(rng.choice([1, 2, 5, 50])))
        if rng.random() < 0.3:
            kwargs["threshold"] = 0.05
        if rng.random() < 0.3:
            kwargs["edge"] = str(rng.choice(["both", "falling"]))
        result = official(x.copy(), show=False, **kwargs)
        expected = result[0] if isinstance(result, tuple) else result
        np.testing.assert_array_equal(detect_peaks(x, **kwargs), expected)


def test_detect_peaks_matches_eqtransformer_copy():
    official = _eqt_detect_peaks()
    rng = np.random.default_rng(1)
    for x in _random_traces(2):
        kwargs = dict(mph=float(rng.choice([0.1, 0.3])), mpd=int(rng.choice([1, 3])))
        np.testing.assert_array_equal(detect_peaks(x, **kwargs), official(x.copy(), **kwargs))


def test_trigger_onset_matches_obspy():
    oracle_package("obspy", "1.5.1", "requirements-earthquake.txt")
    from obspy.signal.trigger import trigger_onset as obspy_trigger_onset

    rng = np.random.default_rng(3)
    for x in _random_traces(4, count=2000):
        x = np.nan_to_num(x)
        thres1 = float(np.round(rng.random(), 2))
        thres2 = float(np.round(rng.random() * thres1, 2))
        expected = np.array(obspy_trigger_onset(x, thres1, thres2), dtype=int).reshape(-1, 2)
        np.testing.assert_array_equal(trigger_onset(x, thres1, thres2), expected)
    # The thresholds of EQTransformer (equal on/off) and of gpd_predict.py (0.95 / 0.1).
    x = rng.random(5000)
    for thres1, thres2 in ((0.2, 0.2), (0.95, 0.1)):
        expected = np.array(obspy_trigger_onset(x, thres1, thres2), dtype=int).reshape(-1, 2)
        np.testing.assert_array_equal(trigger_onset(x, thres1, thres2), expected)


def test_score_picks_matches_official_phasenet_evaluation():
    source = oracle_repo("PhaseNet") / "phasenet" / "util.py"
    # correct_picks reads dt from DataConfig (0.01 s); provide just that.
    defs = load_definitions(source, ["correct_picks", "metrics"], {"np": np, "DataConfig": lambda: type("C", (), {"dt": 0.01})()})
    rng = np.random.default_rng(5)
    picks, true_p, true_s = [], [], []
    for _ in range(300):
        tp, ts = [float(rng.integers(100, 2000))], [float(rng.integers(2000, 2900))]
        pred_p = [tp[0] + float(rng.normal(0, 15)) for _ in range(int(rng.integers(0, 3)))]
        pred_s = [ts[0] + float(rng.normal(0, 30)) for _ in range(int(rng.integers(0, 3)))] + [float(rng.integers(0, 3000))]
        picks.append([[pred_p], [pred_s]])
        true_p.append(tp)
        true_s.append(ts)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tp_p, tp_s, np_p, np_s, nt_p, nt_s, diff_p, diff_s = defs["correct_picks"](picks, true_p, true_s, tol=0.1)
    for phase, (tp, n_pred, n_true, diffs, manual) in {
        "P": (tp_p, np_p, nt_p, diff_p, true_p),
        "S": (tp_s, np_s, nt_s, diff_s, true_s),
    }.items():
        column = 0 if phase == "P" else 1
        predicted = [trace[column][0] for trace in picks]
        score = score_picks(predicted, manual, tolerance=10.0, residual_window=50.0)
        assert (score.n_tp, score.n_pred, score.n_true) == (tp, n_pred, n_true)
        np.testing.assert_allclose(sorted(score.residuals), sorted(np.concatenate([d.ravel() for d in diffs])))
        precision, recall, f1 = defs["metrics"](tp, n_pred, n_true)
        values = score.metrics(100.0)
        assert values["precision"] == pytest.approx(precision)
        assert values["recall"] == pytest.approx(recall)
        assert values["f1"] == pytest.approx(f1)
