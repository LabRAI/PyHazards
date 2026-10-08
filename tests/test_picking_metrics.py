import math

import numpy as np
import pytest

from pyhazards.metrics.picking import (
    PICKING_PROTOCOLS,
    detect_peaks,
    detection_scores,
    peak_picks,
    score_picks,
    trigger_onset,
)


def test_detect_peaks_reference_examples():
    # Examples from the docstring of Duarte's detect_peaks.
    assert detect_peaks([0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0], mpd=2).tolist() == [1, 5, 9]
    assert detect_peaks([0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0], mpd=4).tolist() == [5]
    assert detect_peaks([0, 1, 1, 0, 1, 1, 0], edge="both").tolist() == [1, 2, 4, 5]
    assert detect_peaks([0, 1, 1, 0, 1, 1, 0]).tolist() == [1, 4]
    assert detect_peaks([-2, 1, -2, 2, 1, 1, 3, 0], threshold=2).tolist() == [1, 6]
    assert detect_peaks([0, 3, 0, 1, 0], mph=2).tolist() == [1]
    assert detect_peaks([0, 1]).tolist() == []
    x = np.array([0, 1, 0, np.nan, 0, 2, 0], dtype=float)
    assert detect_peaks(x).tolist() == [1, 5]


def test_trigger_onset_on_off_semantics():
    assert trigger_onset([0, 0, 1, 1, 1, 0, 0], 0.5, 0.5).tolist() == [[2, 4]]
    assert trigger_onset([1, 1, 0, 0, 1], 0.5, 0.5).tolist() == [[0, 1], [4, 4]]
    assert trigger_onset([0, 0.6, 0.4, 0.6, 0.2, 0], 0.5, 0.1).tolist() == [[1, 4]]
    assert trigger_onset([0, 0.5, 0.5, 0], 0.5, 0.5).tolist() == [[1, 2]]
    assert trigger_onset([0, 0.3, 0.6, 0.3, 0.1], 0.5, 0.2).tolist() == [[2, 3]]
    assert trigger_onset([0.2] * 5, 0.5, 0.5).shape == (0, 2)
    with pytest.raises(ValueError, match="thres2"):
        trigger_onset([0, 1], 0.2, 0.5)


def test_score_picks_follows_the_phasenet_counting():
    tolerance, window = 10, 50  # 0.1 s and 0.5 s at 100 Hz
    predicted = [[100.0, 400.0], [205.0], [], [math.nan]]
    manual = [[103.0], [230.0], [50.0], [math.nan]]
    score = score_picks(predicted, manual, tolerance, window)
    assert (score.n_true, score.n_pred, score.n_tp) == (3, 3, 1)
    assert sorted(score.residuals) == [-25.0, -3.0]
    metrics = score.metrics(100.0)
    assert metrics["precision"] == pytest.approx(1 / 3)
    assert metrics["recall"] == pytest.approx(1 / 3)
    assert metrics["f1"] == pytest.approx(1 / 3)
    assert metrics["residual_mean"] == pytest.approx(-0.14)
    assert metrics["mae"] == pytest.approx(0.14)


def test_empty_scores_are_zero_and_nan():
    metrics = score_picks([[]], [[math.nan]], 10, 50).metrics(100.0)
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0.0
    assert math.isnan(metrics["residual_mean"]) and math.isnan(metrics["mae"])


def test_protocols_and_detection_scores():
    assert PICKING_PROTOCOLS["phasenet"] == {"tolerance_s": 0.1, "residual_window_s": 0.5}
    assert PICKING_PROTOCOLS["eqtransformer"]["tolerance_s"] == 0.5
    scores = detection_scores([True, True, False, False], [True, False, True, False])
    assert scores == {"precision": 0.5, "recall": 0.5, "f1": 0.5}
    assert peak_picks([0, 0.2, 0.9, 0.1, 0], 0.5, 1) == [(2.0, 0.9)]
