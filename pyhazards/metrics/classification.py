# pyhazards/metrics/classification.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


@dataclass(frozen=True)
class ClassificationResults:
    accuracy: float
    report_dict: Dict
    report_csv: str
    confusion: np.ndarray
    confusion_normalized: np.ndarray
    labels: List[str]


def _safe_row_normalize(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    # avoid division by zero for classes absent in test set
    row_sums[row_sums == 0.0] = 1.0
    return cm / row_sums


def evaluate_multiclass(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: Sequence[str],
    *,
    digits: int = 4,
    zero_division: int = 0,
) -> ClassificationResults:
    """
    Strict: metrics computed ONLY from the provided y_true/y_pred (caller must ensure test split).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))

    # sklearn report with per-class precision/recall/F1 + macro/weighted
    rep = classification_report(
        y_true,
        y_pred,
        target_names=list(label_names),
        digits=digits,
        output_dict=True,
        zero_division=zero_division,
    )

    rep_csv = classification_report(
        y_true,
        y_pred,
        target_names=list(label_names),
        digits=digits,
        output_dict=False,
        zero_division=zero_division,
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    cm_norm = _safe_row_normalize(cm)

    return ClassificationResults(
        accuracy=acc,
        report_dict=rep,
        report_csv=rep_csv,
        confusion=cm,
        confusion_normalized=cm_norm,
        labels=list(label_names),
    )
