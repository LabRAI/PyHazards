"""SmokeBench metrics: accuracy, mean IoU, and the smoke-area / Weber-contrast bins.

Paper (Qi, Li & Barnes, WACV 2026, Sec. 4.2 and 4.4):

* accuracy = mean of ``1[pred == label]`` with ``True -> 1`` and ``False -> 0``. Table 1 reports it
  on the smoke images only (per area bin and overall), Table 3 on the 1,000 smoke-free images;
* mIoU = mean over images of ``|P n G| / |P u G|``. For the tile and grid tasks ``P`` and ``G`` are
  sets of tile / cell ids (:func:`set_iou`). For detection the authors' script scores an image by the
  largest pairwise IoU between a predicted and a ground-truth box (:func:`detection_iou`);
* smoke area = pixel area of the ground-truth box, binned at the quintiles of the 5,046 FIgLib
  boxes: ``(42, 4356], (4356, 11232], (11232, 28565], (28565, 83157], (83157, 2232020]``. These edges
  are exactly the quintiles of the SmokeyNet boxes (checked in tests/oracle/test_smokebench_oracle.py);
* Weber contrast ``|I_smoke - I_background| / I_background``, binned at its quintiles over the same
  boxes. The paper prints them to 3 significant digits, ``(0, 0.0246], (0.0246, 0.0549],
  (0.0549, 0.0878], (0.0878, 0.13], (0.13, 0.728]``; :data:`CONTRAST_BIN_EDGES` holds the full-precision
  quintiles that PyHazards computes on the 5,046 SmokeBench images, which round to exactly those
  values (so the contrast definition below is the authors').

The paper does not define the background region or how the area of several boxes is combined; both
follow the authors' ``summary_evaluation.py`` (commit 183cdca, test oracle only): the area is that of
the first box, and the background is a 20-pixel frame around each box in the OpenCV grey image
(pixels equal to 0 are ignored), averaged over boxes.

:func:`box_iou` is a port of ``torchvision.ops.box_iou`` for ``(N, 4)`` xyxy boxes (torchvision,
BSD-3-Clause, Copyright (c) Soumith Chintala 2016, torchvision/ops/boxes.py); torchvision is not a
PyHazards dependency.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

AREA_BIN_EDGES: Tuple[float, ...] = (42, 4356, 11232, 28565, 83157, 2232020)
AREA_BIN_LABELS: Tuple[str, ...] = ("Very Small", "Small", "Medium", "Large", "Very Large")
# Quintiles of weber_contrast over the 5,046 SmokeBench smoke images (FIgLib JPEGs decoded with
# Pillow 12.3 / libjpeg-turbo); the paper prints them as 0.0246, 0.0549, 0.0878, 0.13 and 0.728.
CONTRAST_BIN_EDGES: Tuple[float, ...] = (
    0.0,
    0.02461763919669502,
    0.054876825799815156,
    0.08778381526434637,
    0.1297575902787221,
    0.7280443703122176,
)
CONTRAST_BIN_EDGES_PRINTED: Tuple[float, ...] = (0.0, 0.0246, 0.0549, 0.0878, 0.13, 0.728)
CONTRAST_BIN_LABELS: Tuple[str, ...] = ("Very Low", "Low", "Medium", "High", "Very High")
CONTRAST_PAD = 20


def accuracy(predictions: Sequence[bool], labels: Sequence[bool]) -> float:
    if len(predictions) != len(labels):
        raise ValueError(f"{len(predictions)} predictions for {len(labels)} labels")
    if not labels:
        raise ValueError("accuracy of an empty set is undefined")
    return float(np.mean([bool(p) == bool(y) for p, y in zip(predictions, labels)]))


def set_iou(predicted: Iterable[int], target: Iterable[int]) -> float:
    """``|P n G| / |P u G|`` for sets of tile or cell ids (raises when both are empty)."""
    pred, true = set(predicted), set(target)
    union = pred | true
    if not union:
        raise ValueError("IoU of two empty sets is undefined")
    return len(pred & true) / len(union)


def _upcast(t: torch.Tensor) -> torch.Tensor:
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    return t if t.dtype in (torch.int32, torch.int64) else t.int()


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU ``(N, M)`` of xyxy boxes ``(N, 4)`` and ``(M, 4)`` (torchvision semantics)."""
    for name, boxes in (("boxes1", boxes1), ("boxes2", boxes2)):
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError(f"{name} must have shape (N, 4), got {tuple(boxes.shape)}")
    area1 = _box_area(boxes1)
    area2 = _box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = _upcast(rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union


def detection_iou(predicted_boxes: Sequence, target_boxes: Sequence) -> Tuple[float, Optional[str]]:
    """Per-image detection score: the largest pairwise IoU (``np.max(box_iou(P, G))``).

    ``predicted_boxes`` are the raw JSON values from :func:`~pyhazards.prompted.parsers.parse_detection`.
    Returns ``(score, problem)``; ``problem`` is ``None`` for well-formed ``(N, 4)`` numeric boxes.
    Boxes that are not ``(N, 4)`` numbers score 0 (the authors' script scores fewer than four
    coordinates 0 and crashes on other malformed values); a NaN IoU (zero union) also scores 0.
    """
    target = torch.tensor([np.asarray(box).tolist() for box in target_boxes])
    try:
        pred = torch.tensor(predicted_boxes)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        return 0.0, "non_numeric"
    if pred.ndim != 2 or pred.shape[-1] != 4:
        return 0.0, "malformed"
    score = float(np.max(box_iou(pred, target).tolist()))
    if math.isnan(score):
        return 0.0, "nan"
    return score, None


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean of an empty set is undefined")
    return float(np.mean(values))


def box_area(box: Sequence[float]) -> float:
    """``(x2 - x1) * (y2 - y1)`` of one box."""
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def smoke_area(boxes: Sequence[Sequence[float]]) -> float:
    """SmokeBench smoke area: the area of the first ground-truth box."""
    if not boxes:
        raise ValueError("a smoke image needs at least one box")
    return box_area(boxes[0])


def opencv_gray(image: np.ndarray) -> np.ndarray:
    """Grey level of an RGB uint8 image exactly as ``cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)``.

    OpenCV uses 15-bit fixed point: ``(9798 R + 19235 G + 3735 B + 2**14) >> 15`` (checked against
    OpenCV 4.10 on all 2**24 colours in the oracle test).
    """
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError(f"expected an RGB uint8 image of shape (H, W, 3), got {array.shape} {array.dtype}")
    rgb = array.astype(np.int64)
    gray = (rgb[..., 0] * 9798 + rgb[..., 1] * 19235 + rgb[..., 2] * 3735 + (1 << 14)) >> 15
    return gray.astype(np.uint8)


def weber_contrast(image: np.ndarray, boxes: Sequence[Sequence[int]], pad: int = CONTRAST_PAD) -> float:
    """Mean over boxes of ``|mean(box) - mean(frame)| / mean(frame)`` on the OpenCV grey image.

    The frame is the box grown by ``pad`` pixels (clipped to the image) minus the box; grey value 0
    pixels are excluded from it. Boxes with an empty frame or a zero frame mean are skipped; 0 if
    none is left. Box coordinates are integer pixel indices.
    """
    gray = opencv_gray(image)
    height, width = gray.shape
    ratios: List[float] = []
    for x1, y1, x2, y2 in (tuple(int(v) for v in box) for box in boxes):
        smoke_mean = float(np.mean(gray[y1:y2, x1:x2]))
        bx1, by1 = max(0, x1 - pad), max(0, y1 - pad)
        bx2, by2 = min(width, x2 + pad), min(height, y2 + pad)
        frame = gray[by1:by2, bx1:bx2].copy()
        frame[(y1 - by1):(y2 - by1), (x1 - bx1):(x2 - bx1)] = 0
        pixels = frame[frame > 0]
        if len(pixels) == 0:
            continue
        background = float(np.mean(pixels))
        if background != 0:
            ratios.append(abs(smoke_mean - background) / background)
    return float(np.mean(ratios)) if ratios else 0.0


def assign_bin(value: float, edges: Sequence[float]) -> int:
    """Index of the right-closed bin ``(edges[k], edges[k+1]]`` holding ``value``.

    The first bin also takes values at or below ``edges[0]`` and the last bin values above
    ``edges[-1]``, as ``pandas.qcut`` assigns the minimum and maximum of the data it was fitted on.
    """
    for k in range(len(edges) - 2):
        if value <= edges[k + 1]:
            return k
    return len(edges) - 2


def quantile_edges(values: Sequence[float], q: int = 5) -> List[float]:
    """Quantile edges as ``pandas.qcut(values, q)`` computes them (linear interpolation)."""
    if not values:
        raise ValueError("quantiles of an empty set are undefined")
    return [float(v) for v in np.quantile(np.asarray(values, dtype=float), np.linspace(0, 1, q + 1))]


def binned_means(values: Sequence[float], keys: Sequence[float], edges: Sequence[float], labels: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Mean of ``values`` per bin of ``keys``: ``{label: {"mean": m, "n": n}}`` (empty bins: mean None)."""
    if len(values) != len(keys):
        raise ValueError("values and keys must have the same length")
    groups: Dict[int, List[float]] = {k: [] for k in range(len(labels))}
    for value, key in zip(values, keys):
        groups[assign_bin(key, edges)].append(float(value))
    return {
        labels[k]: {"mean": (float(np.mean(group)) if group else None), "n": len(group)}
        for k, group in groups.items()
    }


__all__ = [
    "AREA_BIN_EDGES",
    "AREA_BIN_LABELS",
    "CONTRAST_BIN_EDGES",
    "CONTRAST_BIN_EDGES_PRINTED",
    "CONTRAST_BIN_LABELS",
    "CONTRAST_PAD",
    "accuracy",
    "assign_bin",
    "binned_means",
    "box_area",
    "box_iou",
    "detection_iou",
    "mean",
    "opencv_gray",
    "quantile_edges",
    "set_iou",
    "smoke_area",
    "weber_contrast",
]
