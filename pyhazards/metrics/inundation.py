"""Flood inundation metrics: depth errors, wet-area overlap and the scores of the flood neural-operator papers.

``relative_l2``, ``nash_sutcliffe``, ``pearson_correlation`` and ``critical_success_index`` are defined as
in the UrbanFloodCast evaluation (Xu et al., J. Hydrology 2025; ``DNO/utils25.py`` of
HydroPML/UrbanFloodCast): per sample (one event), the relative L2 error of every variable over all
cells and time steps averaged over the variables (FNO's relative ``LpLoss``), and NSE / Pearson r over all
values of the sample; Pearson r divides the (population) covariance by the product of the sample
standard deviations, as the reference does. The CSI counts cells where both depths exceed the threshold
(hits) against misses and false alarms, ``hits / (hits + misses + false_alarms + 1e-8)``. The definitions
were written from the formulas and are checked against the reference (``tests/oracle``).

:func:`inundation_metrics` reports, for a set of samples:

- ``pixel_mae`` and ``rmse``: mean absolute / root mean squared depth error over every cell of every
  sample (PyHazards);
- ``iou`` and ``f1``: overlap of the predicted (depth >= 0.5) and observed (depth > 0) wet cells over
  all samples (PyHazards' earlier inundation score, kept for comparability);
- ``csi_1cm``, ``csi_10cm``, ``csi_50cm``: depth CSI at 0.01 / 0.1 / 0.5 m, averaged over samples;
- ``relative_l2``, ``nse``, ``pearson_r``: per sample over all predicted variables, averaged over samples.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

CSI_THRESHOLDS = {"csi_1cm": 0.01, "csi_10cm": 0.1, "csi_50cm": 0.5}
INUNDATION_METRICS = ("pixel_mae", "rmse", "iou", "f1", *CSI_THRESHOLDS, "relative_l2", "nse", "pearson_r")


def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Relative L2 error per sample: ``mean_v ||pred[:, :, v] - target[:, :, v]|| / ||target[:, :, v]||``.

    Inputs are ``(batch, points, variables)``; returns ``(batch,)``.
    """
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError(
            f"relative_l2 expects pred and target of the same shape (batch, points, variables); got "
            f"{tuple(pred.shape)} and {tuple(target.shape)}."
        )
    diff_norms = torch.norm(pred - target, 2, 1)
    target_norms = torch.norm(target, 2, 1)
    return (diff_norms / target_norms).mean(-1)


def nash_sutcliffe(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """NSE over all values: ``1 - sum((target - pred)^2) / sum((target - mean(target))^2)``."""
    numerator = torch.sum((target - pred) ** 2)
    denominator = torch.sum((target - torch.mean(target)) ** 2)
    return 1.0 - numerator / denominator


def pearson_correlation(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``mean((t - mean t)(p - mean p)) / (std(t) std(p))`` over all values (std with Bessel's correction)."""
    mean_true = torch.mean(target)
    mean_pred = torch.mean(pred)
    cov = torch.mean((target - mean_true) * (pred - mean_pred))
    return cov / (torch.std(target) * torch.std(pred))


def critical_success_index(pred: torch.Tensor, target: torch.Tensor, threshold: float) -> torch.Tensor:
    """CSI of the cells deeper than ``threshold``: ``hits / (hits + misses + false_alarms + 1e-8)``."""
    hits = torch.sum(torch.logical_and(target > threshold, pred > threshold))
    misses = torch.sum(torch.logical_and(target > threshold, pred <= threshold))
    false_alarms = torch.sum(torch.logical_and(target <= threshold, pred > threshold))
    return hits / (hits + misses + false_alarms + 1e-8)


def inundation_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    depth_index: Optional[int] = None,
) -> Dict[str, float]:
    """Inundation metrics of ``n_samples`` predictions (first dimension = samples).

    With ``depth_index`` the last dimension holds several variables (e.g. depth and the two unit
    discharges of UrbanFloodCast) and depth is ``[..., depth_index]``; without it every value is a depth.
    ``mask`` (same shape, True = valid) multiplies predictions and targets before scoring, as the
    UrbanFloodCast evaluation does with its NaN mask.
    """
    pred = preds.detach().float().cpu()
    target = targets.detach().float().cpu()
    if tuple(pred.shape) != tuple(target.shape):
        raise ValueError(f"flood.inundation predictions {tuple(pred.shape)} must match targets {tuple(target.shape)}.")
    if pred.ndim < 2:
        raise ValueError(f"flood.inundation expects a leading sample dimension, got shape {tuple(pred.shape)}.")
    if mask is not None:
        mask = mask.detach().cpu()
        if tuple(mask.shape) != tuple(pred.shape):
            raise ValueError(f"mask {tuple(mask.shape)} must match predictions {tuple(pred.shape)}.")
        pred = pred * mask
        target = target * mask
    n_samples = pred.shape[0]
    if depth_index is None:
        pred_vars = pred.reshape(n_samples, -1, 1)
        target_vars = target.reshape(n_samples, -1, 1)
        pred_depth, target_depth = pred, target
    else:
        n_vars = pred.shape[-1]
        pred_vars = pred.reshape(n_samples, -1, n_vars)
        target_vars = target.reshape(n_samples, -1, n_vars)
        pred_depth, target_depth = pred[..., depth_index], target[..., depth_index]

    pred_mask = (pred_depth >= 0.5).float()
    target_mask = (target_depth > 0).float()
    intersection = (pred_mask * target_mask).sum()
    union = pred_mask.sum() + target_mask.sum() - intersection
    metrics = {
        "pixel_mae": float(torch.mean(torch.abs(pred_depth - target_depth))),
        "rmse": float(torch.sqrt(torch.mean((pred_depth - target_depth) ** 2))),
        "iou": float(intersection / union.clamp(min=1.0)),
        "f1": float(2 * intersection / (pred_mask.sum() + target_mask.sum()).clamp(min=1.0)),
    }
    per_sample: Dict[str, list] = {name: [] for name in (*CSI_THRESHOLDS, "relative_l2", "nse", "pearson_r")}
    for i in range(n_samples):
        depth_p = pred_depth[i].reshape(1, -1, 1)
        depth_t = target_depth[i].reshape(1, -1, 1)
        for name, threshold in CSI_THRESHOLDS.items():
            per_sample[name].append(float(critical_success_index(depth_p, depth_t, threshold)))
        per_sample["relative_l2"].append(float(relative_l2(pred_vars[i : i + 1], target_vars[i : i + 1]).sum()))
        per_sample["nse"].append(float(nash_sutcliffe(pred_vars[i : i + 1], target_vars[i : i + 1])))
        per_sample["pearson_r"].append(float(pearson_correlation(pred_vars[i : i + 1], target_vars[i : i + 1])))
    for name, values in per_sample.items():
        metrics[name] = float(sum(values) / len(values)) if values else float("nan")
    return metrics


def rollout_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """RMSE over nodes at every rollout step: ``(steps, nodes)`` -> ``(steps,)`` (HydroGraphNet ``inference.py``)."""
    if pred.shape != target.shape or pred.ndim != 2:
        raise ValueError(f"rollout_rmse expects (steps, nodes) tensors of equal shape, got {tuple(pred.shape)} and {tuple(target.shape)}.")
    return torch.sqrt(torch.mean((pred - target) ** 2, dim=1))


__all__ = [
    "CSI_THRESHOLDS",
    "INUNDATION_METRICS",
    "critical_success_index",
    "inundation_metrics",
    "nash_sutcliffe",
    "pearson_correlation",
    "relative_l2",
    "rollout_rmse",
]
