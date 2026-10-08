"""Wavefield-forecast metrics of WaveCastNet (earthquake.forecasting).

Ports of ``validation_acc``, ``validation_rfne`` and ``validation_rmse`` in
``src/models_earthquake/Validation_pixel.py`` of dwlyu/WaveCastNet at commit
``c859e04c85acd6657306f08bce8c6f9129ba1480`` (MIT License, Copyright (c) 2023 dwlyu), which compute
equations (17) and (18) of Lyu et al., "Rapid wavefield forecasting for earthquake early warning via deep
sequence to sequence learning", Nature Communications 16:10622 (2025):

- ACC = sum_{t,h,w} pred * target / sqrt(sum_{t,h,w} target^2 * sum_{t,h,w} pred^2)
- RFNE = ||pred - target||_F / ||target||_F
- RMSE = sqrt(mean_{t,h,w} (pred - target)^2)

Each value is computed for every (sample, channel) pair over the time and space axes of
``(batch, channels, time, height, width)`` tensors; the official validation then averages all pairs
(:func:`wavefield_metrics` returns that mean and the per-channel means). Like the official code, nothing
guards against a zero norm: an all-zero target channel gives NaN or infinity.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch


def _check(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.shape != target.shape or target.ndim != 5:
        raise ValueError(
            "Wavefield metrics expect pred and target shaped (batch, channels, time, height, width); "
            f"got shapes {tuple(pred.shape)} and {tuple(target.shape)}."
        )


def wavefield_acc(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """ACC (eq. 17) of every (sample, channel) pair: a ``(batch, channels)`` tensor."""
    _check(pred, target)
    pred, target = pred.flatten(2), target.flatten(2)
    sum1 = (pred * target).sum(dim=-1)
    sum2 = (target * target).sum(dim=-1)
    sum3 = (pred * pred).sum(dim=-1)
    return sum1 / torch.sqrt(sum2 * sum3)


def wavefield_rfne(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """RFNE (eq. 18) of every (sample, channel) pair: a ``(batch, channels)`` tensor."""
    _check(pred, target)
    return torch.linalg.vector_norm((target - pred).flatten(2), dim=-1) / torch.linalg.vector_norm(
        target.flatten(2), dim=-1
    )


def wavefield_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """RMSE of every (sample, channel) pair: a ``(batch, channels)`` tensor."""
    _check(pred, target)
    return torch.sqrt(((pred - target) ** 2).flatten(2).mean(dim=-1))


def wavefield_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    channel_names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """ACC, RFNE and RMSE averaged over all (sample, channel) pairs, as the official validation does,
    plus ``acc_<channel>`` and ``rfne_<channel>`` averaged over the samples of each channel."""
    acc, rfne, rmse = (fn(pred, target).double() for fn in (wavefield_acc, wavefield_rfne, wavefield_rmse))
    names = list(channel_names) if channel_names is not None else [f"c{i}" for i in range(acc.shape[1])]
    if len(names) != acc.shape[1]:
        raise ValueError(f"Got {len(names)} channel names for {acc.shape[1]} channels.")
    metrics = {"acc": float(acc.mean()), "rfne": float(rfne.mean()), "rmse": float(rmse.mean())}
    for index, name in enumerate(names):
        metrics[f"acc_{name}"] = float(acc[:, index].mean())
        metrics[f"rfne_{name}"] = float(rfne[:, index].mean())
    return metrics


__all__ = ["wavefield_acc", "wavefield_metrics", "wavefield_rfne", "wavefield_rmse"]
