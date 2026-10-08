"""Per-pixel logistic regression over a local neighbourhood.

WildfireSpreadTS (Gerard et al., NeurIPS 2023 D&B, ``src/models/LogisticRegression.py``, MIT)
implements its logistic-regression baseline as a single ``Conv2d(C, 1, kernel_size=3, padding=1)``:
every output pixel is a linear function of the ``3 x 3 x C`` input window followed by a sigmoid
in the loss. Next Day Wildfire Spread (Huot et al., IEEE TGRS 2022) fits the same model with
scikit-learn on the 3 x 3 neighbourhood of each pixel; only the fitting procedure differs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PixelLogisticRegression(nn.Module):
    """Linear ``kernel_size x kernel_size`` window classifier returning logits.

    Accepts ``(batch, channels, height, width)`` or ``(batch, time, channels, height, width)``;
    the latter is flattened to ``time * channels`` input channels, as in WildfireSpreadTS.
    """

    def __init__(self, in_channels: int, out_channels: int = 1, kernel_size: int = 3):
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        self.in_channels = int(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            x = x.flatten(start_dim=1, end_dim=2)
        if x.ndim != 4:
            raise ValueError(
                "PixelLogisticRegression expects input shape (batch, channels, height, width) or "
                f"(batch, time, channels, height, width), got {tuple(x.shape)}."
            )
        if x.size(1) != self.in_channels:
            raise ValueError(f"PixelLogisticRegression expected {self.in_channels} channels, got {x.size(1)}.")
        return self.conv(x)


def logistic_regression_builder(
    task: str,
    in_channels: int,
    out_channels: int = 1,
    history: int = 1,
    kernel_size: int = 3,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() != "segmentation":
        raise ValueError(f"logistic_regression supports task='segmentation', got {task!r}.")
    if history <= 0:
        raise ValueError(f"history must be positive, got {history}")
    return PixelLogisticRegression(in_channels * history, out_channels, kernel_size)


__all__ = ["PixelLogisticRegression", "logistic_regression_builder"]
