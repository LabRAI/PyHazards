"""CNN-ASPP for next-day wildfire spread (Marjani et al., IEEE GRSL 2024).

Reimplementation of the network described in Marjani, Mahdianpari, Ahmadi, Hemmati,
Mohammadimanesh & Mesgari, "Application of Explainable Artificial Intelligence in Predicting
Wildfire Spread: An ASPP-Enabled CNN Approach", IEEE Geoscience and Remote Sensing Letters 21
(2024) 2504005, doi:10.1109/LGRS.2024.3417624 (Section II-C and Fig. 2). No code was released.

Architecture (all convolutions keep the 64 x 64 resolution of Next Day Wildfire Spread tiles):
two 3x3 convolutions with 64 and 128 filters; four parallel 3x3 atrous convolutions with 32
filters and dilation rates 1, 3, 6 and 12, concatenated (128 channels); two 3x3 convolutions
with 32 filters; batch normalisation; a 1x1 convolution to one output. Every convolution
except the last uses ReLU. The paper applies a sigmoid in the last layer; this module returns
the logits and leaves the sigmoid to the loss. The paper trains with the Tversky loss
(alpha = 0.7 on false negatives, beta = 0.3 on false positives), batch size 8 and learning
rate 4e-4 in TensorFlow; Keras defaults (Glorot-uniform kernels, zero biases, BatchNorm
epsilon 1e-3 and momentum 0.99) are used for what the paper does not state.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class WildfireCNNASPP(nn.Module):
    """CNN-ASPP: input ``(batch, channels, height, width)``, output ``(batch, 1, height, width)`` logits."""

    def __init__(self, in_channels: int = 12, dilations: Sequence[int] = (1, 3, 6, 12)):
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        dilations = tuple(int(d) for d in dilations)
        if len(dilations) != 4 or min(dilations) <= 0:
            raise ValueError(f"dilations must be four positive integers, got {dilations}")
        self.in_channels = int(in_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.aspp = nn.ModuleList(
            nn.Sequential(nn.Conv2d(128, 32, kernel_size=3, padding=d, dilation=d), nn.ReLU())
            for d in dilations
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32 * len(dilations), 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # Keras BatchNormalization defaults: epsilon 1e-3, moving-average momentum 0.99.
            nn.BatchNorm2d(32, eps=1e-3, momentum=0.01),
        )
        self.head = nn.Conv2d(32, 1, kernel_size=1)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"WildfireCNNASPP expects input shape (batch, channels, height, width), got {tuple(x.shape)}."
            )
        if x.size(1) != self.in_channels:
            raise ValueError(f"WildfireCNNASPP expected {self.in_channels} channels, got {x.size(1)}.")
        features = self.encoder(x)
        features = torch.cat([branch(features) for branch in self.aspp], dim=1)
        return self.head(self.decoder(features))


def cnn_aspp_builder(
    task: str,
    in_channels: int = 12,
    dilations: Sequence[int] = (1, 3, 6, 12),
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() != "segmentation":
        raise ValueError(f"wildfire_aspp supports task='segmentation', got {task!r}.")
    return WildfireCNNASPP(in_channels=in_channels, dilations=dilations)


__all__ = ["WildfireCNNASPP", "cnn_aspp_builder"]
