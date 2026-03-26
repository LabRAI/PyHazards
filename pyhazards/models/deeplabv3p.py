from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ._wildfire_layers import ASPPBlock, ConvBlock, check_image_input


class TinyDeepLabV3P(nn.Module):
    """Compact DeepLabV3+ style wildfire segmentation baseline."""

    def __init__(self, in_channels: int = 1, hidden_dim: int = 32, out_dim: int = 1):
        super().__init__()
        self.in_channels = int(in_channels)
        self.encoder = ConvBlock(in_channels, hidden_dim)
        self.aspp = ASPPBlock(hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_image_input(x, self.in_channels, 'TinyDeepLabV3P')
        return self.decoder(self.aspp(self.encoder(x)))


def deeplabv3p_builder(task: str, in_channels: int = 1, hidden_dim: int = 32, out_dim: int = 1, **kwargs: Any) -> nn.Module:
    _ = kwargs
    if task.lower() not in {'segmentation', 'regression'}:
        raise ValueError(f"deeplabv3p supports task='segmentation' or 'regression', got {task!r}.")
    return TinyDeepLabV3P(in_channels=in_channels, hidden_dim=hidden_dim, out_dim=out_dim)


__all__ = ['TinyDeepLabV3P', 'deeplabv3p_builder']
