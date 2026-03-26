from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ._wildfire_layers import ConvBlock, Downsample, Upsample, check_image_input


class TinyUNet(nn.Module):
    """Compact U-Net baseline for wildfire raster prediction."""

    def __init__(self, in_channels: int = 1, base_channels: int = 16, out_dim: int = 1):
        super().__init__()
        self.in_channels = int(in_channels)
        self.stem = ConvBlock(in_channels, base_channels)
        self.down1 = Downsample(base_channels, base_channels * 2)
        self.down2 = Downsample(base_channels * 2, base_channels * 4)
        self.up1 = Upsample(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up2 = Upsample(base_channels * 2, base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_image_input(x, self.in_channels, 'TinyUNet')
        s1 = self.stem(x)
        s2 = self.down1(s1)
        bottleneck = self.down2(s2)
        x = self.up1(bottleneck, s2)
        x = self.up2(x, s1)
        return self.head(x)


def unet_builder(task: str, in_channels: int = 1, base_channels: int = 16, out_dim: int = 1, **kwargs: Any) -> nn.Module:
    _ = kwargs
    if task.lower() not in {'segmentation', 'regression'}:
        raise ValueError(f"unet supports task='segmentation' or 'regression', got {task!r}.")
    return TinyUNet(in_channels=in_channels, base_channels=base_channels, out_dim=out_dim)


__all__ = ['TinyUNet', 'unet_builder']
