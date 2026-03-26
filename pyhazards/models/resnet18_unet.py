from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ._wildfire_layers import ConvBlock, Downsample, ResidualBlock, Upsample, check_image_input


class TinyResNet18UNet(nn.Module):
    """Residual encoder-decoder baseline inspired by ResNet18 U-Net."""

    def __init__(self, in_channels: int = 1, base_channels: int = 16, out_dim: int = 1):
        super().__init__()
        self.in_channels = int(in_channels)
        self.stem = ConvBlock(in_channels, base_channels)
        self.res1 = ResidualBlock(base_channels)
        self.down1 = Downsample(base_channels, base_channels * 2)
        self.res2 = ResidualBlock(base_channels * 2)
        self.down2 = Downsample(base_channels * 2, base_channels * 4)
        self.res3 = ResidualBlock(base_channels * 4)
        self.up1 = Upsample(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up2 = Upsample(base_channels * 2, base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_image_input(x, self.in_channels, 'TinyResNet18UNet')
        s1 = self.res1(self.stem(x))
        s2 = self.res2(self.down1(s1))
        bottleneck = self.res3(self.down2(s2))
        x = self.up1(bottleneck, s2)
        x = self.up2(x, s1)
        return self.head(x)


def resnet18_unet_builder(task: str, in_channels: int = 1, base_channels: int = 16, out_dim: int = 1, **kwargs: Any) -> nn.Module:
    _ = kwargs
    if task.lower() not in {'segmentation', 'regression'}:
        raise ValueError(f"resnet18_unet supports task='segmentation' or 'regression', got {task!r}.")
    return TinyResNet18UNet(in_channels=in_channels, base_channels=base_channels, out_dim=out_dim)


__all__ = ['TinyResNet18UNet', 'resnet18_unet_builder']
