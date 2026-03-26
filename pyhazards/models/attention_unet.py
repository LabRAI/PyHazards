from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ._wildfire_layers import ConvBlock, Downsample, Upsample, check_image_input


class AttentionGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class TinyAttentionUNet(nn.Module):
    """Compact Attention U-Net baseline for wildfire spread masks."""

    def __init__(self, in_channels: int = 1, base_channels: int = 16, out_dim: int = 1):
        super().__init__()
        self.in_channels = int(in_channels)
        self.stem = ConvBlock(in_channels, base_channels)
        self.down1 = Downsample(base_channels, base_channels * 2)
        self.down2 = Downsample(base_channels * 2, base_channels * 4)
        self.attn2 = AttentionGate(base_channels * 2)
        self.attn1 = AttentionGate(base_channels)
        self.up1 = Upsample(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up2 = Upsample(base_channels * 2, base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_image_input(x, self.in_channels, 'TinyAttentionUNet')
        s1 = self.stem(x)
        s2 = self.down1(s1)
        bottleneck = self.down2(s2)
        x = self.up1(bottleneck, self.attn2(s2))
        x = self.up2(x, self.attn1(s1))
        return self.head(x)


def attention_unet_builder(task: str, in_channels: int = 1, base_channels: int = 16, out_dim: int = 1, **kwargs: Any) -> nn.Module:
    _ = kwargs
    if task.lower() not in {'segmentation', 'regression'}:
        raise ValueError(f"attention_unet supports task='segmentation' or 'regression', got {task!r}.")
    return TinyAttentionUNet(in_channels=in_channels, base_channels=base_channels, out_dim=out_dim)


__all__ = ['TinyAttentionUNet', 'attention_unet_builder']
