from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._wildfire_layers import ConvBlock, Downsample, TemporalAttentionFusion, Upsample, check_sequence_input


class TinySwinUNet(nn.Module):
    """Compact Swin-UNet style wildfire sequence segmenter."""

    def __init__(self, history: int = 4, in_channels: int = 1, base_channels: int = 16, out_dim: int = 1, window_size: int = 4):
        super().__init__()
        self.history = int(history)
        self.in_channels = int(in_channels)
        self.window_size = int(window_size)
        self.fusion = TemporalAttentionFusion(in_channels, base_channels)
        self.stem = ConvBlock(base_channels, base_channels)
        self.down = Downsample(base_channels, base_channels * 2)
        self.up = Upsample(base_channels * 2, base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, out_dim, kernel_size=1)

    def _window_mix(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.avg_pool2d(x, kernel_size=self.window_size, stride=1, padding=self.window_size // 2)
        pooled = pooled[..., : x.size(-2), : x.size(-1)]
        return 0.5 * x + 0.5 * pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_sequence_input(x, self.in_channels, 'TinySwinUNet')
        if x.size(1) != self.history:
            raise ValueError(f"TinySwinUNet expected history={self.history}, got {x.size(1)}.")
        fused = self._window_mix(self.fusion(x))
        s1 = self.stem(fused)
        s2 = self.down(s1)
        x = self.up(s2, s1)
        return self.head(x)


def swin_unet_builder(task: str, history: int = 4, in_channels: int = 1, base_channels: int = 16, out_dim: int = 1, window_size: int = 4, **kwargs: Any) -> nn.Module:
    _ = kwargs
    if task.lower() not in {'segmentation', 'regression'}:
        raise ValueError(f"swin_unet supports task='segmentation' or 'regression', got {task!r}.")
    return TinySwinUNet(history=history, in_channels=in_channels, base_channels=base_channels, out_dim=out_dim, window_size=window_size)


__all__ = ['TinySwinUNet', 'swin_unet_builder']
