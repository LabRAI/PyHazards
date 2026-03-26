from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ._wildfire_layers import PatchMixer, TemporalAttentionFusion, check_sequence_input


class TinyRainformer(nn.Module):
    """Compact temporal-attention baseline inspired by Rainformer-style sequence fusion."""

    def __init__(self, history: int = 4, in_channels: int = 1, hidden_dim: int = 24, out_dim: int = 1):
        super().__init__()
        self.history = int(history)
        self.in_channels = int(in_channels)
        self.fusion = TemporalAttentionFusion(in_channels, hidden_dim)
        self.decoder = nn.Sequential(PatchMixer(hidden_dim, hidden_dim), nn.Conv2d(hidden_dim, out_dim, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_sequence_input(x, self.in_channels, 'TinyRainformer')
        if x.size(1) != self.history:
            raise ValueError(f"TinyRainformer expected history={self.history}, got {x.size(1)}.")
        return self.decoder(self.fusion(x))


def rainformer_builder(task: str, history: int = 4, in_channels: int = 1, hidden_dim: int = 24, out_dim: int = 1, **kwargs: Any) -> nn.Module:
    _ = kwargs
    if task.lower() not in {'segmentation', 'regression'}:
        raise ValueError(f"rainformer supports task='segmentation' or 'regression', got {task!r}.")
    return TinyRainformer(history=history, in_channels=in_channels, hidden_dim=hidden_dim, out_dim=out_dim)


__all__ = ['TinyRainformer', 'rainformer_builder']
