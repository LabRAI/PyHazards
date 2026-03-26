from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ._wildfire_layers import PatchMixer, TemporalAttentionFusion, check_sequence_input


class TinySegFormer(nn.Module):
    """Compact SegFormer-style wildfire sequence segmenter."""

    def __init__(self, history: int = 4, in_channels: int = 1, hidden_dim: int = 32, out_dim: int = 1):
        super().__init__()
        self.history = int(history)
        self.in_channels = int(in_channels)
        self.fusion = TemporalAttentionFusion(in_channels, hidden_dim)
        self.encoder = nn.Sequential(PatchMixer(hidden_dim, hidden_dim), PatchMixer(hidden_dim, hidden_dim))
        self.head = nn.Conv2d(hidden_dim, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_sequence_input(x, self.in_channels, 'TinySegFormer')
        if x.size(1) != self.history:
            raise ValueError(f"TinySegFormer expected history={self.history}, got {x.size(1)}.")
        return self.head(self.encoder(self.fusion(x)))


def segformer_builder(task: str, history: int = 4, in_channels: int = 1, hidden_dim: int = 32, out_dim: int = 1, **kwargs: Any) -> nn.Module:
    _ = kwargs
    if task.lower() not in {'segmentation', 'regression'}:
        raise ValueError(f"segformer supports task='segmentation' or 'regression', got {task!r}.")
    return TinySegFormer(history=history, in_channels=in_channels, hidden_dim=hidden_dim, out_dim=out_dim)


__all__ = ['TinySegFormer', 'segformer_builder']
