from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ._wildfire_layers import PatchMixer, TemporalAttentionFusion, check_sequence_input


class TinyEarthFormer(nn.Module):
    """Compact EarthFormer-style baseline for wildfire spread sequence prediction."""

    def __init__(self, history: int = 4, in_channels: int = 1, hidden_dim: int = 32, out_dim: int = 1):
        super().__init__()
        self.history = int(history)
        self.in_channels = int(in_channels)
        self.fusion = TemporalAttentionFusion(in_channels, hidden_dim)
        self.encoder = PatchMixer(hidden_dim, hidden_dim)
        self.head = nn.Conv2d(hidden_dim, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_sequence_input(x, self.in_channels, 'TinyEarthFormer')
        if x.size(1) != self.history:
            raise ValueError(f"TinyEarthFormer expected history={self.history}, got {x.size(1)}.")
        fused = self.fusion(x)
        return self.head(self.encoder(fused))


def earthformer_builder(task: str, history: int = 4, in_channels: int = 1, hidden_dim: int = 32, out_dim: int = 1, **kwargs: Any) -> nn.Module:
    _ = kwargs
    if task.lower() not in {'segmentation', 'regression'}:
        raise ValueError(f"earthformer supports task='segmentation' or 'regression', got {task!r}.")
    return TinyEarthFormer(history=history, in_channels=in_channels, hidden_dim=hidden_dim, out_dim=out_dim)


__all__ = ['TinyEarthFormer', 'earthformer_builder']
