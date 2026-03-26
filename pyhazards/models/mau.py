from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ._wildfire_layers import TemporalAttentionFusion, check_sequence_input


class TinyMAU(nn.Module):
    """Compact multi-axis temporal fusion baseline for wildfire spread masks."""

    def __init__(self, history: int = 4, in_channels: int = 1, hidden_dim: int = 24, out_dim: int = 1):
        super().__init__()
        self.history = int(history)
        self.in_channels = int(in_channels)
        self.fusion = TemporalAttentionFusion(in_channels, hidden_dim)
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_sequence_input(x, self.in_channels, 'TinyMAU')
        if x.size(1) != self.history:
            raise ValueError(f"TinyMAU expected history={self.history}, got {x.size(1)}.")
        return self.head(self.fusion(x))


def mau_builder(task: str, history: int = 4, in_channels: int = 1, hidden_dim: int = 24, out_dim: int = 1, **kwargs: Any) -> nn.Module:
    _ = kwargs
    if task.lower() not in {'segmentation', 'regression'}:
        raise ValueError(f"mau supports task='segmentation' or 'regression', got {task!r}.")
    return TinyMAU(history=history, in_channels=in_channels, hidden_dim=hidden_dim, out_dim=out_dim)


__all__ = ['TinyMAU', 'mau_builder']
