from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ._wildfire_layers import check_sequence_input


class TinyTCN(nn.Module):
    """Compact temporal-convolution wildfire spread baseline."""

    def __init__(self, history: int = 4, in_channels: int = 1, hidden_dim: int = 24, out_dim: int = 1):
        super().__init__()
        self.history = int(history)
        self.in_channels = int(in_channels)
        self.temporal = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=(3, 3, 3), padding=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(3, 3, 3), padding=(2, 1, 1), dilation=(2, 1, 1)),
            nn.GELU(),
        )
        self.head = nn.Conv2d(hidden_dim, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_sequence_input(x, self.in_channels, 'TinyTCN')
        if x.size(1) != self.history:
            raise ValueError(f"TinyTCN expected history={self.history}, got {x.size(1)}.")
        encoded = self.temporal(x.permute(0, 2, 1, 3, 4))
        return self.head(torch.mean(encoded, dim=2))


def tcn_builder(task: str, history: int = 4, in_channels: int = 1, hidden_dim: int = 24, out_dim: int = 1, **kwargs: Any) -> nn.Module:
    _ = kwargs
    if task.lower() not in {'segmentation', 'regression'}:
        raise ValueError(f"tcn supports task='segmentation' or 'regression', got {task!r}.")
    return TinyTCN(history=history, in_channels=in_channels, hidden_dim=hidden_dim, out_dim=out_dim)


__all__ = ['TinyTCN', 'tcn_builder']
