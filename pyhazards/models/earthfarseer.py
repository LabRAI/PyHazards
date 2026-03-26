from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ._wildfire_layers import check_sequence_input


class TinyEarthFarseer(nn.Module):
    """Compact multi-scale temporal baseline inspired by EarthFarseer."""

    def __init__(self, history: int = 4, in_channels: int = 1, hidden_dim: int = 24, out_dim: int = 1):
        super().__init__()
        self.history = int(history)
        self.in_channels = int(in_channels)
        self.branch1 = nn.Conv3d(in_channels, hidden_dim, kernel_size=(3, 3, 3), padding=1)
        self.branch2 = nn.Conv3d(in_channels, hidden_dim, kernel_size=(3, 3, 3), padding=(2, 1, 1), dilation=(2, 1, 1))
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_sequence_input(x, self.in_channels, 'TinyEarthFarseer')
        if x.size(1) != self.history:
            raise ValueError(f"TinyEarthFarseer expected history={self.history}, got {x.size(1)}.")
        x3d = x.permute(0, 2, 1, 3, 4)
        f1 = torch.mean(torch.gelu(self.branch1(x3d)), dim=2)
        f2 = torch.mean(torch.gelu(self.branch2(x3d)), dim=2)
        return self.project(torch.cat([f1, f2], dim=1))


def earthfarseer_builder(task: str, history: int = 4, in_channels: int = 1, hidden_dim: int = 24, out_dim: int = 1, **kwargs: Any) -> nn.Module:
    _ = kwargs
    if task.lower() not in {'segmentation', 'regression'}:
        raise ValueError(f"earthfarseer supports task='segmentation' or 'regression', got {task!r}.")
    return TinyEarthFarseer(history=history, in_channels=in_channels, hidden_dim=hidden_dim, out_dim=out_dim)


__all__ = ['TinyEarthFarseer', 'earthfarseer_builder']
