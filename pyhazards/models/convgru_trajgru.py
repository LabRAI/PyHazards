from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ._wildfire_layers import check_sequence_input


class ConvGRUCell(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.update = nn.Conv2d(in_channels + hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.reset = nn.Conv2d(in_channels + hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.out = nn.Conv2d(in_channels + hidden_dim, hidden_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        z = torch.sigmoid(self.update(torch.cat([x, h], dim=1)))
        r = torch.sigmoid(self.reset(torch.cat([x, h], dim=1)))
        candidate = torch.tanh(self.out(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * h + z * candidate


class TinyConvGRTrajGRU(nn.Module):
    """Compact ConvGRU-style baseline used for trajectory-aware wildfire sequence prediction."""

    def __init__(self, history: int = 4, in_channels: int = 1, hidden_dim: int = 24, out_dim: int = 1):
        super().__init__()
        self.history = int(history)
        self.in_channels = int(in_channels)
        self.cell = ConvGRUCell(in_channels, hidden_dim)
        self.head = nn.Conv2d(hidden_dim, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_sequence_input(x, self.in_channels, 'TinyConvGRTrajGRU')
        if x.size(1) != self.history:
            raise ValueError(f"TinyConvGRTrajGRU expected history={self.history}, got {x.size(1)}.")
        b, _, _, h, w = x.shape
        h_t = x.new_zeros((b, self.cell.hidden_dim, h, w))
        for t in range(self.history):
            h_t = self.cell(x[:, t], h_t)
        return self.head(h_t)


def convgru_trajgru_builder(task: str, history: int = 4, in_channels: int = 1, hidden_dim: int = 24, out_dim: int = 1, **kwargs: Any) -> nn.Module:
    _ = kwargs
    if task.lower() not in {'segmentation', 'regression'}:
        raise ValueError(f"convgru_trajgru supports task='segmentation' or 'regression', got {task!r}.")
    return TinyConvGRTrajGRU(history=history, in_channels=in_channels, hidden_dim=hidden_dim, out_dim=out_dim)


__all__ = ['TinyConvGRTrajGRU', 'convgru_trajgru_builder']
