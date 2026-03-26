from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .convlstm import ConvLSTMCell
from ._wildfire_layers import check_sequence_input


class TinySwinLSTM(nn.Module):
    """Compact windowed recurrent baseline inspired by SwinLSTM."""

    def __init__(self, history: int = 4, in_channels: int = 1, hidden_dim: int = 24, out_dim: int = 1, window_size: int = 4):
        super().__init__()
        self.history = int(history)
        self.in_channels = int(in_channels)
        self.window_size = int(window_size)
        self.cell = ConvLSTMCell(in_channels, hidden_dim)
        self.head = nn.Conv2d(hidden_dim, out_dim, kernel_size=1)

    def _window_smooth(self, x: torch.Tensor) -> torch.Tensor:
        if self.window_size <= 1:
            return x
        pooled = F.avg_pool2d(x, kernel_size=self.window_size, stride=1, padding=self.window_size // 2)
        return 0.5 * x + 0.5 * pooled[..., : x.size(-2), : x.size(-1)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_sequence_input(x, self.in_channels, 'TinySwinLSTM')
        if x.size(1) != self.history:
            raise ValueError(f"TinySwinLSTM expected history={self.history}, got {x.size(1)}.")
        b, _, _, h, w = x.shape
        h_t = x.new_zeros((b, self.cell.hidden_dim, h, w))
        c_t = x.new_zeros((b, self.cell.hidden_dim, h, w))
        for t in range(self.history):
            h_t, c_t = self.cell(self._window_smooth(x[:, t]), (h_t, c_t))
        return self.head(h_t)


def swinlstm_builder(task: str, history: int = 4, in_channels: int = 1, hidden_dim: int = 24, out_dim: int = 1, window_size: int = 4, **kwargs: Any) -> nn.Module:
    _ = kwargs
    if task.lower() not in {'segmentation', 'regression'}:
        raise ValueError(f"swinlstm supports task='segmentation' or 'regression', got {task!r}.")
    return TinySwinLSTM(history=history, in_channels=in_channels, hidden_dim=hidden_dim, out_dim=out_dim, window_size=window_size)


__all__ = ['TinySwinLSTM', 'swinlstm_builder']
