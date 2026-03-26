from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ._wildfire_layers import check_sequence_input


class PredCell(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.hidden_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.memory_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, h: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_proj = torch.tanh(self.input_proj(x))
        h = torch.tanh(self.hidden_proj(h) + x_proj + self.memory_proj(m))
        m = 0.7 * m + 0.3 * h
        return h, m


class TinyPredRNNv2(nn.Module):
    """Compact predictive recurrent baseline inspired by PredRNN-v2."""

    def __init__(self, history: int = 4, in_channels: int = 1, hidden_dim: int = 24, out_dim: int = 1):
        super().__init__()
        self.history = int(history)
        self.in_channels = int(in_channels)
        self.cell = PredCell(in_channels, hidden_dim)
        self.head = nn.Conv2d(hidden_dim, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_sequence_input(x, self.in_channels, 'TinyPredRNNv2')
        if x.size(1) != self.history:
            raise ValueError(f"TinyPredRNNv2 expected history={self.history}, got {x.size(1)}.")
        b, _, _, h, w = x.shape
        h_t = x.new_zeros((b, self.cell.hidden_dim, h, w))
        m_t = x.new_zeros((b, self.cell.hidden_dim, h, w))
        for t in range(self.history):
            h_t, m_t = self.cell(x[:, t], h_t, m_t)
        return self.head(h_t)


def predrnn_v2_builder(task: str, history: int = 4, in_channels: int = 1, hidden_dim: int = 24, out_dim: int = 1, **kwargs: Any) -> nn.Module:
    _ = kwargs
    if task.lower() not in {'segmentation', 'regression'}:
        raise ValueError(f"predrnn_v2 supports task='segmentation' or 'regression', got {task!r}.")
    return TinyPredRNNv2(history=history, in_channels=in_channels, hidden_dim=hidden_dim, out_dim=out_dim)


__all__ = ['TinyPredRNNv2', 'predrnn_v2_builder']
