from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ._wildfire_layers import check_sequence_input


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gates = nn.Conv2d(in_channels + hidden_dim, hidden_dim * 4, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        gates = self.gates(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class TinyConvLSTM(nn.Module):
    """Compact ConvLSTM baseline for wildfire spread prediction from raster histories."""

    def __init__(self, history: int = 4, in_channels: int = 1, hidden_dim: int = 24, out_dim: int = 1):
        super().__init__()
        self.history = int(history)
        self.in_channels = int(in_channels)
        self.cell = ConvLSTMCell(in_channels, hidden_dim)
        self.head = nn.Conv2d(hidden_dim, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_sequence_input(x, self.in_channels, 'TinyConvLSTM')
        if x.size(1) != self.history:
            raise ValueError(f"TinyConvLSTM expected history={self.history}, got {x.size(1)}.")
        b, _, _, h, w = x.shape
        h_t = x.new_zeros((b, self.cell.hidden_dim, h, w))
        c_t = x.new_zeros((b, self.cell.hidden_dim, h, w))
        for t in range(self.history):
            h_t, c_t = self.cell(x[:, t], (h_t, c_t))
        return self.head(h_t)


def convlstm_builder(task: str, history: int = 4, in_channels: int = 1, hidden_dim: int = 24, out_dim: int = 1, **kwargs: Any) -> nn.Module:
    _ = kwargs
    if task.lower() not in {'segmentation', 'regression'}:
        raise ValueError(f"convlstm supports task='segmentation' or 'regression', got {task!r}.")
    return TinyConvLSTM(history=history, in_channels=in_channels, hidden_dim=hidden_dim, out_dim=out_dim)


__all__ = ['TinyConvLSTM', 'convlstm_builder']
