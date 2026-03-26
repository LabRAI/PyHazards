from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ._wildfire_layers import PatchMixer, check_sequence_input


class TinyUTAE(nn.Module):
    """Compact temporal-attention encoder for wildfire spread prediction."""

    def __init__(self, history: int = 4, in_channels: int = 1, hidden_dim: int = 24, out_dim: int = 1):
        super().__init__()
        self.history = int(history)
        self.in_channels = int(in_channels)
        self.frame_encoder = PatchMixer(in_channels, hidden_dim)
        self.score = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden_dim, 1, kernel_size=1))
        self.head = nn.Sequential(PatchMixer(hidden_dim, hidden_dim), nn.Conv2d(hidden_dim, out_dim, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_sequence_input(x, self.in_channels, 'TinyUTAE')
        if x.size(1) != self.history:
            raise ValueError(f"TinyUTAE expected history={self.history}, got {x.size(1)}.")
        frames = [self.frame_encoder(x[:, t]) for t in range(self.history)]
        scores = torch.stack([self.score(frame).flatten(1) for frame in frames], dim=1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1).unsqueeze(-1)
        fused = torch.sum(torch.stack(frames, dim=1) * weights, dim=1)
        return self.head(fused)


def utae_builder(task: str, history: int = 4, in_channels: int = 1, hidden_dim: int = 24, out_dim: int = 1, **kwargs: Any) -> nn.Module:
    _ = kwargs
    if task.lower() not in {'segmentation', 'regression'}:
        raise ValueError(f"utae supports task='segmentation' or 'regression', got {task!r}.")
    return TinyUTAE(history=history, in_channels=in_channels, hidden_dim=hidden_dim, out_dim=out_dim)


__all__ = ['TinyUTAE', 'utae_builder']
