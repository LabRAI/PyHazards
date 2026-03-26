from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .unet import TinyUNet
from ._wildfire_layers import check_image_input


class DeepEnsemble(nn.Module):
    """Mean-ensemble wrapper over multiple compact wildfire segmentation members."""

    def __init__(self, in_channels: int = 1, base_channels: int = 16, out_dim: int = 1, ensemble_size: int = 3):
        super().__init__()
        self.in_channels = int(in_channels)
        self.members = nn.ModuleList([
            TinyUNet(in_channels=in_channels, base_channels=base_channels, out_dim=out_dim)
            for _ in range(int(ensemble_size))
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_image_input(x, self.in_channels, 'DeepEnsemble')
        outputs = [member(x) for member in self.members]
        return torch.stack(outputs, dim=0).mean(dim=0)


def deep_ensemble_builder(task: str, in_channels: int = 1, base_channels: int = 16, out_dim: int = 1, ensemble_size: int = 3, **kwargs: Any) -> nn.Module:
    _ = kwargs
    if task.lower() not in {'segmentation', 'regression'}:
        raise ValueError(f"deep_ensemble supports task='segmentation' or 'regression', got {task!r}.")
    return DeepEnsemble(in_channels=in_channels, base_channels=base_channels, out_dim=out_dim, ensemble_size=ensemble_size)


__all__ = ['DeepEnsemble', 'deep_ensemble_builder']
