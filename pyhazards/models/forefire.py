from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForeFireAdapter(nn.Module):
    """Lightweight deterministic spread adapter inspired by simulator-style fronts."""

    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 1,
        diffusion_steps: int = 2,
    ):
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels != 1:
            raise ValueError(f"ForeFireAdapter only supports out_channels=1, got {out_channels}")
        if diffusion_steps <= 0:
            raise ValueError(f"diffusion_steps must be positive, got {diffusion_steps}")
        self.in_channels = int(in_channels)
        self.diffusion_steps = int(diffusion_steps)
        kernel = torch.tensor(
            [[0.05, 0.15, 0.05], [0.15, 0.20, 0.15], [0.05, 0.15, 0.05]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("spread_kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "ForeFireAdapter expects input shape (batch, channels, height, width), "
                f"got {tuple(x.shape)}."
            )
        if x.size(1) != self.in_channels:
            raise ValueError(f"ForeFireAdapter expected in_channels={self.in_channels}, got {x.size(1)}.")
        state = torch.sigmoid(x[:, :1])
        fuel = torch.sigmoid(x[:, 1:2])
        wind = torch.tanh(x[:, 2:3]).abs()
        for _ in range(self.diffusion_steps):
            neighborhood = F.conv2d(state, self.spread_kernel, padding=1)
            state = torch.clamp(0.45 * state + 0.4 * neighborhood + 0.1 * fuel + 0.05 * wind, 0.0, 1.0)
        return state


def forefire_builder(
    task: str,
    in_channels: int = 12,
    out_channels: int = 1,
    diffusion_steps: int = 2,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() not in {"segmentation", "regression"}:
        raise ValueError(f"forefire supports task='segmentation' or 'regression', got {task!r}.")
    return ForeFireAdapter(
        in_channels=in_channels,
        out_channels=out_channels,
        diffusion_steps=diffusion_steps,
    )


__all__ = ["ForeFireAdapter", "forefire_builder"]
