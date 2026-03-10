from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WRFSFireAdapter(nn.Module):
    """Lightweight raster adapter inspired by WRF-SFIRE style spread diffusion."""

    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 1,
        diffusion_steps: int = 3,
    ):
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels != 1:
            raise ValueError(f"WRFSFireAdapter only supports out_channels=1, got {out_channels}")
        if diffusion_steps <= 0:
            raise ValueError(f"diffusion_steps must be positive, got {diffusion_steps}")
        self.in_channels = int(in_channels)
        self.diffusion_steps = int(diffusion_steps)
        kernel = torch.tensor(
            [[0.02, 0.08, 0.02], [0.08, 0.60, 0.08], [0.02, 0.08, 0.02]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("transport_kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "WRFSFireAdapter expects input shape (batch, channels, height, width), "
                f"got {tuple(x.shape)}."
            )
        if x.size(1) != self.in_channels:
            raise ValueError(f"WRFSFireAdapter expected in_channels={self.in_channels}, got {x.size(1)}.")
        fireline = torch.sigmoid(x[:, :1])
        terrain = torch.sigmoid(x[:, 1:2])
        moisture = torch.sigmoid(x[:, 2:3])
        for _ in range(self.diffusion_steps):
            fireline = F.conv2d(fireline, self.transport_kernel, padding=1)
            fireline = torch.clamp(fireline * (0.9 + 0.1 * terrain) * (1.0 - 0.15 * moisture), 0.0, 1.0)
        return fireline


def wrf_sfire_builder(
    task: str,
    in_channels: int = 12,
    out_channels: int = 1,
    diffusion_steps: int = 3,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() not in {"segmentation", "regression"}:
        raise ValueError(f"wrf_sfire supports task='segmentation' or 'regression', got {task!r}.")
    return WRFSFireAdapter(
        in_channels=in_channels,
        out_channels=out_channels,
        diffusion_steps=diffusion_steps,
    )


__all__ = ["WRFSFireAdapter", "wrf_sfire_builder"]
