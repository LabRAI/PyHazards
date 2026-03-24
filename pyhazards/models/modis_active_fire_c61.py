from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MODISActiveFireC61(nn.Module):
    """Algorithm-inspired MODIS Collection 6.1 active-fire detector with learnable calibration."""

    def __init__(
        self,
        in_channels: int = 5,
        hidden_dim: int = 24,
        out_dim: int = 1,
        context_kernel: int = 9,
        dropout: float = 0.1,
    ):
        super().__init__()
        if in_channels < 5:
            raise ValueError(
                "MODISActiveFireC61 expects at least 5 channels: "
                "mid_ir, long_ir, frp_proxy, cloud_free, dryness."
            )
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")
        if context_kernel <= 1 or context_kernel % 2 == 0:
            raise ValueError(f"context_kernel must be an odd integer > 1, got {context_kernel}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0,1), got {dropout}")

        self.in_channels = int(in_channels)
        self.context_pool = nn.AvgPool2d(kernel_size=context_kernel, stride=1, padding=context_kernel // 2)

        evidence_channels = self.in_channels + 5
        self.evidence_encoder = nn.Sequential(
            nn.Conv2d(evidence_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.calibration_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "MODISActiveFireC61 expects input shape (batch, channels, height, width), "
                f"got {tuple(x.shape)}."
            )
        if x.size(1) < 5:
            raise ValueError(f"MODISActiveFireC61 expected at least 5 channels, got {x.size(1)}.")

        x = x[:, : self.in_channels]
        mid_ir = x[:, 0:1]
        long_ir = x[:, 1:2]
        frp_proxy = x[:, 2:3]
        cloud_free = x[:, 3:4]
        dryness = x[:, 4:5]

        local_background = self.context_pool(mid_ir)
        thermal_excess = mid_ir - local_background
        split_window = mid_ir - long_ir
        fire_signal = F.relu(thermal_excess) + 0.4 * F.relu(split_window)
        contextual_ratio = fire_signal / (torch.abs(local_background) + 1.0)
        confidence_gate = torch.sigmoid(cloud_free) * torch.sigmoid(dryness)

        evidence = torch.cat(
            [
                x,
                thermal_excess,
                split_window,
                fire_signal,
                contextual_ratio,
                confidence_gate + frp_proxy,
            ],
            dim=1,
        )
        encoded = self.evidence_encoder(evidence)
        return self.calibration_head(encoded)


def modis_active_fire_c61_builder(
    task: str,
    in_channels: int = 5,
    hidden_dim: int = 24,
    out_dim: int = 1,
    context_kernel: int = 9,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() != "segmentation":
        raise ValueError(
            f"modis_active_fire_c61 is segmentation-only in PyHazards, got task={task!r}."
        )
    return MODISActiveFireC61(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        context_kernel=context_kernel,
        dropout=dropout,
    )


__all__ = ["MODISActiveFireC61", "modis_active_fire_c61_builder"]
