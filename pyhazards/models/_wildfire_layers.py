from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def check_image_input(x: torch.Tensor, in_channels: int, name: str) -> None:
    if x.ndim != 4:
        raise ValueError(f"{name} expects input shape (batch, channels, height, width), got {tuple(x.shape)}.")
    if x.size(1) != in_channels:
        raise ValueError(f"{name} expected in_channels={in_channels}, got {x.size(1)}.")


def check_sequence_input(x: torch.Tensor, in_channels: int, name: str) -> None:
    if x.ndim != 5:
        raise ValueError(
            f"{name} expects input shape (batch, history, channels, height, width), got {tuple(x.shape)}."
        )
    if x.size(2) != in_channels:
        raise ValueError(f"{name} expected in_channels={in_channels}, got {x.size(2)}.")


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.block(x))


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class ASPPBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilations: tuple[int, ...] = (1, 3, 6, 12)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=d, dilation=d)
            for d in dilations
        ])
        self.project = nn.Conv2d(len(dilations) * out_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [F.gelu(branch(x)) for branch in self.branches]
        return F.gelu(self.project(torch.cat(branches, dim=1)))


class TemporalAttentionFusion(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=(3, 3, 3), padding=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(3, 3, 3), padding=1),
            nn.GELU(),
        )
        self.score = nn.Conv3d(hidden_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x.permute(0, 2, 1, 3, 4))
        weights = torch.softmax(self.score(encoded), dim=2)
        return torch.sum(encoded * weights, dim=2)


class PatchMixer(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.mix = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=max(1, hidden_dim // 8)),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.proj(x))
        return self.mix(x)
