from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Basic blocks
# ---------------------------------------------------------------------

class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        d: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ---------------------------------------------------------------------
# ASPP
# ---------------------------------------------------------------------

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP).

    Parallel atrous convolutions + image pooling branch,
    followed by projection.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dilations: Sequence[int] = (1, 3, 6, 12),
    ):
        super().__init__()

        if len(dilations) != 4:
            raise ValueError("ASPP expects exactly 4 dilation rates")

        d1, d2, d3, d4 = dilations

        self.b1 = ConvBNReLU(in_ch, out_ch, k=1, p=0, d=d1)
        self.b2 = ConvBNReLU(in_ch, out_ch, k=3, p=d2, d=d2)
        self.b3 = ConvBNReLU(in_ch, out_ch, k=3, p=d3, d=d3)
        self.b4 = ConvBNReLU(in_ch, out_ch, k=3, p=d4, d=d4)

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNReLU(in_ch, out_ch, k=1, p=0),
        )

        self.proj = ConvBNReLU(out_ch * 5, out_ch, k=1, p=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        p = self.pool(x)
        p = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)

        y = torch.cat(
            [self.b1(x), self.b2(x), self.b3(x), self.b4(x), p],
            dim=1,
        )
        return self.proj(y)


# ---------------------------------------------------------------------
# CNN + ASPP model
# ---------------------------------------------------------------------

class WildfireCNNASPP(nn.Module):
    """
    CNN + ASPP wildfire segmentation model.

    Input:
        x : (B, C, H, W) float tensor

    Output:
        logits : (B, 1, H, W) float tensor
        (sigmoid applied externally)
    """

    def __init__(
        self,
        in_channels: int = 12,
        base_channels: int = 32,
        aspp_channels: int = 32,
        dilations: Sequence[int] = (1, 3, 6, 12),
        dropout: float = 0.0,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, base_channels, k=3, p=1),
            ConvBNReLU(base_channels, base_channels, k=3, p=1),
        )

        self.aspp = ASPP(
            in_ch=base_channels,
            out_ch=aspp_channels,
            dilations=dilations,
        )

        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Conv2d(aspp_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected input of shape (B,C,H,W), got {tuple(x.shape)}"
            )

        f = self.stem(x)
        y = self.aspp(f)
        y = self.drop(y)
        return self.head(y)


# ---------------------------------------------------------------------
# PyHazards model builder
# ---------------------------------------------------------------------

def cnn_aspp_builder(
    task: str,
    in_channels: int = 12,
    base_channels: int = 32,
    aspp_channels: int = 32,
    dilations: Sequence[int] = (1, 3, 6, 12),
    dropout: float = 0.0,
    **kwargs,
) -> nn.Module:
    """
    PyHazards-style model builder.
    """
    _ = kwargs  # explicitly ignore unused builder args

    if "segmentation" not in task:
        raise ValueError(
            f"WildfireCNNASPP is segmentation-only. Got task='{task}'"
        )

    return WildfireCNNASPP(
        in_channels=in_channels,
        base_channels=base_channels,
        aspp_channels=aspp_channels,
        dilations=dilations,
        dropout=dropout,
    )
