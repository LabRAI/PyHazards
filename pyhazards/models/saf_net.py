"""SAF-Net: spatio-temporal attention fusion network for 24-hour typhoon intensity prediction.

Paper: G. Xu, K. Lin, X. Li and Y. Ye, "SAF-Net: A spatio-temporal deep learning method for
typhoon intensity prediction", Pattern Recognition Letters 155, 121-127 (2022),
doi:10.1016/j.patrec.2021.11.012.

The official code (github.com/xuguangning1218/TI_Prediction, ``SAF-Net.ipynb``) has no license,
so this module is written from the paper's description of the network; the notebook and its
released checkpoint (``model_saver/SAF_Net.pkl``) are used only as a test oracle
(tests/oracle/test_saf_net_oracle.py) and are never copied or redistributed. Module and parameter
names follow the official state dict so that the released checkpoint loads with ``strict=True``,
and parameters are created in the same order, so the same seed gives the same initial weights.

SAF-Net is a wide-and-deep regressor of the maximum sustained wind 24 hours ahead:

* wide part: 96 hand-crafted CLIPER-style predictors from the CMA best track (current and
  past positions, pressures and winds, their 6-hourly changes and derived climatology factors),
  MinMax-scaled;
* deep part: ERA-Interim u and v wind on a 31 x 31 storm-centred grid at four pressure levels
  (1000, 750, 500, 250 hPa) for four times (t, t-6 h, t-12 h, t-18 h), MinMax-scaled, as a tensor
  ``(batch, 2, level, height, width, time)`` with u first;
* each time step runs three convolution stages (3 x 3 convolution with 64 / 128 / 256 channels,
  ReLU, 2 x 2 max pooling) shared by a u branch, a v branch and a fused branch. After every stage a
  spatial-attention block (1 x 1 conv, BatchNorm, ReLU, 1 x 1 conv, BatchNorm, sigmoid) rescales
  the features. Before stages 2 and 3, and after stage 3, the u and v branches are mixed with
  learnable weights (``cross_unit``) and merged with the fused branch with learnable weights
  (``fuse_unit``); each pair of weights ``(a, b)`` mixes as ``a / (a + b) * x + b / (a + b) * y``;
* the fused 256 x 3 x 3 map of each time step goes through ``fc1`` (no activation); the four
  128-d time features are concatenated after the 96 wide predictors, then ``fc2`` (ReLU) and
  ``fc3`` (ReLU) give one MinMax-scaled intensity.

Behaviour of the official network that is kept because it changes outputs: the fused branch of
stage 3 passes through the stage-3 attention block a second time after the final mix; ``fc1`` has
no activation; the output goes through a ReLU, so predictions of the scaled target are never
negative.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_U, _V = 0, 1


def _spatial_attention(channels: int) -> nn.Sequential:
    """1x1 conv - BN - ReLU - 1x1 conv - BN - sigmoid, giving a per-pixel, per-channel gate."""
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=1, padding=0),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels, channels, kernel_size=1, padding=0),
        nn.BatchNorm2d(channels),
        nn.Sigmoid(),
    )


def _weighted_mix(weights: torch.Tensor, first: int, second: int, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    total = weights[first] + weights[second]
    return weights[first] / total * x + weights[second] / total * y


class SAFNet(nn.Module):
    """SAF-Net intensity regressor.

    ``forward(wide, deep)`` takes ``wide`` ``(batch, wide_dim)`` and ``deep``
    ``(batch, 2, num_levels, grid_size, grid_size, num_times)`` (u then v; times ordered t, t-6 h,
    t-12 h, t-18 h in the paper) and returns ``(batch, 1)``: the MinMax-scaled maximum sustained
    wind ``lead`` hours after time t. A mapping ``{"wide": ..., "deep": ...}`` may be passed as
    the only argument.
    """

    def __init__(
        self,
        wide_dim: int = 96,
        num_times: int = 4,
        num_levels: int = 4,
        grid_size: int = 31,
        channels: Sequence[int] = (64, 128, 256),
        time_feature_dim: int = 128,
        hidden_dim: int = 64,
    ):
        super().__init__()
        channels = tuple(int(c) for c in channels)
        if len(channels) != 3:
            raise ValueError("SAF-Net has exactly three convolution stages; pass three channel widths")
        spatial = int(grid_size)
        for _ in channels:
            spatial //= 2
        if spatial < 1:
            raise ValueError(f"grid_size={grid_size} is too small for three 2x2 poolings")
        self.wide_dim = int(wide_dim)
        self.num_times = int(num_times)
        self.num_levels = int(num_levels)
        self.grid_size = int(grid_size)
        self.channels = channels
        self.time_feature_dim = int(time_feature_dim)

        # Creation order follows the official network (attention blocks, mixing weights,
        # convolutions, fully connected layers) so that a seed gives the official initial weights.
        self.att_block_1 = _spatial_attention(channels[0])
        self.att_block_2 = _spatial_attention(channels[1])
        self.att_block_3 = _spatial_attention(channels[2])
        # Per time step: three (u, v) cross weights and two (fused, cross) fuse weights.
        self.cross_unit = nn.Parameter(torch.ones(self.num_times, 6))
        self.fuse_unit = nn.Parameter(torch.ones(self.num_times, 4))
        self.conv1 = nn.Conv2d(self.num_levels, channels[0], kernel_size=3, padding=(1, 1))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=(1, 1))
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=(1, 1))
        self.fc1 = nn.Linear(channels[2] * spatial * spatial, self.time_feature_dim)
        self.fc2 = nn.Linear(self.wide_dim + self.time_feature_dim * self.num_times, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def _stage(self, index: int, x: torch.Tensor) -> torch.Tensor:
        conv = getattr(self, f"conv{index}")
        attention = getattr(self, f"att_block_{index}")
        features = self.pool(F.relu(conv(x)))
        return attention(features) * features

    def _time_step_features(self, u: torch.Tensor, v: torch.Tensor, step: int) -> torch.Tensor:
        cross = self.cross_unit[step]
        fuse = self.fuse_unit[step]
        u = self._stage(1, u)
        v = self._stage(1, v)
        fused = self._stage(2, _weighted_mix(cross, 0, 1, u, v))
        u = self._stage(2, u)
        v = self._stage(2, v)
        fused = self._stage(3, _weighted_mix(fuse, 0, 1, fused, _weighted_mix(cross, 2, 3, u, v)))
        u = self._stage(3, u)
        v = self._stage(3, v)
        fused = _weighted_mix(fuse, 2, 3, fused, _weighted_mix(cross, 4, 5, u, v))
        fused = self.att_block_3(fused) * fused
        return self.fc1(fused.flatten(1))

    def _check_inputs(self, wide: torch.Tensor, deep: torch.Tensor) -> None:
        expected_deep = (2, self.num_levels, self.grid_size, self.grid_size, self.num_times)
        if deep.ndim != 6 or tuple(deep.shape[1:]) != expected_deep:
            raise ValueError(
                "SAF-Net expects deep inputs shaped (batch, 2 [u, v], {levels}, {size}, {size}, {times}), got {shape}".format(
                    levels=self.num_levels, size=self.grid_size, times=self.num_times, shape=tuple(deep.shape)
                )
            )
        if wide.ndim != 2 or wide.size(1) != self.wide_dim or wide.size(0) != deep.size(0):
            raise ValueError(
                f"SAF-Net expects wide inputs shaped (batch, {self.wide_dim}) matching the deep batch, got {tuple(wide.shape)}"
            )

    def forward(self, wide, deep: Optional[torch.Tensor] = None) -> torch.Tensor:
        if isinstance(wide, Mapping):
            wide, deep = wide["wide"], wide["deep"]
        if deep is None:
            raise ValueError("SAF-Net needs both the wide predictors and the deep wind fields")
        self._check_inputs(wide, deep)
        time_features = [
            self._time_step_features(deep[:, _U, :, :, :, step], deep[:, _V, :, :, :, step], step)
            for step in range(self.num_times)
        ]
        joined = torch.cat([wide, *time_features], dim=1)
        return F.relu(self.fc3(F.relu(self.fc2(joined))))


def saf_net_builder(
    task: str,
    wide_dim: int = 96,
    num_times: int = 4,
    num_levels: int = 4,
    grid_size: int = 31,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() != "regression":
        raise ValueError("saf_net is a regression model (24-hour intensity); use task='regression'.")
    return SAFNet(wide_dim=wide_dim, num_times=num_times, num_levels=num_levels, grid_size=grid_size)


__all__ = ["SAFNet", "saf_net_builder"]
