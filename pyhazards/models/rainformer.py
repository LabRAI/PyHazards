"""Rainformer: local/global feature-balanced U-shaped network for radar nowcasting.

Architecture paper: Bai, Sun, Zhang, Song and Chen, "Rainformer: Features Extraction Balanced
Network for Radar-Based Precipitation Nowcasting", IEEE Geoscience and Remote Sensing Letters 19,
2022 (doi:10.1109/LGRS.2022.3162882). Wildfire usage: the Sim2Real-Fire benchmark (Li et al.,
NeurIPS 2024 Datasets and Benchmarks) reports it as its strongest fire-forecasting baseline.

The official code (github.com/Zjut-MultimediaPlus/Rainformer) has no license, so this module is
written from the paper and from permissively licensed parts; the official code is used only as a
test oracle (tests/oracle/test_rainformer_oracle.py). Module and parameter names
match the official ``Net`` so its state dicts (including the released KNMI checkpoint) load with
``strict=True``, and modules are created in the same order, so the same seed gives the same
initial weights.

Attribution:

- ``Residual``, ``PreNorm``, ``FeedForward``, the shifted-window masks, the relative-position
  table, ``WindowAttention`` and ``SwinBlock`` are ported from berniwal/swin-transformer-pytorch
  ``swin_transformer_pytorch/swin_transformer.py`` at commit
  ``c921ebf914c6ea9734bb260ada395e3746c85402`` (MIT License, Copyright (c) 2021 Bernhard Walser),
  the Swin implementation the official Rainformer builds on; the einops rearrangements are
  written as reshapes and permutes.
- The channel and spatial attention of the local branch follow CBAM (Woo et al., ECCV 2018), as
  the Rainformer paper states; the gate fusion unit, patch merging/expanding and the U-shaped
  stage layout are written from the Rainformer paper (Sec. II-B to II-D).

Behaviour of the official code that is kept (it changes outputs or parameter counts):

- every window attention is followed by a channel attention (``ca``) with separate average- and
  max-pool MLPs, and the relative position bias is one table shared by all heads;
- the gate fusion unit computes ``Z * g + R * l`` from ``Z, R = sigmoid(LN(conv(cat(g, l))))``;
  the paper's extra convolutions of ``g`` and ``l`` (``conv_2``, ``conv_3``) exist, are counted
  and initialised, but are never applied;
- the LayerNorms of the gate fusion unit normalise over ``(channels, height, width)``, so the
  parameter count depends on the input size; the official code hard-codes the stage sizes for
  288x288 inputs, PyHazards derives them from ``img_size``;
- decoder blocks use an MLP width of only twice their channel count (``4 * hidden``, with
  ``2 * hidden`` channels);
- inside a stage, every (window block, shifted block, CNN, CASA, gate) group reads the stage input
  and only the last group's output is kept. With the reference ``layers=(2, 2, 2, 2)`` each stage
  has one group; larger ``layers`` add groups whose outputs are discarded (as in the reference).
"""

from __future__ import annotations

from typing import Sequence, Union

import torch
import torch.nn as nn

from .swin_blocks import to_2tuple


# --------------------------------------------------------------------------- Swin parts (MIT)


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def create_mask(window_size: int, displacement: int, upper_lower: bool, left_right: bool) -> torch.Tensor:
    """``-inf`` between window positions that come from different sides of the cyclic shift."""
    mask = torch.zeros(window_size ** 2, window_size ** 2)
    cut = displacement * window_size
    if upper_lower:
        mask[-cut:, :-cut] = float("-inf")
        mask[:-cut, -cut:] = float("-inf")
    if left_right:
        mask = mask.view(window_size, window_size, window_size, window_size)
        mask[:, -displacement:, :, :-displacement] = float("-inf")
        mask[:, :-displacement, :, -displacement:] = float("-inf")
        mask = mask.view(window_size ** 2, window_size ** 2)
    return mask


def get_relative_distances(window_size: int) -> torch.Tensor:
    coords = torch.stack(
        torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij"), dim=-1
    ).reshape(-1, 2)
    return coords[None, :, :] - coords[:, None, :]


class WindowChannelAttention(nn.Module):
    """Channel attention on channels-last features with separate average- and max-pool MLPs."""

    def __init__(self, input_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        middle = input_channels // reduction_ratio
        if middle < 10:
            middle = input_channels
        self.MLP1 = nn.Sequential(nn.Flatten(), nn.Linear(input_channels, middle), nn.ReLU(), nn.Linear(middle, input_channels))
        self.MLP2 = nn.Sequential(nn.Flatten(), nn.Linear(input_channels, middle), nn.ReLU(), nn.Linear(middle, input_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        weights = torch.sigmoid(self.MLP1(self.avg_pool(x)) + self.MLP2(self.max_pool(x)))
        return (x * weights[:, :, None, None]).permute(0, 2, 3, 1)


class WindowAttention(nn.Module):
    """(Shifted) window multi-head self-attention on ``(B, H, W, C)`` features."""

    def __init__(self, dim: int, heads: int, head_dim: int, shifted: bool, window_size: int, relative_pos_embedding: bool):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted
        if shifted:
            self.displacement = window_size // 2
            # Parameters without gradient (not buffers), as in the reference: same state-dict keys
            # and the same parameter count.
            self.upper_lower_mask = nn.Parameter(
                create_mask(window_size, self.displacement, upper_lower=True, left_right=False), requires_grad=False
            )
            self.left_right_mask = nn.Parameter(
                create_mask(window_size, self.displacement, upper_lower=False, left_right=True), requires_grad=False
            )
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        if relative_pos_embedding:
            self.register_buffer("relative_indices", get_relative_distances(window_size) + window_size - 1, persistent=False)
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))
        self.to_out = nn.Linear(inner_dim, dim)
        self.ca = WindowChannelAttention(dim)

    def _to_windows(self, t: torch.Tensor, nw_h: int, nw_w: int) -> torch.Tensor:
        b, w = t.shape[0], self.window_size
        t = t.reshape(b, nw_h, w, nw_w, w, self.heads, self.head_dim)
        return t.permute(0, 5, 1, 3, 2, 4, 6).reshape(b, self.heads, nw_h * nw_w, w * w, self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shifted:
            x = torch.roll(x, shifts=(-self.displacement, -self.displacement), dims=(1, 2))
        b, n_h, n_w, _ = x.shape
        w = self.window_size
        nw_h, nw_w = n_h // w, n_w // w
        q, k, v = (self._to_windows(t, nw_h, nw_w) for t in self.to_qkv(x).chunk(3, dim=-1))

        dots = torch.einsum("bhwid,bhwjd->bhwij", q, k) * self.scale
        if self.relative_pos_embedding:
            dots = dots + self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots = dots + self.pos_embedding
        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask  # last row of windows
            dots[:, :, nw_w - 1 :: nw_w] += self.left_right_mask  # last column of windows
        out = torch.einsum("bhwij,bhwjd->bhwid", dots.softmax(dim=-1), v)

        out = out.reshape(b, self.heads, nw_h, nw_w, w, w, self.head_dim)
        out = out.permute(0, 2, 4, 3, 5, 1, 6).reshape(b, n_h, n_w, self.heads * self.head_dim)
        out = self.to_out(out)
        if self.shifted:
            out = torch.roll(out, shifts=(self.displacement, self.displacement), dims=(1, 2))
        return self.ca(out)


class SwinBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, mlp_dim: int, shifted: bool, window_size: int, relative_pos_embedding: bool):
        super().__init__()
        self.attention_block = Residual(
            PreNorm(dim, WindowAttention(dim, heads, head_dim, shifted, window_size, relative_pos_embedding))
        )
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_block(self.attention_block(x))


# --------------------------------------------------------------------------- local branch and gate


class DoubleConv(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ChannelAttention(nn.Module):
    """CBAM channel attention: a shared MLP on average- and max-pooled channels."""

    def __init__(self, input_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        middle = input_channels // reduction_ratio
        if middle <= 0:
            middle = input_channels
        self.MLP = nn.Sequential(nn.Flatten(), nn.Linear(input_channels, middle), nn.ReLU(), nn.Linear(middle, input_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.sigmoid(self.MLP(self.avg_pool(x)) + self.MLP(self.max_pool(x)))
        return x * weights[:, :, None, None]


class SpatialAttention(nn.Module):
    """CBAM spatial attention: 3x3 convolution of the channel mean and max, BatchNorm, sigmoid."""

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([x.mean(dim=1, keepdim=True), x.max(dim=1, keepdim=True)[0]], dim=1)
        return x * torch.sigmoid(self.bn(self.conv(pooled)))


class CASA(nn.Module):
    def __init__(self, in_channel: int):
        super().__init__()
        self.ca = ChannelAttention(in_channel)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


class GateFusionUnit(nn.Module):
    """GFU: gates ``Z`` (global) and ``R`` (local) from the concatenated features."""

    def __init__(self, channel: int, h_w: Sequence[int]):
        super().__init__()
        height, width = h_w
        self.conv_1 = nn.Sequential(
            nn.Conv2d(channel * 2, channel * 2, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([channel * 2, height, width]),
        )
        # conv_2 / conv_3 are built by the reference but not used in its forward pass.
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1), nn.LayerNorm([channel, height, width])
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1), nn.LayerNorm([channel, height, width])
        )
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, g: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        z, r = torch.chunk(self.conv_1(torch.cat((g, l), dim=1)), 2, dim=1)
        return torch.sigmoid(z) * g + torch.sigmoid(r) * l


# --------------------------------------------------------------------------- stages


class PatchMerging(nn.Module):
    """Space-to-depth by ``downscaling_factor`` and a linear projection (channels last out)."""

    def __init__(self, in_channels: int, out_channels: int, downscaling_factor: int):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        f = self.downscaling_factor
        x = x.reshape(b, c, h // f, f, w // f, f).permute(0, 1, 3, 5, 2, 4)
        return self.linear(x.reshape(b, c * f * f, h // f, w // f).permute(0, 2, 3, 1))


class PatchExpanding(nn.Module):
    """Depth-to-space by ``upscaling_factor`` and a linear projection (channels last out)."""

    def __init__(self, in_channels: int, out_channels: int, upscaling_factor: int):
        super().__init__()
        self.upscaling_factor = upscaling_factor
        self.linear = nn.Linear(in_channels // (upscaling_factor ** 2), out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        f = self.upscaling_factor
        new_c = c // (f * f)
        x = x.reshape(b, new_c, f, f, h, w).permute(0, 1, 4, 2, 5, 3)
        return self.linear(x.reshape(b, new_c, h * f, w * f).permute(0, 2, 3, 1))


def _febm(dim: int, heads: int, head_dim: int, mlp_dim: int, window_size: int, relative_pos_embedding: bool, h_w) -> nn.ModuleList:
    """One feature-extraction-balance module: W-MSA and SW-MSA blocks, CNN, CASA and the gate."""
    attention = dict(heads=heads, head_dim=head_dim, mlp_dim=mlp_dim, window_size=window_size, relative_pos_embedding=relative_pos_embedding)
    return nn.ModuleList(
        [
            SwinBlock(dim=dim, shifted=False, **attention),
            SwinBlock(dim=dim, shifted=True, **attention),
            DoubleConv(dim, dim),
            CASA(dim),
            GateFusionUnit(dim, h_w),
        ]
    )


def _run_febms(layers: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
    """``x`` channels last; returns channels first. Every group reads ``x``; the last one wins."""
    out = None
    for regular_block, shifted_block, cnn, casa, gate in layers:
        local_x = casa(cnn(x.permute(0, 3, 1, 2)))
        global_x = shifted_block(regular_block(x)).permute(0, 3, 1, 2)
        out = gate(global_x, local_x)
    return out


class StageModule(nn.Module):
    """Encoder stage (and the last decoder stage with ``expand=True``): resample, then FEBM."""

    def __init__(self, in_channels: int, hidden_dimension: int, layers: int, scaling_factor: int, num_heads: int, head_dim: int, window_size: int, relative_pos_embedding: bool, h_w, expand: bool = False):
        super().__init__()
        if expand:
            self.patch_partition = PatchExpanding(in_channels, hidden_dimension, scaling_factor)
        else:
            self.patch_partition = PatchMerging(in_channels, hidden_dimension, scaling_factor)
        self.layers = nn.ModuleList(
            [
                _febm(hidden_dimension, num_heads, head_dim, hidden_dimension * 4, window_size, relative_pos_embedding, h_w)
                for _ in range(layers // 2)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _run_febms(self.layers, self.patch_partition(x))


class StageModuleUp(nn.Module):
    """Decoder stage: patch expanding, concatenation with the encoder skip, then FEBM."""

    def __init__(self, in_channels: int, hidden_dimension: int, layers: int, upscaling_factor: int, num_heads: int, head_dim: int, window_size: int, relative_pos_embedding: bool, h_w):
        super().__init__()
        self.patch_partition = PatchExpanding(in_channels, hidden_dimension, upscaling_factor)
        self.in_channel = in_channels
        self.hidden_dimension = hidden_dimension
        self.layers = nn.ModuleList(
            [
                _febm(hidden_dimension * 2, num_heads, head_dim, hidden_dimension * 4, window_size, relative_pos_embedding, h_w)
                for _ in range(layers // 2)
            ]
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat((self.patch_partition(x), skip.permute(0, 2, 3, 1)), dim=-1)
        return _run_febms(self.layers, x)


def rainformer_stage_sizes(img_size: Union[int, Sequence[int]], downscaling_factors: Sequence[int]) -> list:
    """Feature-map sizes ``(h, w)`` of stages 1-8 for an input of ``img_size``.

    Stages 1-4 divide by the downscaling factors, stages 5-7 mirror stages 3-1 and stage 8 is at
    the input resolution: 288x288 with (4, 2, 2, 2) gives 72, 36, 18, 9, 18, 36, 72, 288.
    """
    h, w = to_2tuple(img_size)
    sizes = []
    for factor in downscaling_factors:
        if factor <= 0 or h % factor or w % factor:
            raise ValueError(
                f"Rainformer input size {to_2tuple(img_size)} is not divisible by the downscaling factors "
                f"{tuple(downscaling_factors)} (stage {len(sizes) + 1} gets {(h, w)})."
            )
        h, w = h // factor, w // factor
        sizes.append((h, w))
    return sizes + [sizes[2], sizes[1], sizes[0], to_2tuple(img_size)]


class Rainformer(nn.Module):
    """Rainformer (official ``Net``): ``input_channel`` frames in, as many frames out.

    Input ``(B, input_channel, H, W)`` (frames stacked as channels, the official layout) or
    ``(B, T, C, H, W)`` with ``T * C == input_channel`` (flattened time-major and reshaped back).
    """

    def __init__(
        self,
        input_channel: int = 9,
        hidden_dim: int = 96,
        downscaling_factors: Sequence[int] = (4, 2, 2, 2),
        layers: Sequence[int] = (2, 2, 2, 2),
        heads: Sequence[int] = (3, 6, 12, 24),
        head_dim: int = 32,
        window_size: int = 9,
        relative_pos_embedding: bool = True,
        img_size: Union[int, Sequence[int]] = 288,
    ):
        super().__init__()
        d, layers, heads = tuple(downscaling_factors), tuple(layers), tuple(heads)
        if len(d) != 4 or len(layers) != 4 or len(heads) != 4:
            raise ValueError("Rainformer needs four downscaling_factors, layers and heads (one per encoder stage).")
        if any(n < 2 or n % 2 for n in layers):
            raise ValueError(f"Rainformer stage layers must be even and >= 2 (regular + shifted block), got {layers}.")
        if window_size < 2:
            raise ValueError(f"window_size must be >= 2 for shifted windows, got {window_size}.")
        if input_channel <= 0 or hidden_dim <= 0:
            raise ValueError(f"input_channel and hidden_dim must be positive, got {input_channel} and {hidden_dim}.")
        sizes = rainformer_stage_sizes(img_size, d)
        for stage, (h, w) in enumerate(sizes, start=1):
            if h % window_size or w % window_size:
                raise ValueError(
                    f"Rainformer stage {stage} feature map {(h, w)} (input {to_2tuple(img_size)}) is not divisible by "
                    f"window_size {window_size}; the official 288x288 setting gives 72/36/18/9 with window 9."
                )
        for channels, factor in ((hidden_dim * 8, d[3]), (hidden_dim * 8, d[2]), (hidden_dim * 4, d[1]), (hidden_dim * 2, d[0])):
            if channels % (factor * factor):
                raise ValueError(f"patch expanding needs {channels} channels divisible by {factor}**2; change hidden_dim.")
        self.input_channel = input_channel
        self.img_size = to_2tuple(img_size)
        self.stage_sizes = sizes

        common = dict(head_dim=head_dim, window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage1 = StageModule(input_channel, hidden_dim, layers[0], d[0], heads[0], h_w=sizes[0], **common)
        self.stage2 = StageModule(hidden_dim, hidden_dim * 2, layers[1], d[1], heads[1], h_w=sizes[1], **common)
        self.stage3 = StageModule(hidden_dim * 2, hidden_dim * 4, layers[2], d[2], heads[2], h_w=sizes[2], **common)
        self.stage4 = StageModule(hidden_dim * 4, hidden_dim * 8, layers[3], d[3], heads[3], h_w=sizes[3], **common)
        self.stage5 = StageModuleUp(hidden_dim * 8, hidden_dim * 4, layers[3], d[3], heads[3], h_w=sizes[4], **common)
        self.stage6 = StageModuleUp(hidden_dim * 8, hidden_dim * 2, layers[2], d[2], heads[2], h_w=sizes[5], **common)
        self.stage7 = StageModuleUp(hidden_dim * 4, hidden_dim, layers[1], d[1], heads[1], h_w=sizes[6], **common)
        self.stage8 = StageModule(hidden_dim * 2, input_channel, layers[0], d[0], heads[0], h_w=sizes[7], expand=True, **common)

    def _check_input(self, x: torch.Tensor) -> None:
        layout = None
        if x.ndim == 4:
            layout = x.shape[1] == self.input_channel
        elif x.ndim == 5:
            layout = x.shape[1] * x.shape[2] == self.input_channel
        if not layout or tuple(x.shape[-2:]) != self.img_size:
            raise ValueError(
                f"Rainformer expects input shape (batch, {self.input_channel}, {self.img_size[0]}, {self.img_size[1]}) or "
                f"(batch, time, channels, {self.img_size[0]}, {self.img_size[1]}) with time * channels = "
                f"{self.input_channel}, got {tuple(x.shape)}."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        frames = x.shape[:3] if x.ndim == 5 else None
        if frames is not None:
            x = x.flatten(1, 2)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4, x3)
        x6 = self.stage6(x5, x2)
        x7 = self.stage7(x6, x1)
        out = self.stage8(x7)
        return out.reshape(*frames, *out.shape[-2:]) if frames is not None else out


class RainformerSegmenter(Rainformer):
    """PyHazards adaptation for next-step masks: ``(B, T, C, H, W)`` -> logits ``(B, out, H, W)``.

    Runs :class:`Rainformer` and maps the ``C`` channels of the first predicted frame to
    ``out_channels`` logits with a 1x1 convolution (``segmentation_head``). The Rainformer
    parameters keep their reference names at the top level.
    """

    def __init__(self, in_channels: int, out_channels: int = 1, **kwargs):
        super().__init__(**kwargs)
        if in_channels <= 0 or self.input_channel % in_channels:
            raise ValueError(f"in_channels {in_channels} must divide input_channel {self.input_channel}.")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}.")
        self.in_channels = in_channels
        self.segmentation_head = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[2] != self.in_channels:
            raise ValueError(
                f"Rainformer segmentation expects input shape (batch, time, {self.in_channels}, height, width), "
                f"got {tuple(x.shape)}."
            )
        return self.segmentation_head(super().forward(x)[:, 0])


def rainformer_builder(
    task: str,
    in_channels: int = 1,
    history: int = 9,
    img_size: Union[int, Sequence[int]] = 288,
    out_channels: int = 1,
    hidden_dim: int = 96,
    downscaling_factors: Sequence[int] = (4, 2, 2, 2),
    layers: Sequence[int] = (2, 2, 2, 2),
    heads: Sequence[int] = (3, 6, 12, 24),
    head_dim: int = 32,
    window_size: int = 9,
    relative_pos_embedding: bool = True,
    **kwargs,
) -> nn.Module:
    """Rainformer with the official KNMI configuration (``train.py``: 9 frames, 288x288).

    ``task="forecasting"``: ``(B, history, in_channels, H, W)`` -> the next ``history`` frames.
    ``task="segmentation"``: the same input -> ``(B, out_channels, H, W)`` logits (PyHazards
    adaptation, see :class:`RainformerSegmenter`).
    """
    _ = kwargs
    task = task.lower()
    if task not in {"forecasting", "segmentation"}:
        raise ValueError(f"rainformer supports task='forecasting' or 'segmentation', got {task!r}.")
    if history <= 0 or in_channels <= 0:
        raise ValueError(f"history and in_channels must be positive, got {history} and {in_channels}.")
    config = dict(
        input_channel=history * in_channels, hidden_dim=hidden_dim, downscaling_factors=downscaling_factors,
        layers=layers, heads=heads, head_dim=head_dim, window_size=window_size,
        relative_pos_embedding=relative_pos_embedding, img_size=img_size,
    )
    if task == "segmentation":
        return RainformerSegmenter(in_channels=in_channels, out_channels=out_channels, **config)
    return Rainformer(**config)


__all__ = ["Rainformer", "RainformerSegmenter", "rainformer_builder", "rainformer_stage_sizes"]
