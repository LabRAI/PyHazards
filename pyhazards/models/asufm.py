from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


def _window_partition(x: torch.Tensor, window_size: int) -> tuple[torch.Tensor, int, int]:
    b, h, w, c = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = x.permute(0, 3, 1, 2).contiguous()
        x = F.pad(x, (0, pad_w, 0, pad_h))
        x = x.permute(0, 2, 3, 1).contiguous()
    hp, wp = h + pad_h, w + pad_w
    x = x.view(b, hp // window_size, window_size, wp // window_size, window_size, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(-1, window_size * window_size, c), hp, wp


def _window_reverse(windows: torch.Tensor, window_size: int, hp: int, wp: int, batch_size: int) -> torch.Tensor:
    channels = windows.shape[-1]
    x = windows.view(batch_size, hp // window_size, wp // window_size, window_size, window_size, channels)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(batch_size, hp, wp, channels)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 4,
        in_channels: int = 6,
        embed_dim: int = 96,
        patch_norm: bool = True,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.embed_dim = int(embed_dim)
        self.proj = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.norm = nn.LayerNorm(self.embed_dim) if patch_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"PatchEmbed expected (B,C,H,W), got {tuple(x.shape)}")
        _, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(f"PatchEmbed expected {self.in_channels} channels, got {channels}")
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"PatchEmbed expected spatial size ({self.image_size}, {self.image_size}), "
                f"got ({height}, {width})"
            )
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class PatchMerging(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(f"PatchMerging requires even spatial dims, got ({h}, {w})")
        x = x.permute(0, 2, 3, 1).contiguous()
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x.permute(0, 3, 1, 2).contiguous()


class PatchExpand(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.expand = nn.Conv2d(dim, 2 * dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = F.pixel_shuffle(x, upscale_factor=2)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class FinalPatchExpandX4(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.expand = nn.Conv2d(dim, 16 * dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = F.pixel_shuffle(x, upscale_factor=4)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class FocalModulation(nn.Module):
    def __init__(
        self,
        dim: int,
        focal_window: int = 3,
        focal_level: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.focal_level = int(focal_level)
        self.proj = nn.Linear(self.dim, 2 * self.dim + self.focal_level + 1)
        self.depthwise_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    self.dim,
                    self.dim,
                    kernel_size=focal_window + 2 * level,
                    padding=(focal_window + 2 * level) // 2,
                    groups=self.dim,
                    bias=False,
                )
                for level in range(self.focal_level)
            ]
        )
        self.mix = nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=False)
        self.out = nn.Linear(self.dim, self.dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"FocalModulation expected (B,H,W,C), got {tuple(x.shape)}")
        gates_dim = self.focal_level + 1
        projected = self.proj(x)
        q, ctx, gates = torch.split(projected, [self.dim, self.dim, gates_dim], dim=-1)
        ctx = ctx.permute(0, 3, 1, 2).contiguous()
        gates = gates.permute(0, 3, 1, 2).contiguous()

        aggregated = 0.0
        for level, layer in enumerate(self.depthwise_layers):
            aggregated = aggregated + layer(ctx) * gates[:, level : level + 1]
        global_ctx = ctx.mean(dim=(2, 3), keepdim=True)
        aggregated = aggregated + global_ctx * gates[:, -1:]

        modulator = self.mix(aggregated).permute(0, 2, 3, 1).contiguous()
        out = q * modulator
        out = self.out(out)
        return self.drop(out)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwinFocalBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        use_focal: bool = False,
        focal_window: int = 3,
        focal_level: int = 2,
    ):
        super().__init__()
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.use_focal = bool(use_focal)
        self.norm1 = nn.LayerNorm(self.dim)
        self.focal = (
            FocalModulation(
                dim=self.dim,
                focal_window=focal_window,
                focal_level=focal_level,
                dropout=dropout,
            )
            if self.use_focal
            else None
        )
        self.attn = nn.MultiheadAttention(self.dim, num_heads=int(num_heads), dropout=dropout, batch_first=True)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(self.dim)
        self.mlp = MLP(dim=self.dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"SwinFocalBlock expected (B,C,H,W), got {tuple(x.shape)}")
        b, c, h, w = x.shape
        if c != self.dim:
            raise ValueError(f"SwinFocalBlock expected {self.dim} channels, got {c}")

        x_hw = x.permute(0, 2, 3, 1).contiguous()
        x_hw = self.norm1(x_hw)
        if self.focal is not None:
            x_hw = self.focal(x_hw)

        effective_window = min(self.window_size, h, w)
        effective_shift = 0 if effective_window <= 1 else min(self.shift_size, effective_window // 2)
        if effective_shift > 0:
            x_hw = torch.roll(x_hw, shifts=(-effective_shift, -effective_shift), dims=(1, 2))

        windows, hp, wp = _window_partition(x_hw, effective_window)
        attn_out, _ = self.attn(windows, windows, windows, need_weights=False)
        windows = windows + attn_out
        x_hw = _window_reverse(windows, effective_window, hp, wp, b)

        if effective_shift > 0:
            x_hw = torch.roll(x_hw, shifts=(effective_shift, effective_shift), dims=(1, 2))
        x_hw = x_hw[:, :h, :w, :]
        x_attn = x_hw.permute(0, 3, 1, 2).contiguous()
        x = x + self.drop_path(x_attn)

        tokens = x.permute(0, 2, 3, 1).reshape(b, h * w, c).contiguous()
        tokens = tokens + self.drop_path(self.mlp(self.norm2(tokens)))
        return tokens.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class EncoderStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        dropout: float,
        drop_path_rates: Sequence[float],
        use_focal: bool,
        focal_window: int,
        focal_level: int,
        downsample: bool,
    ):
        super().__init__()
        shift = max(1, window_size // 2)
        self.blocks = nn.ModuleList(
            [
                SwinFocalBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if block_idx % 2 == 0 else shift,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=drop_path_rates[block_idx],
                    use_focal=use_focal,
                    focal_window=focal_window,
                    focal_level=focal_level,
                )
                for block_idx in range(depth)
            ]
        )
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x)
        skip = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x, skip


class SpatialAttentionGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, 1, kernel_size=1),
        )

    def forward(self, skip: torch.Tensor, gating: torch.Tensor) -> torch.Tensor:
        if gating.shape[-2:] != skip.shape[-2:]:
            gating = F.interpolate(gating, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        mask = torch.sigmoid(self.project(torch.cat([skip, gating], dim=1)))
        return skip * mask


class DecoderStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        dropout: float,
        drop_path_rates: Sequence[float],
        spatial_attention: bool,
    ):
        super().__init__()
        self.skip_gate = SpatialAttentionGate(dim) if spatial_attention else None
        self.concat_proj = nn.Conv2d(2 * dim, dim, kernel_size=1, bias=False)
        shift = max(1, window_size // 2)
        self.blocks = nn.ModuleList(
            [
                SwinFocalBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if block_idx % 2 == 0 else shift,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=drop_path_rates[block_idx],
                    use_focal=False,
                )
                for block_idx in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        if skip is not None:
            if self.skip_gate is not None:
                skip = self.skip_gate(skip, x)
            x = self.concat_proj(torch.cat([x, skip], dim=1))
        for block in self.blocks:
            x = block(x)
        return x


class ASUFM(nn.Module):
    """
    Self-contained ASUFM port for PyHazards.

    This implementation follows the official ASUFM design at a high level:
    patch embedding, hierarchical Swin-style encoder stages, focal modulation in
    the encoder, and an attention-gated U-Net-style decoder. It intentionally
    avoids external dependencies such as `timm` and `einops` so the model can be
    built directly inside the main PyHazards library.
    """

    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 4,
        in_channels: int = 6,
        out_dim: int = 1,
        embed_dim: int = 96,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        focal_window: int = 3,
        focal_level: int = 2,
        use_focal_modulation: bool = True,
        spatial_attention: bool = True,
        skip_num: int = 3,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        _ = use_checkpoint

        if len(depths) != 4:
            raise ValueError(f"ASUFM expects 4 encoder depths, got {tuple(depths)}")
        if len(num_heads) != len(depths):
            raise ValueError("num_heads must have the same length as depths")
        if skip_num < 0 or skip_num > 3:
            raise ValueError(f"skip_num must be in [0, 3], got {skip_num}")

        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.out_dim = int(out_dim)
        self.skip_num = int(skip_num)

        dims = [int(embed_dim * (2**idx)) for idx in range(len(depths))]
        for dim, heads in zip(dims, num_heads):
            if dim % int(heads) != 0:
                raise ValueError(f"Channel dim {dim} must be divisible by num_heads={heads}")

        total_blocks = sum(int(depth) for depth in depths)
        drop_path_values = torch.linspace(0.0, float(drop_path_rate), total_blocks).tolist()

        self.patch_embed = PatchEmbed(
            image_size=self.image_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=int(embed_dim),
            patch_norm=True,
        )

        cursor = 0
        self.encoder_stages = nn.ModuleList()
        for stage_idx, (dim, depth, heads) in enumerate(zip(dims, depths, num_heads)):
            stage_dpr = drop_path_values[cursor : cursor + depth]
            cursor += depth
            self.encoder_stages.append(
                EncoderStage(
                    dim=dim,
                    depth=int(depth),
                    num_heads=int(heads),
                    window_size=int(window_size),
                    mlp_ratio=float(mlp_ratio),
                    dropout=float(dropout),
                    drop_path_rates=stage_dpr,
                    use_focal=bool(use_focal_modulation),
                    focal_window=int(focal_window),
                    focal_level=int(focal_level),
                    downsample=stage_idx < len(depths) - 1,
                )
            )

        reverse_depths = list(reversed(depths[:-1]))
        reverse_heads = list(reversed(num_heads[:-1]))
        reverse_dims = list(reversed(dims[:-1]))
        reverse_drop_paths = list(reversed(drop_path_values[:-depths[-1]]))

        self.upsamplers = nn.ModuleList(
            [
                PatchExpand(dim=dims[-1]),
                PatchExpand(dim=dims[-2]),
                PatchExpand(dim=dims[-3]),
            ]
        )
        self.decoder_stages = nn.ModuleList()
        cursor = 0
        for dim, depth, heads in zip(reverse_dims, reverse_depths, reverse_heads):
            stage_dpr = reverse_drop_paths[cursor : cursor + depth]
            cursor += depth
            self.decoder_stages.append(
                DecoderStage(
                    dim=dim,
                    depth=int(depth),
                    num_heads=int(heads),
                    window_size=int(window_size),
                    mlp_ratio=float(mlp_ratio),
                    dropout=float(dropout),
                    drop_path_rates=stage_dpr,
                    spatial_attention=bool(spatial_attention),
                )
            )

        self.norm_up = nn.LayerNorm(dims[0])
        self.final_up = FinalPatchExpandX4(dim=dims[0])
        self.output_head = nn.Conv2d(dims[0], self.out_dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"ASUFM expected input of shape (B,C,H,W), got {tuple(x.shape)}")
        _, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(f"ASUFM expected {self.in_channels} input channels, got {channels}")

        required_factor = self.patch_size * (2 ** (len(self.encoder_stages) - 1))
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"ASUFM expected image_size={self.image_size}, got spatial size ({height}, {width})"
            )
        if height % required_factor != 0 or width % required_factor != 0:
            raise ValueError(
                f"ASUFM requires H and W divisible by {required_factor}, got ({height}, {width})"
            )

        x = self.patch_embed(x)
        skips: list[torch.Tensor] = []
        for stage_idx, stage in enumerate(self.encoder_stages):
            x, skip = stage(x)
            if stage_idx < len(self.encoder_stages) - 1:
                skips.append(skip)

        for decoder_idx, (upsample, decoder_stage) in enumerate(zip(self.upsamplers, self.decoder_stages), start=1):
            x = upsample(x)
            skip = skips[-decoder_idx] if decoder_idx <= self.skip_num else None
            x = decoder_stage(x, skip)

        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm_up(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.final_up(x)
        return self.output_head(x)


def asufm_builder(
    task: str,
    image_size: int = 64,
    patch_size: int = 4,
    in_channels: int = 6,
    out_dim: int = 1,
    embed_dim: int = 96,
    depths: Sequence[int] = (2, 2, 2, 2),
    num_heads: Sequence[int] = (3, 6, 12, 24),
    window_size: int = 8,
    mlp_ratio: float = 4.0,
    dropout: float = 0.0,
    drop_path_rate: float = 0.1,
    focal_window: int = 3,
    focal_level: int = 2,
    use_focal_modulation: bool = True,
    spatial_attention: bool = True,
    skip_num: int = 3,
    use_checkpoint: bool = False,
    in_chans: int | None = None,
    num_classes: int | None = None,
    focal: bool | None = None,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    normalized_task = task.lower()
    if normalized_task != "segmentation":
        raise ValueError(f"ASUFM is segmentation-only. Got task='{task}'")

    if in_chans is not None:
        in_channels = int(in_chans)
    if num_classes is not None:
        out_dim = int(num_classes)
    if focal is not None:
        use_focal_modulation = bool(focal)

    return ASUFM(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        out_dim=out_dim,
        embed_dim=embed_dim,
        depths=tuple(int(v) for v in depths),
        num_heads=tuple(int(v) for v in num_heads),
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        drop_path_rate=drop_path_rate,
        focal_window=focal_window,
        focal_level=focal_level,
        use_focal_modulation=use_focal_modulation,
        spatial_attention=spatial_attention,
        skip_num=skip_num,
        use_checkpoint=use_checkpoint,
    )


__all__ = ["ASUFM", "asufm_builder"]
