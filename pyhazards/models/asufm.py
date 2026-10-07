"""ASUFM: Attention Swin U-Net with Focal Modulation for next-day wildfire spread.

Paper: Li & Rad, "Wildfire Spread Prediction in North America Using Satellite Imagery and Vision
Transformer", IEEE CAI 2024, pp. 1536-1541, doi:10.1109/CAI59869.2024.00278. Official code:
github.com/bronteee/fire-asufm at ``688dda94dec9e3b22f149f2b9335d3b38fd2277f`` (Apache-2.0,
Copyright 2024 Bronte Sihan Li); its configurations ``get_asfum_6_configs`` and
``get_asufm_12_configs`` are reproduced here.

Licensing of the port: the fire-asufm model package states it is "largely based on" Attention
Swin U-Net (github.com/NITR098/AttSwinUNet, no license), which in turn builds on Swin-Unet (no
license) and Swin Transformer (MIT). This module therefore does not copy the fire-asufm/AttSwinUNet
sources. The Swin blocks and focal modulation come from :mod:`pyhazards.models.swin_blocks`
(ported from microsoft/Swin-Transformer and microsoft/FocalNet, both MIT); the encoder/decoder
assembly below is written for PyHazards to reproduce the behaviour of the fire-asufm code
(Apache-2.0), with its parameter names and creation order so that its state dicts load with
``strict=True`` and seeded initialisation is identical. The official code is used only as a test
oracle (tests/oracle/test_swin_oracle.py).

Behaviour of the official code that is kept (it changes outputs or parameter names):

- encoder blocks apply focal modulation between ``norm1`` and window attention, followed by
  ``norm1`` again; decoder blocks build a focal-modulation module but never apply it;
- in every decoder stage the skip connection is concatenated and projected by
  ``concat_back_dim`` twice (the configured ``spatial_attention="1"`` branch does it once more);
- the top-level ``patch_embed`` (built with ``in_chans = patch_size = 4``) and ``decoder.norm``
  exist but are unused;
- drop-path rate 0.1, patch size 4, patch norm and no absolute position embedding are fixed in
  the official code whatever the config says.

The encoder-to-decoder "spatial attention" maps of the official code are added to a value that
is then discarded, so they never influence the output; they are not computed here.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .swin_blocks import (
    BasicLayer,
    BasicLayer_up,
    FinalPatchExpand_X4,
    PatchEmbed,
    PatchExpand,
    check_swin_geometry,
    init_swin_weights,
    stochastic_depth_rates,
)

PATCH_SIZE = 4
DROP_PATH_RATE = 0.1


class ASUFMEncoder(nn.Module):
    def __init__(
        self,
        in_chans: int,
        img_size: int,
        embed_dim: int,
        depths: Sequence[int],
        num_heads: Sequence[int],
        window_size: int,
        mlp_ratio: float,
        qkv_bias: bool,
        qk_scale: Optional[float],
        drop_rate: float,
        drop_path_rate: float,
        focal: bool,
        use_checkpoint: bool,
    ):
        super().__init__()
        norm_layer = nn.LayerNorm
        self.num_layers = len(depths)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.norm = norm_layer(self.num_features)
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=PATCH_SIZE, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)
        resolution = self.patch_embed.patches_resolution
        self.patches_resolution = resolution
        dpr = stochastic_depth_rates(drop_path_rate, depths)
        self.layers = nn.ModuleList(
            BasicLayer(
                dim=int(embed_dim * 2 ** i),
                input_resolution=(resolution[0] // 2 ** i, resolution[1] // 2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=0.0,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                norm_layer=norm_layer,
                downsample=i < self.num_layers - 1,
                use_checkpoint=use_checkpoint,
                focal_modulation="applied" if focal else None,
            )
            for i in range(self.num_layers)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.pos_drop(self.patch_embed(x))
        skips = []
        for layer in self.layers:
            skips.append(x)
            x = layer(x)
        return self.norm(x), skips


class ASUFMDecoder(nn.Module):
    def __init__(
        self,
        img_size: int,
        patches_resolution: Sequence[int],
        num_classes: int,
        embed_dim: int,
        depths: Sequence[int],
        num_heads: Sequence[int],
        window_size: int,
        mlp_ratio: float,
        qkv_bias: bool,
        qk_scale: Optional[float],
        drop_rate: float,
        drop_path_rate: float,
        focal: bool,
        use_checkpoint: bool,
    ):
        super().__init__()
        norm_layer = nn.LayerNorm
        self.num_layers = len(depths)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.patches_resolution = list(patches_resolution)
        resolution = self.patches_resolution
        dpr = stochastic_depth_rates(drop_path_rate, depths)
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i in range(self.num_layers):
            j = self.num_layers - 1 - i
            dim = int(embed_dim * 2 ** j)
            stage_resolution = (resolution[0] // 2 ** j, resolution[1] // 2 ** j)
            concat_linear = nn.Linear(2 * dim, dim) if i > 0 else nn.Identity()
            if i == 0:
                layer_up = PatchExpand(stage_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(
                    dim=dim,
                    input_resolution=stage_resolution,
                    depth=depths[j],
                    num_heads=num_heads[j],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=0.0,
                    drop_path=dpr[sum(depths[:j]) : sum(depths[: j + 1])],
                    norm_layer=norm_layer,
                    upsample=i < self.num_layers - 1,
                    use_checkpoint=use_checkpoint,
                    focal_modulation="unused" if focal else None,
                )
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)
        self.norm = norm_layer(self.num_features)  # unused, as in the reference
        self.norm_up = norm_layer(embed_dim)
        self.up = FinalPatchExpand_X4(input_resolution=(img_size // PATCH_SIZE, img_size // PATCH_SIZE), dim_scale=4, dim=embed_dim)
        self.output = nn.Conv2d(in_channels=embed_dim, out_channels=num_classes, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, skips: Sequence[torch.Tensor]) -> torch.Tensor:
        for i, layer_up in enumerate(self.layers_up):
            if i > 0:
                skip = skips[self.num_layers - 1 - i]
                # The reference fuses the same skip twice with the same linear layer.
                x = self.concat_back_dim[i](torch.cat([x, skip], -1))
                x = self.concat_back_dim[i](torch.cat([x, skip], -1))
            x = layer_up(x)
        x = self.norm_up(x)
        h, w = self.patches_resolution
        x = self.up(x).view(x.shape[0], 4 * h, 4 * w, -1).permute(0, 3, 1, 2)
        return self.output(x)


class ASUFM(nn.Module):
    """Attention Swin U-Net with Focal Modulation (official ``mode="swin"`` configuration).

    Input ``(batch, in_chans, img_size, img_size)`` (one NDWS day; a 5-D input with a time axis of
    length one is accepted), output logits ``(batch, num_classes, img_size, img_size)``.
    """

    def __init__(
        self,
        in_chans: int = 6,
        num_classes: int = 1,
        img_size: int = 64,
        embed_dim: int = 96,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        drop_path_rate: float = DROP_PATH_RATE,
        focal: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        depths, num_heads = list(depths), list(num_heads)
        if len(depths) != len(num_heads):
            raise ValueError(f"depths and num_heads need the same length, got {depths} and {num_heads}.")
        if in_chans <= 0 or num_classes <= 0:
            raise ValueError(f"in_chans and num_classes must be positive, got {in_chans} and {num_classes}.")
        check_swin_geometry(img_size, PATCH_SIZE, window_size, len(depths))
        self.in_chans = int(in_chans)
        self.num_classes = int(num_classes)
        self.img_size = int(img_size)
        self.focal = focal

        # Unused stem of the reference (its in_chans is the patch size); kept for state-dict parity.
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=PATCH_SIZE, in_chans=PATCH_SIZE, embed_dim=embed_dim, norm_layer=nn.LayerNorm)
        common = dict(
            embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=window_size,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
            drop_path_rate=drop_path_rate, focal=focal, use_checkpoint=use_checkpoint,
        )
        self.encoder = ASUFMEncoder(in_chans=in_chans, img_size=img_size, **common)
        self.decoder = ASUFMDecoder(
            img_size=img_size, patches_resolution=self.patch_embed.patches_resolution, num_classes=num_classes, **common
        )
        self.apply(init_swin_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5 and x.size(1) == 1:
            x = x[:, 0]
        if x.ndim != 4:
            raise ValueError(
                f"ASUFM expects input shape (batch, channels, height, width), got {tuple(x.shape)}."
            )
        if x.size(1) == 1 and self.in_chans == 3:
            x = x.repeat(1, 3, 1, 1)  # reference: grey-scale input to three channels
        if x.size(1) != self.in_chans or tuple(x.shape[-2:]) != (self.img_size, self.img_size):
            raise ValueError(
                f"ASUFM expects input shape (batch, {self.in_chans}, {self.img_size}, {self.img_size}), "
                f"got {tuple(x.shape)}."
            )
        x, skips = self.encoder(x)
        return self.decoder(x, skips)


def asufm_builder(
    task: str,
    in_channels: int = 6,
    out_channels: int = 1,
    img_size: int = 64,
    embed_dim: int = 96,
    depths: Sequence[int] = (2, 2, 2, 2),
    num_heads: Sequence[int] = (3, 6, 12, 24),
    window_size: int = 8,
    mlp_ratio: float = 4.0,
    drop_rate: float = 0.0,
    drop_path_rate: float = DROP_PATH_RATE,
    focal: bool = True,
    use_checkpoint: bool = False,
    **kwargs,
) -> nn.Module:
    """ASUFM with the official configuration (6 NDWS channels by default; 12 for all features)."""
    _ = kwargs
    if task.lower() != "segmentation":
        raise ValueError(f"asufm supports task='segmentation', got {task!r}.")
    return ASUFM(
        in_chans=in_channels,
        num_classes=out_channels,
        img_size=img_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        focal=focal,
        use_checkpoint=use_checkpoint,
    )


__all__ = ["ASUFM", "asufm_builder"]
