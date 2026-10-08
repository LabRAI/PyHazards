"""Swin-Unet: a U-shaped pure Swin Transformer for dense prediction.

Architecture paper: Cao et al., "Swin-Unet: Unet-like Pure Transformer for Medical Image
Segmentation", ECCV Workshops 2022 (arXiv:2105.05537). Wildfire usage: Lahrichi, Bova, Johnson
and Malof, "Improved Wildfire Spread Prediction with Time-Series Data and the WSTS+ Benchmark",
WACV 2026 (arXiv:2502.12003), which trains "SwinUnet-Tiny" on WildfireSpreadTS.

The official code (github.com/HuCaoFighting/Swin-Unet) has no license, so this module is written
from the paper and from the MIT-licensed microsoft/Swin-Transformer blocks in
:mod:`pyhazards.models.swin_blocks`; the official code is only used as a test oracle
(tests/oracle/test_swin_oracle.py). Module and parameter names match the official
``SwinTransformerSys``, so its state dicts load with ``strict=True``, and modules are created in
the same order, so the same seed gives the same initial weights.

Behaviour of the official code that is kept:

- the decoder mirrors the encoder depths (``depths[2], depths[1], depths[0]`` after a plain
  patch-expanding layer); the official ``depths_decoder`` argument and the yaml's
  ``DECODER_DEPTHS`` are never used, so they are not exposed here;
- decoder blocks reuse the stochastic-depth rates of the mirrored encoder stage;
- :meth:`SwinUnet.load_swin_pretrained` reproduces the official ``load_from``: Swin-T ImageNet
  weights initialise the encoder *and* (stage ``i`` copied to ``layers_up.{3-i}``) the decoder,
  and any tensor whose shape differs (e.g. the patch embedding when ``in_chans != 3``) keeps its
  random initialisation.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

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
    trunc_normal_,
)

# Swin-T ImageNet-1k checkpoint from microsoft/Swin-Transformer (MIT), the Swin-Unet initialisation.
SWIN_TINY_IMAGENET_URL = (
    "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"
)
SWIN_TINY_IMAGENET_SHA256 = "9f71c168d837d1b99dd1dc29e14990a7a9e8bdc5f673d46b04fe36fe15590ad3"


def _swin_tiny_imagenet_path() -> Path:
    path = Path(torch.hub.get_dir()) / "checkpoints" / Path(SWIN_TINY_IMAGENET_URL).name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(
            SWIN_TINY_IMAGENET_URL, str(path), hash_prefix=SWIN_TINY_IMAGENET_SHA256, progress=False
        )
    return path


class SwinUnet(nn.Module):
    """Swin-Unet (``SwinTransformerSys`` with ``final_upsample="expand_first"``).

    Input ``(batch, in_chans, img_size, img_size)``, or ``(batch, time, channels, img_size,
    img_size)`` flattened time-major to ``time * channels`` input channels (data-level fusion as
    in WSTS+). Returns logits ``(batch, num_classes, img_size, img_size)``.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 1,
        embed_dim: int = 96,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        depths, num_heads = list(depths), list(num_heads)
        if len(depths) != len(num_heads):
            raise ValueError(f"depths and num_heads need the same length, got {depths} and {num_heads}.")
        if in_chans <= 0 or num_classes <= 0:
            raise ValueError(f"in_chans and num_classes must be positive, got {in_chans} and {num_classes}.")
        check_swin_geometry(img_size, patch_size, window_size, len(depths))
        norm_layer = nn.LayerNorm

        self.img_size = int(img_size)
        self.in_chans = int(in_chans)
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = "expand_first"

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        resolution = self.patch_embed.patches_resolution
        self.patches_resolution = resolution
        if ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = stochastic_depth_rates(drop_path_rate, depths)
        stage_kwargs = dict(
            window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer, use_checkpoint=use_checkpoint,
        )

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(
                BasicLayer(
                    dim=int(embed_dim * 2 ** i),
                    input_resolution=(resolution[0] // 2 ** i, resolution[1] // 2 ** i),
                    depth=depths[i],
                    num_heads=num_heads[i],
                    drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    downsample=i < self.num_layers - 1,
                    **stage_kwargs,
                )
            )

        # Decoder: step i works at encoder stage j = num_layers - 1 - i. The skip-fusion linear
        # is created before its stage (reference order); layers_up is registered first.
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
                    drop_path=dpr[sum(depths[:j]) : sum(depths[: j + 1])],
                    upsample=i < self.num_layers - 1,
                    **stage_kwargs,
                )
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(embed_dim)
        self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size), dim_scale=4, dim=embed_dim)
        self.output = nn.Conv2d(in_channels=embed_dim, out_channels=num_classes, kernel_size=1, bias=False)
        self.apply(init_swin_weights)

    def forward_features(self, x: torch.Tensor):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        skips = []
        for layer in self.layers:
            skips.append(x)
            x = layer(x)
        return self.norm(x), skips

    def forward_up_features(self, x: torch.Tensor, skips) -> torch.Tensor:
        for i, layer_up in enumerate(self.layers_up):
            if i > 0:
                x = self.concat_back_dim[i](torch.cat([x, skips[self.num_layers - 1 - i]], -1))
            x = layer_up(x)
        return self.norm_up(x)

    def up_x4(self, x: torch.Tensor) -> torch.Tensor:
        h, w = self.patches_resolution
        b = x.shape[0]
        x = self.up(x).view(b, 4 * h, 4 * w, -1).permute(0, 3, 1, 2)
        return self.output(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            x = x.flatten(start_dim=1, end_dim=2)
        if x.ndim != 4:
            raise ValueError(
                "SwinUnet expects input shape (batch, channels, height, width) or "
                f"(batch, time, channels, height, width), got {tuple(x.shape)}."
            )
        if x.size(1) == 1 and self.in_chans == 3:
            x = x.repeat(1, 3, 1, 1)  # official SwinUnet wrapper: grey-scale input to RGB
        if x.size(1) != self.in_chans:
            raise ValueError(f"SwinUnet expected {self.in_chans} input channels, got shape {tuple(x.shape)}.")
        if tuple(x.shape[-2:]) != (self.img_size, self.img_size):
            raise ValueError(
                f"SwinUnet expects height and width {self.img_size} (pad smaller inputs; WSTS+ zero-pads "
                f"128 to 224), got shape {tuple(x.shape)}."
            )
        x, skips = self.forward_features(x)
        return self.up_x4(self.forward_up_features(x, skips))

    def load_swin_pretrained(self, checkpoint: Union[str, Path, Mapping[str, torch.Tensor]]):
        """Initialise encoder and decoder from a Swin Transformer classification checkpoint.

        ``checkpoint`` is ``"imagenet"`` (Swin-T ImageNet-1k, downloaded and sha256-checked), a
        path to a microsoft/Swin-Transformer checkpoint, or its loaded dict (``{"model": state}``
        or the state itself). Mirrors the official ``SwinUnet.load_from``; returns the
        ``load_state_dict`` result.
        """
        if isinstance(checkpoint, str) and checkpoint == "imagenet":
            checkpoint = _swin_tiny_imagenet_path()
        if isinstance(checkpoint, (str, Path)):
            checkpoint = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        state = checkpoint["model"] if "model" in checkpoint else checkpoint

        full = copy.deepcopy(dict(state))
        last = self.num_layers - 1
        for key, value in state.items():
            if "layers." in key:
                full["layers_up." + str(last - int(key[7:8])) + key[8:]] = value
        own = self.state_dict()
        for key in list(full):
            if key in own and full[key].shape != own[key].shape:
                del full[key]
        return self.load_state_dict(full, strict=False)


def swin_unet_builder(
    task: str,
    in_channels: int,
    out_channels: int = 1,
    history: int = 1,
    img_size: int = 224,
    patch_size: int = 4,
    embed_dim: int = 96,
    depths: Sequence[int] = (2, 2, 2, 2),
    num_heads: Sequence[int] = (3, 6, 12, 24),
    window_size: int = 7,
    mlp_ratio: float = 4.0,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.2,
    use_checkpoint: bool = False,
    pretrained: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    """Swin-Unet-Tiny with the official configuration (``swin_tiny_patch4_window7_224_lite.yaml``).

    ``history > 1`` multiplies the input channels (multi-day input is flattened, as in WSTS+).
    ``pretrained`` is None, ``"imagenet"`` or a path to a Swin checkpoint.
    """
    _ = kwargs
    if task.lower() != "segmentation":
        raise ValueError(f"swin_unet supports task='segmentation', got {task!r}.")
    if history <= 0:
        raise ValueError(f"history must be positive, got {history}")
    model = SwinUnet(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_channels * history,
        num_classes=out_channels,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        use_checkpoint=use_checkpoint,
    )
    if pretrained is not None:
        model.load_swin_pretrained(pretrained)
    return model


__all__ = ["SWIN_TINY_IMAGENET_SHA256", "SWIN_TINY_IMAGENET_URL", "SwinUnet", "swin_unet_builder"]
