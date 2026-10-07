"""Prithvi-EO-2.0-300M-BurnScars: burn-scar segmentation of single HLS scenes.

The official fine-tune (Hugging Face ``ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars``, Apache-2.0)
of the Prithvi-EO-2.0-300M encoder (Szwarcman et al., arXiv:2412.02732) on the HLS Burn Scars dataset,
built with TerraTorch from ``burn_scars_config.yaml``:

    backbone prithvi_eo_v2_300 (ViT-L/16, 1 frame, 6 HLS bands)
    necks    SelectIndices[5, 11, 17, 23] -> ReshapeTokensToImage -> LearnedInterpolateToPyramidal
    decoder  UNetDecoder [512, 256, 128, 64] -> SegmentationHead (2 classes: not burned, burn scar)

The modules are the plain-PyTorch port in :mod:`pyhazards.models.prithvi` (TerraTorch 0.99.8 and
``prithvi_mae.py``, Apache-2.0; smp 0.4.0, MIT). Inputs are HLS surface reflectance (0-1) of bands
B, G, R, NIR (narrow), SWIR 1, SWIR 2, standardised with :data:`BURN_SCARS_MEAN` / :data:`BURN_SCARS_STD`,
with input shape ``(B, 6, H, W)`` or ``(B, 6, 1, H, W)`` (other shapes raise ``ValueError``); the
official chips are 512 x 512. The output is ``(B, 2, H, W)`` logits.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch.nn as nn

from .prithvi import (
    PRITHVI_BANDS,
    PrithviSegmentation,
    PrithviViT,
    finetuned_state_dict_to_model,
    load_pretrained_state_dict,
)

# Data normalisation of the official fine-tune (burn_scars_config.yaml, reflectance units).
BURN_SCARS_BANDS = PRITHVI_BANDS
BURN_SCARS_MEAN = (
    0.033349706741586264,
    0.05701185520536176,
    0.05889748132001316,
    0.2323245113436119,
    0.1972854853760658,
    0.11944914225186566,
)
BURN_SCARS_STD = (
    0.02269135568823774,
    0.026807560223070237,
    0.04004109844362779,
    0.07791732423672691,
    0.08708738838140137,
    0.07241979477437814,
)
BURN_SCARS_CLASSES = ("Not burned", "Burn scar")

_OFFICIAL = dict(
    in_channels=6,
    num_classes=2,
    img_size=224,
    patch_size=16,
    embed_dim=1024,
    depth=24,
    num_heads=16,
    mlp_ratio=4.0,
    select_indices=(5, 11, 17, 23),
    decoder_channels=(512, 256, 128, 64),
)


def prithvi_burnscars_builder(
    task: str,
    pretrained: bool = False,
    weights_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    in_channels: int = 6,
    num_classes: int = 2,
    img_size: int = 224,
    patch_size: int = 16,
    embed_dim: int = 1024,
    depth: int = 24,
    num_heads: int = 16,
    mlp_ratio: float = 4.0,
    select_indices: Sequence[int] = (5, 11, 17, 23),
    decoder_channels: Sequence[int] = (512, 256, 128, 64),
    drop_path: float = 0.0,
    **kwargs,
) -> nn.Module:
    """Build the burn-scar model; ``pretrained=True`` loads the official checkpoint (1.3 GB download).

    ``weights_path`` loads a local copy of ``Prithvi_EO_V2_300M_BurnScars.pt`` instead of downloading.
    The architecture arguments exist for small test configurations; the official weights need the
    defaults.
    """
    _ = kwargs
    if task.lower() != "segmentation":
        raise ValueError(f"prithvi_burnscars supports task='segmentation', got {task!r}.")
    arch = dict(
        in_channels=in_channels,
        num_classes=num_classes,
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        select_indices=tuple(select_indices),
        decoder_channels=tuple(decoder_channels),
    )
    load = pretrained or weights_path is not None
    if load and arch != _OFFICIAL:
        changed = sorted(k for k in arch if arch[k] != _OFFICIAL[k])
        raise ValueError(f"The official burn-scar weights need the default architecture; changed: {changed}")
    encoder = PrithviViT(
        img_size=img_size,
        patch_size=(1, patch_size, patch_size),
        num_frames=1,
        in_chans=in_channels,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_path=drop_path,
    )
    model = PrithviSegmentation(
        encoder,
        select_indices=select_indices,
        decoder_channels=decoder_channels,
        num_classes=num_classes,
    )
    if load:
        state = load_pretrained_state_dict("prithvi_eo_v2_300_burnscars", weights_path=weights_path, cache_dir=cache_dir)
        model.load_state_dict(finetuned_state_dict_to_model(state), strict=True)
    return model


__all__ = [
    "BURN_SCARS_BANDS",
    "BURN_SCARS_CLASSES",
    "BURN_SCARS_MEAN",
    "BURN_SCARS_STD",
    "PrithviSegmentation",
    "prithvi_burnscars_builder",
]
