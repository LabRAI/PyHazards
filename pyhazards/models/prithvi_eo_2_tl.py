"""Prithvi-EO-2.0-TL encoder (time + location embeddings) with the official burn-scar segmentation head.

Prithvi-EO-2.0 (Szwarcman et al., arXiv:2412.02732) is a multi-temporal ViT masked autoencoder pretrained
on 4.2M HLS samples. The TL variants add sin/cos encodings of acquisition time (year, day of year) and
location (latitude, longitude), each multiplied by a learned scale, to the patch tokens. The encoder is
the plain-PyTorch port of the official ``prithvi_mae.py`` in :mod:`pyhazards.models.prithvi`
(Apache-2.0, IBM); ``pretrained=True`` loads the released MAE checkpoint
(``ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL`` or ``-600M-TL``) into it.

PyHazards builders return a task model, so the encoder is wrapped in the segmentation recipe of the
official Prithvi-EO-2.0 burn-scar fine-tune (TerraTorch ``EncoderDecoderFactory``: SelectIndices at
the official per-size indices -> ReshapeTokensToImage -> LearnedInterpolateToPyramidal -> UNetDecoder
[512, 256, 128, 64] -> SegmentationHead). Only the encoder is pretrained; the neck, decoder and head are
randomly initialised and must be fine-tuned. With ``num_frames = T > 1`` the tokens of all frames are
stacked along channels (``effective_time_dim = T``), as in the official multi-temporal configs.

Input shape is ``(B, C, T, H, W)`` (``(B, C, H, W)`` when ``num_frames == 1``), with optional
``temporal_coords`` ``(B, T, 2)`` and ``location_coords`` ``(B, 2)``; other shapes raise ``ValueError``.
The output is ``(B, num_classes, H, W)`` logits. Use ``model.encoder.forward_features`` for the raw
encoder tokens.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch.nn as nn

from .prithvi import (
    PRITHVI_EO_V2_CONFIGS,
    PrithviSegmentation,
    PrithviViT,
    load_pretrained_state_dict,
    mae_state_dict_to_encoder,
)

TL_WEIGHTS = {"300m": "prithvi_eo_v2_300_tl", "600m": "prithvi_eo_v2_600_tl"}


def prithvi_eo_2_tl_builder(
    task: str,
    variant: str = "300m",
    pretrained: bool = False,
    weights_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    in_channels: int = 6,
    num_classes: int = 2,
    num_frames: int = 1,
    img_size: int = 224,
    select_indices: Optional[Sequence[int]] = None,
    decoder_channels: Sequence[int] = (512, 256, 128, 64),
    head_dropout: float = 0.0,
    drop_path: float = 0.0,
    embed_dim: Optional[int] = None,
    depth: Optional[int] = None,
    num_heads: Optional[int] = None,
    patch_size: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """Build a Prithvi-EO-2.0-TL segmentation model.

    ``variant`` is ``"300m"`` (ViT-L/16) or ``"600m"`` (ViT-H/14). ``pretrained=True`` downloads the
    variant's pretraining checkpoint (1.3 GB / 2.6 GB) and loads its encoder weights;
    ``weights_path`` loads a local copy instead. ``embed_dim``, ``depth``, ``num_heads`` and
    ``patch_size`` override the variant for small untrained test configurations.
    """
    _ = kwargs
    if task.lower() != "segmentation":
        raise ValueError(f"prithvi_eo_2_tl supports task='segmentation', got {task!r}.")
    if variant not in PRITHVI_EO_V2_CONFIGS:
        raise ValueError(f"variant must be one of {sorted(PRITHVI_EO_V2_CONFIGS)}, got {variant!r}")
    cfg = dict(PRITHVI_EO_V2_CONFIGS[variant])
    overrides = dict(embed_dim=embed_dim, depth=depth, num_heads=num_heads)
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if patch_size is not None:
        overrides["patch_size"] = (1, patch_size, patch_size)
    load = pretrained or weights_path is not None
    if load and (overrides or in_channels != 6 or img_size != 224):
        raise ValueError(
            "The official TL weights need the variant's architecture with 6 input bands and img_size=224; "
            f"got overrides {sorted(overrides)}, in_channels={in_channels}, img_size={img_size}"
        )
    cfg.update(overrides)
    encoder = PrithviViT(
        img_size=img_size,
        patch_size=cfg["patch_size"],
        num_frames=num_frames,
        in_chans=in_channels,
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        mlp_ratio=4.0,
        coords_encoding=["time", "location"],
        coords_scale_learn=True,
        drop_path=drop_path,
    )
    model = PrithviSegmentation(
        encoder,
        select_indices=cfg["select_indices"] if select_indices is None else select_indices,
        decoder_channels=decoder_channels,
        num_classes=num_classes,
        head_dropout=head_dropout,
    )
    if load:
        state = load_pretrained_state_dict(TL_WEIGHTS[variant], weights_path=weights_path, cache_dir=cache_dir)
        model.encoder.load_state_dict(mae_state_dict_to_encoder(state, model.encoder), strict=True)
    return model


__all__ = ["PrithviSegmentation", "PrithviViT", "prithvi_eo_2_tl_builder"]
