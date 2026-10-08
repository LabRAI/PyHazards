"""The TS-SatFire next-day prediction baselines with the benchmark's own configurations.

Zhao, Gerard & Ban, "TS-SatFire: A Multi-Task Satellite Image Time-Series Dataset for Wildfire
Detection and Prediction", Scientific Data 12:1817 (2025), doi:10.1038/s41597-025-06271-3,
arXiv:2412.11555; code https://github.com/zhaoyutim/TS-SatFire (no license; used only as a test
oracle). TS-SatFire is a dataset paper: its prediction-task baselines are four spatio-temporal
MONAI networks, built in ``run_spatial_temp_model_pred.py`` at commit ``da75732``. This builder
returns each of them as configured there (the architectures are the PyHazards ports
:mod:`~pyhazards.models.unet3d`, :mod:`~pyhazards.models.attention_unet`,
:mod:`~pyhazards.models.unetr` and :mod:`~pyhazards.models.swin_unetr`):

- ``unet3d``: MONAI U-Net, 3D, channels (64, 128, 256, 512, 1024), stride (1, 2, 2) at every level
  (31,712,970 parameters with 43 channels; Table 3: 31.7M);
- ``attention_unet3d``: MONAI Attention U-Net with the same channels and strides (94,555,506;
  Table 3: 94.5M);
- ``unetr3d``: UNETR with img_size (T, 256, 256), patch (1, 16, 16), (1, 2, 2) up-sampling, a
  (1, 3, 3) ``decoder3`` kernel, feature size 16, hidden 384, MLP 1536, batch norm (28,816,866 at
  T = 6; see the card for the 34.8M of Table 3);
- ``swinunetr3d`` (default; the model behind the paper's Tables 4 and 5): SwinUNETR with patch
  (1, 2, 2), window (T, 4, 4), TS-SatFire's patch merging, ``attn_version="v1"``, feature size 36,
  ``num_heads`` heads in every stage (3 by default, not published), batch norm (33,191,942 at
  T = 6; Table 3: 33.2M).

The input is ``(batch, T, 43, 256, 256)``: 27 bands with the 17-class land cover one-hot encoded
in place (21 + 17 + 5 channels, ``FireDataset.preprocess``), T in {2, 4, 6} days. Every model runs
on ``(batch, 43, T, H, W)`` and, as in the prediction script (``outputs.mean(2)``), the two-class
logits are averaged over time to ``(batch, 2, H, W)``.
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn

from .attention_unet import TemporalAttentionUnet
from .swin_unetr import TemporalSwinUNETR
from .unet3d import TS_SATFIRE_3D_STRIDES, TS_SATFIRE_CHANNELS, TemporalUNet
from .unetr import (
    TS_SATFIRE_UNETR3D_DECODER3_KERNEL,
    TS_SATFIRE_UNETR3D_PATCH,
    TS_SATFIRE_UNETR3D_UP,
    TemporalUNETR,
)

TS_SATFIRE_BASELINES = ("unet3d", "attention_unet3d", "unetr3d", "swinunetr3d")
# 27 raw bands; land cover (index 21) one-hot encoded into 17 channels.
TS_SATFIRE_PRED_CHANNELS = 43
# run_spatial_temp_model_pred.py: hidden/MLP sizes of the default and the "v0" UNETR.
TS_SATFIRE_UNETR_WIDTHS = {"default": (384, 1536), "v0": (768, 3072)}


def ts_satfire_builder(
    task: str,
    baseline: str = "swinunetr3d",
    in_channels: int = TS_SATFIRE_PRED_CHANNELS,
    history: int = 6,
    image_size: int = 256,
    out_channels: int = 2,
    feature_size: Optional[int] = None,
    num_heads: int = 3,
    unetr_version: str = "default",
    time_reduction: str = "mean",
    **kwargs,
) -> nn.Module:
    """Build one TS-SatFire prediction baseline.

    ``history`` is the number of input days T and ``image_size`` the tile size (256 in
    TS-SatFire); UNETR and SwinUNETR are built for exactly ``(history, image_size, image_size)``.
    Every baseline takes ``(batch, history, in_channels, image_size, image_size)`` and returns
    ``(batch, out_channels, image_size, image_size)`` logits (``time_reduction="none"`` keeps the
    time axis). ``feature_size`` overrides the UNETR (16) or SwinUNETR (36) feature size,
    ``num_heads`` is SwinUNETR's heads per stage (the ``-nh`` argument of the script) and
    ``unetr_version="v0"`` selects the script's alternative UNETR widths (768 / 3072). Each baseline
    checks its input shape and raises ``ValueError`` for any other layout.
    """
    _ = kwargs
    if task.lower() != "segmentation":
        raise ValueError(f"ts_satfire supports task='segmentation', got {task!r}.")
    if in_channels <= 0 or history <= 0 or image_size <= 0:
        raise ValueError(
            f"in_channels, history and image_size must be positive, got {in_channels}, {history} and {image_size}."
        )
    img_size = (history, image_size, image_size)
    if baseline in ("unet3d", "attention_unet3d"):
        network = TemporalUNet if baseline == "unet3d" else TemporalAttentionUnet
        return network(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=TS_SATFIRE_CHANNELS,
            strides=TS_SATFIRE_3D_STRIDES,
            stride_mode="shared",
            time_reduction=time_reduction,
        )
    if baseline == "unetr3d":
        if unetr_version not in TS_SATFIRE_UNETR_WIDTHS:
            raise ValueError(f"unetr_version must be one of {sorted(TS_SATFIRE_UNETR_WIDTHS)}, got {unetr_version!r}.")
        hidden_size, mlp_dim = TS_SATFIRE_UNETR_WIDTHS[unetr_version]
        return TemporalUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            spatial_dims=3,
            norm_name="batch",
            feature_size=16 if feature_size is None else feature_size,
            patch_size=TS_SATFIRE_UNETR3D_PATCH,
            kernel_size_up_down=TS_SATFIRE_UNETR3D_UP,
            decoder3_kernel_size=TS_SATFIRE_UNETR3D_DECODER3_KERNEL,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            time_reduction=time_reduction,
        )
    if baseline == "swinunetr3d":
        return TemporalSwinUNETR(
            img_size=img_size,
            patch_size=(1, 2, 2),
            window_size=(history, 4, 4),
            in_channels=in_channels,
            out_channels=out_channels,
            depths=(2, 2, 2, 2),
            num_heads=(num_heads,) * 4,
            feature_size=36 if feature_size is None else feature_size,
            norm_name="batch",
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            attn_version="v1",
            normalize=True,
            use_checkpoint=False,
            spatial_dims=3,
            downsample="ts_satfire",
            wrap_out=True,
            time_reduction=time_reduction,
        )
    raise ValueError(f"baseline must be one of {TS_SATFIRE_BASELINES}, got {baseline!r}.")


__all__ = ["TS_SATFIRE_BASELINES", "TS_SATFIRE_PRED_CHANNELS", "ts_satfire_builder"]
