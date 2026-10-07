"""The WildfireSpreadTS benchmark baselines with the benchmark's own configurations.

Gerard, Zhao & Sullivan, "WildfireSpreadTS: A dataset of multi-modal time series for wildfire
spread prediction", NeurIPS 2023 Datasets & Benchmarks (https://github.com/SebastianGer/WildfireSpreadTS,
MIT). The benchmark trains four learned baselines on next-day active-fire segmentation; this
builder returns each of them exactly as configured in ``src/models``:

- ``logistic_regression``: ``Conv2d(C * T, 1, 3, padding=1)`` (361 parameters for 40 channels, 1 day);
- ``resnet18_unet``: ``smp.Unet("resnet18", encoder_weights=None)`` on the flattened time axis;
- ``convlstm``: one ConvLSTM layer, 64 hidden channels, 3x3 kernels, head on the last cell state
  (240,449 parameters for 40 channels);
- ``utae``: U-TAE with encoder widths (64, 64, 64, 128), decoder widths (32, 32, 64, 128),
  output convolutions (32, 1), 16 heads, d_model 256, d_k 4 (1,099,011 parameters for 40 channels).

With all features, WildfireSpreadTS has 40 channels per day (23 raw features with the 17-class
land cover one-hot encoded) and uses 1 or 5 days of history.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch.nn as nn

from .convlstm import ConvLSTMSegmenter
from .logistic_regression import PixelLogisticRegression
from .resnet_unet import ResNetUNet
from .utae import UTAE

WILDFIRESPREADTS_BASELINES = ("logistic_regression", "resnet18_unet", "convlstm", "utae")
# Static per-day features of the 40-channel "All features" layout (topography and the 17 land-cover
# one-hot channels); FireSpreadDataset.get_static_and_dynamic_feature_ids in WildfireSpreadTS.
WILDFIRESPREADTS_STATIC_FEATURE_IDS = (12, 13, 14, *range(16, 33))


def wildfirespreadts_builder(
    task: str,
    baseline: str = "utae",
    in_channels: int = 40,
    history: int = 5,
    remove_duplicate_features: Optional[bool] = None,
    static_feature_ids: Sequence[int] = WILDFIRESPREADTS_STATIC_FEATURE_IDS,
    **kwargs,
) -> nn.Module:
    """Build one WildfireSpreadTS baseline.

    ``in_channels`` is the number of features per day; ``history`` is the number of days, used
    by the baselines that flatten time into channels (logistic regression and the U-Net). Every
    baseline takes input shape ``(batch, history, in_channels, height, width)`` and returns
    ``(batch, 1, height, width)`` logits; ``utae`` also accepts ``batch_positions`` (day of year).

    ``remove_duplicate_features`` (``resnet18_unet`` only; on by default for the 40-feature layout,
    as in the benchmark's U-Net runs) keeps the static features listed in ``static_feature_ids``
    for the last day only, so a 5-day, 40-feature U-Net has 4 * 20 + 40 = 120 input channels
    instead of 200.
    """
    _ = kwargs
    if task.lower() != "segmentation":
        raise ValueError(f"wildfirespreadts supports task='segmentation', got {task!r}.")
    if in_channels <= 0 or history <= 0:
        raise ValueError(f"in_channels and history must be positive, got {in_channels} and {history}.")
    if baseline == "logistic_regression":
        return PixelLogisticRegression(in_channels * history, out_channels=1, kernel_size=3)
    if baseline == "resnet18_unet":
        if remove_duplicate_features is None:
            # On by default only for the benchmark's own 40-feature layout, where static_feature_ids apply.
            remove_duplicate_features = in_channels == 40
        static = tuple(static_feature_ids) if remove_duplicate_features and history > 1 else None
        if static is not None and (not static or max(static) >= in_channels):
            raise ValueError(f"static_feature_ids must index the {in_channels} input channels, got {static}.")
        channels = in_channels * history if static is None else (history - 1) * (in_channels - len(static)) + in_channels
        return ResNetUNet(
            channels, classes=1, encoder_name="resnet18", encoder_weights=None, static_feature_ids=static
        )
    if baseline == "convlstm":
        return ConvLSTMSegmenter(in_channels, num_classes=1, hidden_dim=64, kernel_size=3, num_layers=1)
    if baseline == "utae":
        return UTAE(
            input_dim=in_channels,
            encoder_widths=[64, 64, 64, 128],
            decoder_widths=[32, 32, 64, 128],
            out_conv=[32, 1],
            str_conv_k=4,
            str_conv_s=2,
            str_conv_p=1,
            agg_mode="att_group",
            encoder_norm="group",
            n_head=16,
            d_model=256,
            d_k=4,
            pad_value=0,
            padding_mode="reflect",
        )
    raise ValueError(f"baseline must be one of {WILDFIRESPREADTS_BASELINES}, got {baseline!r}.")


__all__ = ["WILDFIRESPREADTS_BASELINES", "wildfirespreadts_builder"]
