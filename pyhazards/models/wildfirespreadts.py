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

import torch.nn as nn

from .convlstm import ConvLSTMSegmenter
from .logistic_regression import PixelLogisticRegression
from .resnet_unet import ResNetUNet
from .utae import UTAE

WILDFIRESPREADTS_BASELINES = ("logistic_regression", "resnet18_unet", "convlstm", "utae")


def wildfirespreadts_builder(
    task: str,
    baseline: str = "utae",
    in_channels: int = 40,
    history: int = 5,
    **kwargs,
) -> nn.Module:
    """Build one WildfireSpreadTS baseline.

    ``in_channels`` is the number of features per day; ``history`` is the number of days, used
    by the baselines that flatten time into channels (logistic regression and the U-Net). Every
    baseline takes input shape ``(batch, history, in_channels, height, width)`` and returns
    ``(batch, 1, height, width)`` logits; ``utae`` also accepts ``batch_positions`` (day of year).
    """
    _ = kwargs
    if task.lower() != "segmentation":
        raise ValueError(f"wildfirespreadts supports task='segmentation', got {task!r}.")
    if in_channels <= 0 or history <= 0:
        raise ValueError(f"in_channels and history must be positive, got {in_channels} and {history}.")
    if baseline == "logistic_regression":
        return PixelLogisticRegression(in_channels * history, out_channels=1, kernel_size=3)
    if baseline == "resnet18_unet":
        return ResNetUNet(in_channels * history, classes=1, encoder_name="resnet18", encoder_weights=None)
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
