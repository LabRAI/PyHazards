"""DeepLabV3 with a ResNet encoder, equivalent to ``segmentation_models_pytorch.DeepLabV3``.

Architecture paper: Chen, Papandreou, Schroff & Adam, "Rethinking Atrous Convolution for Semantic
Image Segmentation", arXiv:1706.05587 (2017).

Shadrin et al., "Wildfire spreading prediction using multimodal data and deep neural network
approach", Scientific Reports 14:2606 (2024), compare U-Net, U-Net++, MA-Net and DeepLabV3 with
"encoder backbone ResNet18, number of stages used in encoder equals to 3, ... default values for
the other hyperparameters": the argument names and defaults of segmentation_models_pytorch (smp),
whose ``micro-imagewise`` metric reduction the paper also names. No code was released. This module
re-implements ``smp.DeepLabV3`` in plain PyTorch with smp's parameter names, creation order and
initialisation, so smp state dicts load with ``strict=True`` and seeds give identical weights.

Ported from segmentation_models_pytorch 0.3.2 (MIT License, Copyright (c) 2019 Pavel Iakubovskii):
``decoders/deeplabv3/model.py``, ``base/heads.py``, ``encoders/_base.py`` and ``encoders/_utils.py``
(the ResNet encoder is shared with ``resnet_unet.py``). smp's ``decoders/deeplabv3/decoder.py``
(ASPP and decoder) is itself taken from torchvision (BSD 3-Clause License, Copyright (c) Soumith
Chintala 2016). The DeepLabV3 code path is identical in smp 0.2.0 through 0.3.4 (0.3.x only adds the
input-size check); 0.3.3 was the current release when the wildfire paper was submitted.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet_unet import RESNET_LAYERS, ResNetEncoder


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class ASPPPooling(nn.Sequential):
    """Image-level features: global average pooling, 1x1 conv-BN-ReLU, bilinear up-sampling."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for module in self:
            x = module(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """Atrous spatial pyramid pooling: a 1x1 branch, three 3x3 atrous branches and image pooling."""

    def __init__(self, in_channels: int, out_channels: int, atrous_rates: Sequence[int]):
        super().__init__()
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                ),
                ASPPConv(in_channels, out_channels, rate1),
                ASPPConv(in_channels, out_channels, rate2),
                ASPPConv(in_channels, out_channels, rate3),
                ASPPPooling(in_channels, out_channels),
            ]
        )
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([conv(x) for conv in self.convs], dim=1))


class DeepLabV3Decoder(nn.Sequential):
    """ASPP on the deepest encoder feature, then a 3x3 conv-BN-ReLU (smp / torchvision ``DeepLabHead``)."""

    def __init__(self, in_channels: int, out_channels: int = 256, atrous_rates: Sequence[int] = (12, 24, 36)):
        super().__init__(
            ASPP(in_channels, out_channels, atrous_rates),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.out_channels = out_channels

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        return super().forward(features[-1])


class DeepLabV3(nn.Module):
    """``smp.DeepLabV3`` with a ResNet-18/34 encoder: ``(batch, C, H, W)`` -> ``(batch, classes, H, W)`` logits.

    The defaults are smp's (ResNet-34, five encoder stages) except ``encoder_weights``, which is
    ``None`` here and ``"imagenet"`` in smp. The encoder runs at output stride 8 (strides of
    ``layer3``/``layer4`` replaced by dilation 2/4); height and width must be divisible by
    ``min(8, 2 ** encoder_depth)``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        classes: int = 1,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = None,
        decoder_channels: int = 256,
        upsampling: int = 8,
    ):
        super().__init__()
        if encoder_name not in RESNET_LAYERS:
            raise ValueError(f"encoder_name must be one of {sorted(RESNET_LAYERS)}, got {encoder_name!r}")
        if not 3 <= encoder_depth <= 5:
            raise ValueError(f"encoder_depth must be in [3, 5] for DeepLabV3, got {encoder_depth}")
        if classes <= 0 or decoder_channels <= 0 or upsampling <= 0:
            raise ValueError(
                f"classes, decoder_channels and upsampling must be positive, got {classes}, "
                f"{decoder_channels}, {upsampling}"
            )
        if encoder_weights not in (None, "imagenet"):
            raise ValueError(f"encoder_weights must be None or 'imagenet', got {encoder_weights!r}")
        self.encoder = ResNetEncoder(encoder_name, in_channels=3, depth=encoder_depth)
        if encoder_weights is not None:
            self.encoder.load_imagenet_weights()
        self.encoder.set_in_channels(in_channels, pretrained=encoder_weights is not None)
        self.encoder.make_dilated(8)
        self.in_channels = int(in_channels)
        self.decoder = DeepLabV3Decoder(self.encoder.out_channels[-1], decoder_channels)
        # smp's SegmentationHead: 1x1 conv, UpsamplingBilinear2d (align_corners=True), activation.
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(decoder_channels, classes, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity(),
            nn.Identity(),
        )
        # smp 0.3.x never calls initialize() for DeepLabV3, so the decoder and head keep PyTorch's
        # default initialisation (unlike smp's U-Net); nothing is re-initialised here either.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"DeepLabV3 expects input shape (batch, channels, height, width), got {tuple(x.shape)}.")
        if x.size(1) != self.in_channels:
            raise ValueError(f"DeepLabV3 expected {self.in_channels} input channels, got {x.size(1)}.")
        h, w = x.shape[-2:]
        stride = self.encoder.output_stride
        if h % stride or w % stride:
            raise ValueError(f"DeepLabV3 needs height and width divisible by {stride}, got {(h, w)}.")
        features = self.encoder(x)
        return self.segmentation_head(self.decoder(*features))


def deeplabv3_builder(
    task: str,
    in_channels: int = 58,
    out_channels: int = 1,
    encoder_name: str = "resnet18",
    encoder_depth: int = 3,
    encoder_weights: Optional[str] = None,
    decoder_channels: int = 256,
    upsampling: int = 8,
    **kwargs,
) -> nn.Module:
    """DeepLabV3 in Shadrin et al.'s configuration by default (ResNet-18, three encoder stages)."""
    _ = kwargs
    if task.lower() != "segmentation":
        raise ValueError(f"deeplabv3 supports task='segmentation', got {task!r}.")
    return DeepLabV3(
        in_channels=in_channels,
        classes=out_channels,
        encoder_name=encoder_name,
        encoder_depth=encoder_depth,
        encoder_weights=encoder_weights,
        decoder_channels=decoder_channels,
        upsampling=upsampling,
    )


__all__ = ["ASPP", "DeepLabV3", "DeepLabV3Decoder", "deeplabv3_builder"]
