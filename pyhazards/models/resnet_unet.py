"""U-Net with a ResNet encoder, equivalent to ``segmentation_models_pytorch.Unet``.

WildfireSpreadTS (Gerard et al., NeurIPS 2023 D&B) uses ``smp.Unet(encoder_name="resnet18",
encoder_weights=None, in_channels=C, classes=1)`` from segmentation_models_pytorch 0.3.2
(MIT, Copyright (c) 2019 Pavel Iakubovskii). This module re-implements that network in plain
PyTorch: the torchvision ResNet encoder without its classifier, the smp U-Net decoder
(nearest-neighbour upsampling, two Conv-BN-ReLU per block, no centre block) and the 3x3
segmentation head, with the same initialisation and the same parameter names, so smp state
dicts load with ``strict=True``.

Architecture papers: U-Net (Ronneberger et al., MICCAI 2015) and ResNet (He et al., CVPR 2016).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# ImageNet weights used by smp 0.3.2 (torchvision checkpoints, BSD-3-Clause).
RESNET_IMAGENET_URLS = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
}
RESNET_LAYERS = {
    "resnet18": (2, 2, 2, 2),
    "resnet34": (3, 4, 6, 3),
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNetEncoder(nn.Module):
    """torchvision ResNet (BasicBlock variants) without ``avgpool``/``fc``, returning multi-scale features."""

    def __init__(self, name: str = "resnet18", in_channels: int = 3, depth: int = 5):
        super().__init__()
        if name not in RESNET_LAYERS:
            raise ValueError(f"encoder_name must be one of {sorted(RESNET_LAYERS)}, got {name!r}")
        if not 1 <= depth <= 5:
            raise ValueError(f"encoder_depth must be in [1, 5], got {depth}")
        self.name = name
        self.depth = depth
        self.in_channels = 3
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        layers = RESNET_LAYERS[name]
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        # torchvision builds its 1000-way classifier here and smp deletes it afterwards. Creating and
        # dropping it keeps the random-number stream, and therefore seeded initialisation, identical.
        nn.Linear(512, 1000)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.out_channels = [3, 64, 64, 128, 256, 512][: depth + 1]
        self.set_in_channels(in_channels, pretrained=False)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers.extend(BasicBlock(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def set_in_channels(self, in_channels: int, pretrained: bool) -> None:
        """Adapt the stem to ``in_channels`` the way smp does (``patch_first_conv``)."""
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if in_channels == 3:
            return
        weight = self.conv1.weight.detach()
        self.conv1.in_channels = in_channels
        if not pretrained:
            self.conv1.weight = nn.Parameter(torch.empty(64, in_channels, *self.conv1.kernel_size))
            self.conv1.reset_parameters()
        elif in_channels == 1:
            self.conv1.weight = nn.Parameter(weight.sum(1, keepdim=True))
        else:
            new_weight = torch.empty(64, in_channels, *self.conv1.kernel_size)
            for i in range(in_channels):
                new_weight[:, i] = weight[:, i % 3]
            self.conv1.weight = nn.Parameter(new_weight * (3 / in_channels))
        self.in_channels = in_channels
        self.out_channels[0] = in_channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        stages = [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]
        features = []
        for i in range(self.depth + 1):
            x = stages[i](x)
            features.append(x)
        return features


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, use_batchnorm: bool = True):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, use_batchnorm)
        self.conv2 = Conv2dReLU(out_channels, out_channels, use_batchnorm)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class UnetDecoder(nn.Module):
    def __init__(self, encoder_channels: Sequence[int], decoder_channels: Sequence[int], use_batchnorm: bool = True):
        super().__init__()
        encoder_channels = list(encoder_channels[1:])[::-1]
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        self.center = nn.Identity()
        self.blocks = nn.ModuleList(
            DecoderBlock(in_ch, skip_ch, out_ch, use_batchnorm)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, decoder_channels)
        )

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        features = features[1:][::-1]
        skips = features[1:]
        x = self.center(features[0])
        for i, block in enumerate(self.blocks):
            x = block(x, skips[i] if i < len(skips) else None)
        return x


class ResNetUNet(nn.Module):
    """``smp.Unet`` with a ResNet-18/34 encoder.

    Accepts ``(batch, channels, height, width)``, or ``(batch, time, channels, height, width)``
    which is flattened to ``time * channels`` input channels as WildfireSpreadTS does for its
    multi-day U-Net. Height and width must be divisible by ``2 ** encoder_depth``.
    """

    def __init__(
        self,
        in_channels: int,
        classes: int = 1,
        encoder_name: str = "resnet18",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = None,
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        decoder_use_batchnorm: bool = True,
    ):
        super().__init__()
        if len(decoder_channels) != encoder_depth:
            raise ValueError(
                f"decoder_channels needs {encoder_depth} entries for encoder_depth={encoder_depth}, "
                f"got {len(decoder_channels)}."
            )
        if classes <= 0:
            raise ValueError(f"classes must be positive, got {classes}")
        self.encoder = ResNetEncoder(encoder_name, in_channels=3, depth=encoder_depth)
        if encoder_weights is not None:
            if encoder_weights != "imagenet":
                raise ValueError(f"encoder_weights must be None or 'imagenet', got {encoder_weights!r}")
            state = torch.hub.load_state_dict_from_url(RESNET_IMAGENET_URLS[encoder_name], progress=False)
            state.pop("fc.weight", None)
            state.pop("fc.bias", None)
            self.encoder.load_state_dict(state)
        self.encoder.set_in_channels(in_channels, pretrained=encoder_weights is not None)
        self.in_channels = int(in_channels)
        self.output_stride = 2 ** encoder_depth
        self.decoder = UnetDecoder(self.encoder.out_channels, decoder_channels, use_batchnorm=decoder_use_batchnorm)
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], classes, kernel_size=3, padding=1),
            nn.Identity(),
            nn.Identity(),
        )
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.segmentation_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            x = x.flatten(start_dim=1, end_dim=2)
        if x.ndim != 4:
            raise ValueError(
                "ResNetUNet expects input shape (batch, channels, height, width) or "
                f"(batch, time, channels, height, width), got {tuple(x.shape)}."
            )
        if x.size(1) != self.in_channels:
            raise ValueError(f"ResNetUNet expected {self.in_channels} input channels, got {x.size(1)}.")
        h, w = x.shape[-2:]
        if h % self.output_stride or w % self.output_stride:
            raise ValueError(
                f"ResNetUNet needs height and width divisible by {self.output_stride}, got {(h, w)}."
            )
        features = self.encoder(x)
        return self.segmentation_head(self.decoder(*features))


def resnet18_unet_builder(
    task: str,
    in_channels: int,
    out_channels: int = 1,
    history: int = 1,
    encoder_name: str = "resnet18",
    encoder_weights: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() != "segmentation":
        raise ValueError(f"resnet18_unet supports task='segmentation', got {task!r}.")
    if history <= 0:
        raise ValueError(f"history must be positive, got {history}")
    return ResNetUNet(
        in_channels=in_channels * history,
        classes=out_channels,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
    )


__all__ = ["ResNetEncoder", "ResNetUNet", "resnet18_unet_builder"]
