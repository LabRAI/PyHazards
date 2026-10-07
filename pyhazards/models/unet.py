"""U-Net (Ronneberger, Fischer & Brox, MICCAI 2015), rebuilt from the paper.

Ronneberger, Fischer & Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation",
MICCAI 2015, LNCS 9351, pp. 234-241, arXiv:1505.04597 (Fig. 1 and Sections 2-3).

The authors released only a Caffe/MATLAB tarball without a license (``u-net-release-2015-10-02``),
so nothing is copied from it: the network is written from the paper. The released network
definition (``phseg_v5-train.prototxt``) and its trained weights are used only as a test-time
oracle (tests/oracle/test_unet_oracle.py), and they settle details the paper leaves open:

* module names follow the Caffe layer names (``conv_d0a-b`` -> ``conv_d0a_b``, ...), so the
  released weights map onto this module one to one. ``d0``-``d4`` are the five levels of the
  contracting path, ``u3``-``u0`` the four levels of the expansive path, and the letters
  ``a``-``d`` the successive feature maps within a level;
* the up-sampled map comes first in each concatenation, followed by the cropped skip feature;
* the crop is centred (the release's crop layer aligns the two maps by their receptive fields);
* the two dropout layers (p = 0.5, training only) follow the last ReLU of the 512- and
  1024-channel levels; the first is applied in place, so the skip connection of the 512-channel
  level carries it too;
* initialisation: the release's weight filler draws from N(0, 2 / n) with n = weights per output
  unit (``count / num`` of the Caffe blob), and biases start at zero. For the 2x2 up-convolutions
  this is n = 4 * out_channels, which is what ``kaiming_normal_(mode="fan_in")`` computes for
  ``ConvTranspose2d``.

Architecture (paper Fig. 1): four 2x2 max-pool down-sampling steps and four 2x2 up-convolutions,
64-128-256-512-1024 channels, two unpadded 3x3 convolutions with ReLU per level, centre-cropped
skip connections, and a final 1x1 convolution. With the paper's valid convolutions the output is
smaller than the input (572 -> 388); ``unet_output_size`` and ``unet_input_size`` give the sizes.
``padding="same"`` zero-pads the 3x3 convolutions so the output keeps the input size, which suits
fixed-size wildfire tiles but is not the paper's network.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

UNET_CHANNELS = (64, 128, 256, 512, 1024)
UNET_VARIANTS = ("paper", "phseg_v5")
_LEVELS = 4  # down-sampling steps in the paper's network


def _valid_size_index(size: int) -> int:
    """Return ``k`` with ``size == 16 * k + 60`` (valid-padding tile sizes), or raise ``ValueError``."""
    k, remainder = divmod(int(size) - 60, 16)
    if remainder == 0 and k >= 8:
        return k
    above = 16 * max(k + 1, 8) + 60
    nearest = f"{16 * k + 60} or {above}" if k >= 8 else str(above)
    raise ValueError(
        f"U-Net with valid convolutions needs height and width of the form 16 * k + 60 with k >= 8 "
        f"(188, 204, ..., 572, ...) so that every 2x2 max-pooling sees an even size; got {size} "
        f"(nearest valid: {nearest})."
    )


def unet_output_size(input_size: int, padding: str = "valid") -> int:
    """Output height/width of the U-Net for an input height/width (572 -> 388 for valid padding)."""
    if padding == "same":
        if input_size <= 0 or input_size % (2**_LEVELS):
            raise ValueError(f"U-Net with padding='same' needs sizes divisible by 16, got {input_size}.")
        return int(input_size)
    if padding != "valid":
        raise ValueError(f"padding must be 'valid' or 'same', got {padding!r}")
    return 16 * _valid_size_index(input_size) - 124


def unet_input_size(output_size: int) -> int:
    """Smallest valid-padding input size whose output covers ``output_size`` pixels.

    This is the tile size rule of the authors' overlap-tile script: the input is
    ``16 * k + 60`` and the output ``16 * k - 124`` for the smallest ``k`` that is large enough.
    """
    if output_size <= 0:
        raise ValueError(f"output_size must be positive, got {output_size}")
    k = max(8, math.ceil((int(output_size) + 124) / 16))
    return 16 * k + 60


def _center_crop(skip: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    top = (skip.size(-2) - size[0]) // 2
    left = (skip.size(-1) - size[1]) // 2
    return skip[..., top : top + size[0], left : left + size[1]]


class UNet(nn.Module):
    """Ronneberger et al.'s U-Net: ``(batch, in_channels, H, W)`` -> ``(batch, out_channels, H', W')`` logits.

    ``padding="valid"`` (the paper) gives ``H' = unet_output_size(H)``; ``padding="same"`` keeps
    ``H' = H`` and needs ``H`` and ``W`` divisible by 16. ``upconv_relu`` and
    ``last_upconv_channels`` reproduce the released Caffe network (``phseg_v5``), which applies a
    ReLU after every up-convolution and gives its last up-convolution 128 output channels.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        channels: Sequence[int] = UNET_CHANNELS,
        padding: str = "valid",
        dropout: float = 0.5,
        upconv_relu: bool = False,
        last_upconv_channels: Optional[int] = None,
    ):
        super().__init__()
        channels = tuple(int(c) for c in channels)
        if len(channels) != _LEVELS + 1 or min(channels) <= 0:
            raise ValueError(f"channels needs five positive widths (paper: {UNET_CHANNELS}), got {channels}")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(f"in_channels and out_channels must be positive, got {in_channels}, {out_channels}")
        if padding not in ("valid", "same"):
            raise ValueError(f"padding must be 'valid' or 'same', got {padding!r}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        c0, c1, c2, c3, c4 = channels
        u0 = c0 if last_upconv_channels is None else int(last_upconv_channels)
        if u0 <= 0:
            raise ValueError(f"last_upconv_channels must be positive, got {last_upconv_channels}")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.channels = channels
        self.padding = padding
        self.upconv_relu = bool(upconv_relu)
        pad = 1 if padding == "same" else 0

        def conv(cin: int, cout: int) -> nn.Conv2d:
            return nn.Conv2d(cin, cout, kernel_size=3, padding=pad)

        def upconv(cin: int, cout: int) -> nn.ConvTranspose2d:
            return nn.ConvTranspose2d(cin, cout, kernel_size=2, stride=2)

        # Contracting path (layer order of the released network definition).
        self.conv_d0a_b = conv(in_channels, c0)
        self.conv_d0b_c = conv(c0, c0)
        self.conv_d1a_b = conv(c0, c1)
        self.conv_d1b_c = conv(c1, c1)
        self.conv_d2a_b = conv(c1, c2)
        self.conv_d2b_c = conv(c2, c2)
        self.conv_d3a_b = conv(c2, c3)
        self.conv_d3b_c = conv(c3, c3)
        self.dropout_d3c = nn.Dropout(dropout)
        self.conv_d4a_b = conv(c3, c4)
        self.conv_d4b_c = conv(c4, c4)
        self.dropout_d4c = nn.Dropout(dropout)
        # Expansive path: up-convolution, concatenation [up-sampled, cropped skip], two 3x3 convolutions.
        self.upconv_d4c_u3a = upconv(c4, c3)
        self.conv_u3b_c = conv(c3 + c3, c3)
        self.conv_u3c_d = conv(c3, c3)
        self.upconv_u3d_u2a = upconv(c3, c2)
        self.conv_u2b_c = conv(c2 + c2, c2)
        self.conv_u2c_d = conv(c2, c2)
        self.upconv_u2d_u1a = upconv(c2, c1)
        self.conv_u1b_c = conv(c1 + c1, c1)
        self.conv_u1c_d = conv(c1, c1)
        self.upconv_u1d_u0a = upconv(c1, u0)
        self.conv_u0b_c = conv(u0 + c0, c0)
        self.conv_u0c_d = conv(c0, c0)
        self.conv_u0d_score = nn.Conv2d(c0, out_channels, kernel_size=1)

        # Paper Section 3: Gaussian weights with standard deviation sqrt(2 / N), N = inputs per unit.
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def output_size(self, input_size: int) -> int:
        """Output height/width for an input height/width under this model's padding."""
        return unet_output_size(input_size, self.padding)

    def _up(self, upconv: nn.ConvTranspose2d, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = upconv(x)
        if self.upconv_relu:
            x = F.relu(x)
        return torch.cat([x, _center_crop(skip, x.shape[-2:])], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"UNet expects input shape (batch, channels, height, width), got {tuple(x.shape)}.")
        if x.size(1) != self.in_channels:
            raise ValueError(f"UNet expected {self.in_channels} input channels, got {x.size(1)}.")
        for size in x.shape[-2:]:
            self.output_size(int(size))  # raises ValueError for sizes the network cannot take

        d0c = F.relu(self.conv_d0b_c(F.relu(self.conv_d0a_b(x))))
        d1c = F.relu(self.conv_d1b_c(F.relu(self.conv_d1a_b(F.max_pool2d(d0c, 2)))))
        d2c = F.relu(self.conv_d2b_c(F.relu(self.conv_d2a_b(F.max_pool2d(d1c, 2)))))
        d3c = self.dropout_d3c(F.relu(self.conv_d3b_c(F.relu(self.conv_d3a_b(F.max_pool2d(d2c, 2))))))
        d4c = self.dropout_d4c(F.relu(self.conv_d4b_c(F.relu(self.conv_d4a_b(F.max_pool2d(d3c, 2))))))

        u3d = F.relu(self.conv_u3c_d(F.relu(self.conv_u3b_c(self._up(self.upconv_d4c_u3a, d4c, d3c)))))
        u2d = F.relu(self.conv_u2c_d(F.relu(self.conv_u2b_c(self._up(self.upconv_u3d_u2a, u3d, d2c)))))
        u1d = F.relu(self.conv_u1c_d(F.relu(self.conv_u1b_c(self._up(self.upconv_u2d_u1a, u2d, d1c)))))
        u0d = F.relu(self.conv_u0c_d(F.relu(self.conv_u0b_c(self._up(self.upconv_u1d_u0a, u1d, d0c)))))
        return self.conv_u0d_score(u0d)


def unet_builder(
    task: str,
    in_channels: int = 1,
    out_channels: int = 2,
    padding: str = "valid",
    dropout: float = 0.5,
    variant: str = "paper",
    channels: Sequence[int] = UNET_CHANNELS,
    **kwargs,
) -> nn.Module:
    """Build the U-Net. ``variant="phseg_v5"`` gives the authors' released Caffe network."""
    _ = kwargs
    if task.lower() != "segmentation":
        raise ValueError(f"unet supports task='segmentation', got {task!r}.")
    if variant not in UNET_VARIANTS:
        raise ValueError(f"variant must be one of {UNET_VARIANTS}, got {variant!r}")
    release = variant == "phseg_v5"
    return UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        padding=padding,
        dropout=dropout,
        upconv_relu=release,
        last_upconv_channels=tuple(channels)[1] if release else None,
    )


__all__ = ["UNET_CHANNELS", "UNET_VARIANTS", "UNet", "unet_builder", "unet_input_size", "unet_output_size"]
