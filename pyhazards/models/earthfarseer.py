"""Earthfarseer: local CNN and global Fourier-transformer spatio-temporal forecaster.

Architecture paper: Wu, Liang, Xiong, Zhou, Huang, Wang and Wang, "Earthfarseer: Versatile
Spatio-Temporal Dynamical Systems Modeling in One Model", AAAI 2024 (arXiv:2312.08403; the AAAI
proceedings spell it "Earthfarsser"). Wildfire usage: the Sim2Real-Fire benchmark (Li et al.,
NeurIPS 2024 Datasets and Benchmarks) reports it as a fire-forecasting baseline.

The official code (github.com/easylearningscores/EarthFarseer, now Alexander-wu/EarthFarseer) has
no license, so this module is written from the paper and from permissively licensed parts; the
official code is used only as a test oracle (tests/oracle/test_earthfarseer_oracle.py).
Module and parameter names match the official ``Earthfarseer_model`` so its state dicts load with
``strict=True``, and modules are created in the same order, so the same seed gives the same
initial weights.

Attribution:

- ``GroupConv2d`` and ``Inception`` (the SimVP Inception block) follow ``gInception_ST`` and
  ``GroupConv2d`` in chengtan9907/OpenSTL ``openstl/modules/simvp_modules.py`` at commit
  ``eecf8a3078f0a178dbc7b28723da20f94ce36985`` (Apache-2.0, Copyright the OpenSTL authors).
- The convolutional encoder/decoder, the SimVP translator (``MidXnet``) and the skip branch are
  written from the SimVP paper (Gao et al., CVPR 2022), which Earthfarseer builds on.
- The adaptive Fourier neural operator block is written from the AFNO paper (Guibas et al., ICLR
  2022) and the Earthfarseer paper (Sec. "Spatial block FoTF"); ``trunc_normal_`` is the
  timm-compatible helper of :mod:`pyhazards.models.swin_blocks`.

Behaviour of the official code that is kept (it changes outputs or parameter counts):

- the FoTF spatial block applies the *same* global Fourier transformer (``gf_block``) and local
  branch (``lc_block``) seven times (once, then twice per interaction for three interactions);
- the transformer MLP computes ``fc1`` and replaces ``fc2`` by an average pool over groups of
  ``mlp_ratio`` hidden features (``fc3``); ``fc2`` exists, is initialised and counted, but is
  never used;
- in the second spectral layer of the Fourier operator the imaginary part is computed from the
  *updated* real part;
- the model output is ``decoder(TeDev(encoder(FoTF(x))))`` plus a full SimVP network
  (``skip_conneciton``, spelling kept) whose sizes are fixed (16 / 256 / 4 / 8) whatever the
  model's ``hid_S``, ``hid_T``, ``N_S`` and ``N_T``;
- a second convolutional encoder (``enc``) is built and counted but never used in the forward
  pass;
- the output has as many frames as the input; the paper's temporal projection to another length
  is not in the released code.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_blocks import to_2tuple, trunc_normal_

# Patch size of the global Fourier transformer; its transposed-convolution head upsamples by
# 2 * 2 * 4 = 16, so the two must agree.
GF_PATCH_SIZE = 16

# The skip branch is a SimVP with the constructor defaults of the official ``skip_connection``.
SKIP_HID_S, SKIP_HID_T, SKIP_N_S, SKIP_N_T = 16, 256, 4, 8


def _layer_norm(dim: int) -> nn.LayerNorm:
    return nn.LayerNorm(dim, eps=1e-6)


# --------------------------------------------------------------------------- SimVP parts


def stride_generator(n: int, reverse: bool = False) -> List[int]:
    """Strides of the SimVP encoder: 1, 2, 1, 2, ... (``n`` layers), optionally reversed."""
    strides = [1, 2] * 10
    return list(reversed(strides[:n])) if reverse else strides[:n]


class BasicConv2d(nn.Module):
    """3x3 convolution (transposed for 2x upsampling), GroupNorm(2) and LeakyReLU(0.2)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        transpose: bool = False,
        act_norm: bool = False,
    ):
        super().__init__()
        self.act_norm = act_norm
        if transpose:
            self.conv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                output_padding=stride // 2,
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        return self.act(self.norm(y)) if self.act_norm else y


class ConvSC(nn.Module):
    def __init__(self, C_in: int, C_out: int, stride: int, transpose: bool = False, act_norm: bool = True):
        super().__init__()
        self.conv = BasicConv2d(
            C_in, C_out, kernel_size=3, stride=stride, padding=1, transpose=transpose and stride != 1,
            act_norm=act_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class GroupConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        groups: int,
        act_norm: bool = False,
    ):
        super().__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups
        )
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        return self.activate(self.norm(y)) if self.act_norm else y


class Inception(nn.Module):
    """1x1 reduction, then the sum of grouped convolutions with several kernel sizes."""

    def __init__(self, C_in: int, C_hid: int, C_out: int, incep_ker: Sequence[int] = (3, 5, 7, 11), groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        self.layers = nn.Sequential(
            *[
                GroupConv2d(C_hid, C_out, kernel_size=k, stride=1, padding=k // 2, groups=groups, act_norm=True)
                for k in incep_ker
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y = y + layer(x)
        return y


class Encoder(nn.Module):
    """``N_S`` ConvSC layers (strides 1, 2, 1, 2, ...); returns the latent and the first features."""

    def __init__(self, C_in: int, C_hid: int, N_S: int):
        super().__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]), *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        enc1 = self.enc[0](x)
        latent = enc1
        for layer in list(self.enc)[1:]:
            latent = layer(latent)
        return latent, enc1


class Decoder(nn.Module):
    """Mirror of :class:`Encoder`; the last layer also sees the first encoder features."""

    def __init__(self, C_hid: int, C_out: int, N_S: int):
        super().__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2 * C_hid, C_hid, stride=strides[-1], transpose=True),
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid: torch.Tensor, enc1: torch.Tensor) -> torch.Tensor:
        for layer in list(self.dec)[:-1]:
            hid = layer(hid)
        y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        return self.readout(y)


def _inception_stacks(channel_in: int, channel_hid: int, N_T: int, incep_ker, groups) -> Tuple[list, list]:
    """Encoder and decoder Inception layers of the SimVP translator (U-shaped, with skips)."""
    kw = dict(incep_ker=incep_ker, groups=groups)
    enc = [Inception(channel_in, channel_hid // 2, channel_hid, **kw)]
    enc += [Inception(channel_hid, channel_hid // 2, channel_hid, **kw) for _ in range(1, N_T)]
    dec = [Inception(channel_hid, channel_hid // 2, channel_hid, **kw)]
    dec += [Inception(2 * channel_hid, channel_hid // 2, channel_hid, **kw) for _ in range(1, N_T - 1)]
    dec.append(Inception(2 * channel_hid, channel_hid // 2, channel_in, **kw))
    return enc, dec


def _run_translator(enc: nn.Sequential, dec: nn.Sequential, z: torch.Tensor, N_T: int, middle=None) -> torch.Tensor:
    skips = []
    for i in range(N_T):
        z = enc[i](z)
        if i < N_T - 1:
            skips.append(z)
    if middle is not None:
        z = middle(z)
    z = dec[0](z)
    for i in range(1, N_T):
        z = dec[i](torch.cat([z, skips[-i]], dim=1))
    return z


class MidXnet(nn.Module):
    """SimVP translator: frames stacked as channels through U-shaped Inception layers."""

    def __init__(self, channel_in: int, channel_hid: int, N_T: int, incep_ker=(3, 5, 7, 11), groups: int = 8):
        super().__init__()
        self.N_T = N_T
        enc, dec = _inception_stacks(channel_in, channel_hid, N_T, incep_ker, groups)
        self.enc = nn.Sequential(*enc)
        self.dec = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        z = _run_translator(self.enc, self.dec, x.reshape(b, t * c, h, w), self.N_T)
        return z.reshape(b, t, c, h, w)


class SkipConnection(nn.Module):
    """The SimVP network the official model adds to its prediction (fixed sizes)."""

    def __init__(self, shape_in: Sequence[int], incep_ker=(3, 5, 7, 11), groups: int = 8):
        super().__init__()
        t, c, _, _ = shape_in
        self.enc = Encoder(c, SKIP_HID_S, SKIP_N_S)
        self.hid = MidXnet(t * SKIP_HID_S, SKIP_HID_T, SKIP_N_T, incep_ker, groups)
        self.dec = Decoder(SKIP_HID_S, c, SKIP_N_S)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        embed, skip = self.enc(x.reshape(b * t, c, h, w))
        _, c_, h_, w_ = embed.shape
        hid = self.hid(embed.reshape(b, t, c_, h_, w_)).reshape(b * t, c_, h_, w_)
        return self.dec(hid, skip).reshape(b, t, c, h, w)


# --------------------------------------------------------------------------- Fourier blocks


class Mlp(nn.Module):
    """Linear expansion followed by an average pool back to ``out_features`` (``fc2`` is unused)."""

    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None, act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)  # kept for state-dict compatibility
        self.fc3 = nn.AdaptiveAvgPool1d(out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc3(x))


class AdativeFourierNeuralOperator(nn.Module):
    """Token mixing in the 2D Fourier domain with a two-layer block-diagonal complex MLP.

    Tokens ``(B, h * w, C)`` are reshaped to the ``h x w`` grid, transformed with a real 2D FFT,
    mixed per frequency by two complex linear layers (``num_blocks`` diagonal blocks, ReLU in
    between), transformed back, and added to a 1x1-convolution bias path. (Class name spelled as
    in the reference.)
    """

    def __init__(self, dim: int, h: int = 14, w: int = 14, is_fno_bias: bool = True):
        super().__init__()
        self.hidden_size = dim
        self.h = h
        self.w = w
        self.num_blocks = 2
        if dim % self.num_blocks:
            raise ValueError(f"Fourier operator width {dim} must be divisible by {self.num_blocks}.")
        self.block_size = dim // self.num_blocks
        self.scale = 0.02
        nb, bs = self.num_blocks, self.block_size
        self.w1 = nn.Parameter(self.scale * torch.randn(2, nb, bs, bs))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, nb, bs))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, nb, bs, bs))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, nb, bs))
        self.relu = nn.ReLU()
        self.is_fno_bias = is_fno_bias
        self.bias = nn.Conv1d(dim, dim, 1) if is_fno_bias else None
        self.softshrink = 0.0

    @staticmethod
    def multiply(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...bd,bdk->...bk", x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        if self.bias is not None:
            bias = self.bias(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            bias = torch.zeros_like(x)

        x = torch.fft.rfft2(x.reshape(b, self.h, self.w, c), dim=(1, 2), norm="ortho")
        x = x.reshape(b, x.shape[1], x.shape[2], self.num_blocks, self.block_size)
        re, im = x.real, x.imag
        w1, b1, w2, b2 = self.w1, self.b1, self.w2, self.b2
        hid_re = F.relu(self.multiply(re, w1[0]) - self.multiply(im, w1[1]) + b1[0])
        hid_im = F.relu(self.multiply(re, w1[1]) + self.multiply(im, w1[0]) + b1[1])
        out_re = self.multiply(hid_re, w2[0]) - self.multiply(hid_im, w2[1]) + b2[0]
        # Reference behaviour: the imaginary part uses the already updated real part.
        out_im = self.multiply(out_re, w2[1]) + self.multiply(hid_im, w2[0]) + b2[1]

        x = torch.stack([out_re, out_im], dim=-1)
        if self.softshrink:
            x = F.softshrink(x, lambd=self.softshrink)
        x = torch.view_as_complex(x).reshape(b, x.shape[1], x.shape[2], c)
        x = torch.fft.irfft2(x, s=(self.h, self.w), dim=(1, 2), norm="ortho")
        return x.reshape(b, n, c) + bias


class FourierNetBlock(nn.Module):
    """Pre-norm Fourier token mixing and pre-norm MLP, each with a residual connection."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0, act_layer=nn.GELU, norm_layer=_layer_norm, h: int = 14, w: int = 14):
        super().__init__()
        self.normlayer1 = norm_layer(dim)
        self.filter = AdativeFourierNeuralOperator(dim, h=h, w=w)
        self.drop_path = nn.Identity()  # every reference drop-path rate is 0
        self.normlayer2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.double_skip = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.filter(self.normlayer1(x)))
        return x + self.drop_path(self.mlp(self.normlayer2(x)))


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size: int = 16, in_c: int = 1, embed_dim: int = 768):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.projection = nn.Conv2d(in_c, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.projection(x).flatten(2).transpose(1, 2))


class GFBlock(nn.Module):
    """Global Fourier transformer: patchify each frame, ``depth`` Fourier blocks, unpatchify.

    Unpatchifying uses three transposed convolutions (x2, x2, x4) with tanh in between.
    """

    def __init__(self, img_size, in_channels: int, out_channels: int, input_frames: int, embed_dim: int = 768, depth: int = 12, mlp_ratio: float = 4.0, drop_rate: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = input_frames
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=GF_PATCH_SIZE, in_c=in_channels, embed_dim=embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.h, self.w = self.patch_embed.grid_size
        self.blocks = nn.ModuleList(
            [
                FourierNetBlock(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, h=self.h, w=self.w)
                for _ in range(depth)
            ]
        )
        self.norm = _layer_norm(embed_dim)
        self.linearprojection = nn.Sequential(
            OrderedDict(
                [
                    ("transposeconv1", nn.ConvTranspose2d(embed_dim, out_channels * 16, kernel_size=(2, 2), stride=(2, 2))),
                    ("act1", nn.Tanh()),
                    ("transposeconv2", nn.ConvTranspose2d(out_channels * 16, out_channels * 4, kernel_size=(2, 2), stride=(2, 2))),
                    ("act2", nn.Tanh()),
                    ("transposeconv3", nn.ConvTranspose2d(out_channels * 4, out_channels, kernel_size=(4, 4), stride=(4, 4))),
                ]
            )
        )
        self.final_dropout = nn.Identity()
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = self.pos_drop(self.patch_embed(x.reshape(b * t, c, h, w)) + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x).transpose(1, 2).reshape(-1, self.embed_dim, self.h, self.w)
        x = self.linearprojection(self.final_dropout(x))
        return x.reshape(b, t, c, h, w)


class LocalCNNBranch(nn.Module):
    """Per-frame 1x1 transposed convolution (the released local branch)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = self.upconv(x.reshape(b * t, c, h, w))
        return x.reshape(b, t, self.out_channel, x.shape[2], x.shape[3])


class FoTF(nn.Module):
    """Spatial block: the global and local branches exchange information ``num_interactions`` times."""

    def __init__(self, shape_in: Sequence[int], num_interactions: int = 3, embed_dim: int = 768, depth: int = 12):
        super().__init__()
        t, c, h, w = shape_in
        self.lc_block = LocalCNNBranch(in_channels=c, out_channels=c)
        self.gf_block = GFBlock(img_size=(h, w), in_channels=c, out_channels=c, input_frames=t, embed_dim=embed_dim, depth=depth)
        self.up = nn.ConvTranspose2d(c, c, kernel_size=3, stride=1, padding=1)
        self.down = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(c, c, kernel_size=1)
        self.num_interactions = num_interactions

    def _framewise(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        return layer(x.reshape(b * t, c, h, w)).reshape(b, t, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gf, lc = self.gf_block(x), self.lc_block(x)
        for _ in range(self.num_interactions):
            for exchange in (self.up, self.down):
                combined = self._framewise(exchange, gf) + self._framewise(self.conv1x1, lc)
                gf, lc = self.gf_block(combined), self.lc_block(combined)
        return gf + lc


class TeDev(nn.Module):
    """Temporal block: SimVP Inception translator with 12 Fourier blocks at its bottleneck."""

    def __init__(self, channel_in: int, channel_hid: int, N_T: int, h: int, w: int, incep_ker=(3, 5, 7, 11), groups: int = 8, num_blocks: int = 12):
        super().__init__()
        self.N_T = N_T
        enc, dec = _inception_stacks(channel_in, channel_hid, N_T, incep_ker, groups)
        # Registration order (norm, enc, blocks, dec) and creation order (enc, dec, blocks) follow
        # the reference, so state-dict order and seeded initialisation match.
        self.norm = _layer_norm(channel_hid)
        self.enc = nn.Sequential(*enc)
        self.h, self.w = h, w
        self.blocks = nn.ModuleList(
            [FourierNetBlock(dim=channel_hid, mlp_ratio=4, h=h, w=w) for _ in range(num_blocks)]
        )
        self.dec = nn.Sequential(*dec)

    def _spectral(self, z: torch.Tensor) -> torch.Tensor:
        b, d, h, w = z.shape
        z = z.permute(0, 2, 3, 1).reshape(b, h * w, d)
        for block in self.blocks:
            z = block(z)
        return self.norm(z).permute(0, 2, 1).reshape(b, d, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        z = _run_translator(self.enc, self.dec, x.reshape(b, t * c, h, w), self.N_T, middle=self._spectral)
        return z.reshape(b, t, c, h, w) + x


def _check_shape_in(shape_in: Sequence[int], N_S: int, N_T: int) -> Tuple[int, int, int, int, int, int]:
    if len(shape_in) != 4:
        raise ValueError(f"shape_in must be (frames, channels, height, width), got {tuple(shape_in)}.")
    t, c, h, w = (int(v) for v in shape_in)
    if min(t, c, h, w) <= 0:
        raise ValueError(f"shape_in entries must be positive, got {tuple(shape_in)}.")
    if N_S < 1 or N_T < 2:
        raise ValueError(f"Earthfarseer needs N_S >= 1 and N_T >= 2, got N_S={N_S}, N_T={N_T}.")
    if h % GF_PATCH_SIZE or w % GF_PATCH_SIZE:
        raise ValueError(
            f"Earthfarseer needs height and width divisible by {GF_PATCH_SIZE} (FoTF patch size), got shape_in {tuple(shape_in)}."
        )
    scale = 2 ** stride_generator(N_S).count(2)
    skip_scale = 2 ** stride_generator(SKIP_N_S).count(2)
    for side in (h, w):
        if side % scale or side % skip_scale:
            raise ValueError(
                f"Earthfarseer with N_S={N_S} needs height and width divisible by {max(scale, skip_scale)}, got {(h, w)}."
            )
    return t, c, h, w, h // scale, w // scale


class Earthfarseer(nn.Module):
    """Earthfarseer (``Earthfarseer_model``): ``(B, T, C, H, W)`` -> ``(B, T, C, H, W)``.

    ``shape_in = (T, C, H, W)`` fixes the input size, as in the reference. The output frames are
    the forecast for the ``T`` frames that follow the input. ``gf_embed_dim``, ``gf_depth`` and
    ``num_interactions`` are fixed in the reference (768, 12, 3) and exposed only to build
    smaller models.
    """

    def __init__(
        self,
        shape_in: Sequence[int] = (10, 1, 64, 64),
        hid_S: int = 512,
        hid_T: int = 256,
        N_S: int = 4,
        N_T: int = 8,
        incep_ker: Sequence[int] = (3, 5, 7, 11),
        groups: int = 8,
        gf_embed_dim: int = 768,
        gf_depth: int = 12,
        num_interactions: int = 3,
    ):
        super().__init__()
        t, c, h, w, h1, w1 = _check_shape_in(shape_in, N_S, N_T)
        self.shape_in = (t, c, h, w)
        incep_ker = list(incep_ker)
        # Token grid of the TeDev Fourier blocks. The reference computes it with a formula that
        # adds 1 when H is divisible by 3; that only differs from the true latent size in
        # configurations where the reference fails (see the model card).
        self.H1, self.W1 = h1, w1

        self.fotf_encoder = FoTF(shape_in=self.shape_in, num_interactions=num_interactions, embed_dim=gf_embed_dim, depth=gf_depth)
        self.skip_conneciton = SkipConnection(shape_in=self.shape_in)
        self.latent_projection = Encoder(c, hid_S, N_S)
        self.enc = Encoder(c, hid_S, N_S)  # built and counted by the reference, unused in forward
        self.TeDev_block = TeDev(t * hid_S, hid_T, N_T, self.H1, self.W1, incep_ker, groups)
        self.dec = Decoder(hid_S, c, N_S)

    def _check_input(self, x: torch.Tensor) -> None:
        if x.ndim != 5 or tuple(x.shape[1:]) != self.shape_in:
            raise ValueError(
                "Earthfarseer expects input shape (batch, {}, {}, {}, {}) = (batch, time, channels, height, "
                "width), got {}.".format(*self.shape_in, tuple(x.shape))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        b, t, c, h, w = x.shape
        skip_feature = self.skip_conneciton(x)
        spatial = self.fotf_encoder(x).reshape(b * t, c, h, w)
        embed, enc1 = self.latent_projection(spatial)
        _, c_, h_, w_ = embed.shape
        hidden = self.TeDev_block(embed.reshape(b, t, c_, h_, w_)).reshape(b * t, c_, h_, w_)
        return self.dec(hidden, enc1).reshape(b, t, c, h, w) + skip_feature


class EarthfarseerSegmenter(Earthfarseer):
    """PyHazards adaptation for next-step masks: ``(B, T, C, H, W)`` -> logits ``(B, out, H, W)``.

    Runs :class:`Earthfarseer` and maps the channels of the first predicted frame to
    ``out_channels`` logits with a 1x1 convolution (``segmentation_head``). The Earthfarseer
    parameters keep their reference names at the top level.
    """

    def __init__(self, shape_in: Sequence[int] = (10, 1, 64, 64), out_channels: int = 1, **kwargs):
        super().__init__(shape_in=shape_in, **kwargs)
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}.")
        self.segmentation_head = nn.Conv2d(self.shape_in[1], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.segmentation_head(super().forward(x)[:, 0])


def earthfarseer_builder(
    task: str,
    in_channels: int = 1,
    history: int = 10,
    img_size: Union[int, Sequence[int]] = 64,
    out_channels: int = 1,
    hid_S: int = 512,
    hid_T: int = 256,
    N_S: int = 4,
    N_T: int = 8,
    incep_ker: Sequence[int] = (3, 5, 7, 11),
    groups: int = 8,
    gf_embed_dim: int = 768,
    gf_depth: int = 12,
    num_interactions: int = 3,
    **kwargs,
) -> nn.Module:
    """Earthfarseer with the official default ``Earthfarseer_model(shape_in=(10, 1, 64, 64))``.

    ``task="forecasting"``: ``(B, history, in_channels, H, W)`` -> the next ``history`` frames.
    ``task="segmentation"``: the same input -> ``(B, out_channels, H, W)`` logits (PyHazards
    adaptation, see :class:`EarthfarseerSegmenter`).
    """
    _ = kwargs
    task = task.lower()
    if task not in {"forecasting", "segmentation"}:
        raise ValueError(f"earthfarseer supports task='forecasting' or 'segmentation', got {task!r}.")
    h, w = to_2tuple(img_size)
    config = dict(
        shape_in=(history, in_channels, h, w), hid_S=hid_S, hid_T=hid_T, N_S=N_S, N_T=N_T, incep_ker=incep_ker,
        groups=groups, gf_embed_dim=gf_embed_dim, gf_depth=gf_depth, num_interactions=num_interactions,
    )
    if task == "segmentation":
        return EarthfarseerSegmenter(out_channels=out_channels, **config)
    return Earthfarseer(**config)


__all__ = ["Earthfarseer", "EarthfarseerSegmenter", "earthfarseer_builder"]
