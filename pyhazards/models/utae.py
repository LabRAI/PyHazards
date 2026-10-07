"""U-TAE: U-Net with a Lightweight Temporal Attention Encoder (L-TAE).

Port of Sainte Fare Garnot & Landrieu, "Panoptic Segmentation of Satellite Image Time Series
with Convolutional Temporal Attention Networks" (ICCV 2021), from the reference implementation
``VSainteuf/utae-paps`` (``src/backbones/utae.py``, ``ltae.py``, ``positional_encoding.py``;
MIT License, Copyright (c) 2021 Vivien Sainte Fare Garnot). WildfireSpreadTS (Gerard et al.,
NeurIPS 2023 D&B) vendors the same code unchanged for its UTAE baseline.

Parameter names match the reference modules, so official checkpoints (for example the PASTIS
weights on Zenodo) load with ``strict=True``. Behaviour that the reference code has and this
port keeps on purpose:

- the output convolution block ends in BatchNorm + ReLU, so the returned scores are
  non-negative;
- input frames that equal ``pad_value`` everywhere are treated as temporal padding.
"""

from __future__ import annotations

import copy
import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    """Sinusoidal encoding of acquisition dates (e.g. day of year), repeated once per head."""

    def __init__(self, d: int, T: int = 1000, repeat: Optional[int] = None, offset: int = 0):
        super().__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        denom = torch.pow(T, 2 * (torch.arange(offset, offset + d).float() // 2) / d)
        self.register_buffer("denom", denom, persistent=False)

    def forward(self, batch_positions: torch.Tensor) -> torch.Tensor:
        table = batch_positions[:, :, None] / self.denom[None, None, :]
        table[:, :, 0::2] = torch.sin(table[:, :, 0::2])
        table[:, :, 1::2] = torch.cos(table[:, :, 1::2])
        if self.repeat is not None:
            table = torch.cat([table for _ in range(self.repeat)], dim=-1)
        return table


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature: float, attn_dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, pad_mask=None):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2)) / self.temperature
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        attn = self.dropout(self.softmax(attn))
        return torch.matmul(attn, v), attn


class MultiHeadAttention(nn.Module):
    """L-TAE attention: one learned master query per head, keys from a linear layer."""

    def __init__(self, n_head: int, d_k: int, d_in: int):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in
        self.Q = nn.Parameter(torch.zeros((n_head, d_k)))
        nn.init.normal_(self.Q, mean=0, std=math.sqrt(2.0 / d_k))
        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=math.sqrt(2.0 / d_k))
        self.attention = ScaledDotProductAttention(temperature=math.pow(d_k, 0.5))

    def forward(self, v: torch.Tensor, pad_mask: Optional[torch.Tensor] = None):
        d_k, n_head = self.d_k, self.n_head
        sz_b, seq_len, _ = v.size()
        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(-1, d_k)
        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)
        if pad_mask is not None:
            pad_mask = pad_mask.repeat((n_head, 1))
        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(n_head * sz_b, seq_len, -1)
        output, attn = self.attention(q, k, v, pad_mask=pad_mask)
        attn = attn.view(n_head, sz_b, 1, seq_len).squeeze(dim=2)
        output = output.view(n_head, sz_b, 1, self.d_in // n_head).squeeze(dim=2)
        return output, attn


class LTAE2d(nn.Module):
    """Lightweight Temporal Attention Encoder applied independently at every pixel."""

    def __init__(
        self,
        in_channels: int = 128,
        n_head: int = 16,
        d_k: int = 4,
        mlp: Sequence[int] = (256, 128),
        dropout: float = 0.2,
        d_model: Optional[int] = 256,
        T: int = 1000,
        return_att: bool = False,
        positional_encoding: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        mlp = list(copy.deepcopy(mlp))
        self.return_att = return_att
        self.n_head = n_head
        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None
        if mlp[0] != self.d_model:
            raise ValueError(f"mlp[0] must equal d_model ({self.d_model}), got {mlp[0]}")
        self.positional_encoder = (
            PositionalEncoder(self.d_model // n_head, T=T, repeat=n_head) if positional_encoding else None
        )
        self.attention_heads = MultiHeadAttention(n_head=n_head, d_k=d_k, d_in=self.d_model)
        self.in_norm = nn.GroupNorm(num_groups=n_head, num_channels=self.in_channels)
        self.out_norm = nn.GroupNorm(num_groups=n_head, num_channels=mlp[-1])
        layers: List[nn.Module] = []
        for i in range(len(mlp) - 1):
            layers.extend([nn.Linear(mlp[i], mlp[i + 1]), nn.BatchNorm1d(mlp[i + 1]), nn.ReLU()])
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, batch_positions: Optional[torch.Tensor] = None, pad_mask: Optional[torch.Tensor] = None):
        sz_b, seq_len, d, h, w = x.shape
        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(-1).repeat((1, 1, h)).unsqueeze(-1).repeat((1, 1, 1, w))
            pad_mask = pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)
        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)
        if self.positional_encoder is not None:
            if batch_positions is None:
                raise ValueError("LTAE2d with positional encoding needs batch_positions of shape (batch, time).")
            bp = batch_positions.unsqueeze(-1).repeat((1, 1, h)).unsqueeze(-1).repeat((1, 1, 1, w))
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            out = out + self.positional_encoder(bp)

        out, attn = self.attention_heads(out, pad_mask=pad_mask)
        out = out.permute(1, 0, 2).contiguous().view(sz_b * h * w, -1)
        out = self.dropout(self.mlp(out))
        out = self.out_norm(out) if self.out_norm is not None else out
        out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)
        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(0, 1, 4, 2, 3)
        if self.return_att:
            return out, attn
        return out


class TemporallySharedBlock(nn.Module):
    """Applies a 2D block to every frame of a ``(B, T, C, H, W)`` sequence, skipping padded frames."""

    def __init__(self, pad_value: Optional[float] = None):
        super().__init__()
        self.pad_value = pad_value

    def smart_forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return self.forward(x)
        b, t, c, h, w = x.shape
        out = x.view(b * t, c, h, w)
        if self.pad_value is not None:
            pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
            if pad_mask.any():
                # Padded frames bypass the block and are filled with pad_value. (The reference code
                # also runs the block on an all-zero copy of the batch to read the output shape; with
                # the default GroupNorm that extra pass changes nothing, so it is skipped here.)
                if (~pad_mask).any():
                    valid = self.forward(out[~pad_mask])
                    temp = valid.new_full((b * t, *valid.shape[1:]), float(self.pad_value))
                    temp[~pad_mask] = valid
                else:
                    with torch.no_grad():
                        out_shape = self.forward(out[:1]).shape[1:]
                    temp = out.new_full((b * t, *out_shape), float(self.pad_value))
                out = temp
            else:
                out = self.forward(out)
        else:
            out = self.forward(out)
        _, c, h, w = out.shape
        return out.view(b, t, c, h, w)


class ConvLayer(nn.Module):
    def __init__(
        self,
        nkernels: Sequence[int],
        norm: str = "batch",
        k: int = 3,
        s: int = 1,
        p: int = 1,
        n_groups: int = 4,
        last_relu: bool = True,
        padding_mode: str = "reflect",
    ):
        super().__init__()
        if norm == "batch":
            norm_layer = nn.BatchNorm2d
        elif norm == "instance":
            norm_layer = nn.InstanceNorm2d
        elif norm == "group":
            def norm_layer(num_feats: int) -> nn.Module:
                return nn.GroupNorm(num_channels=num_feats, num_groups=n_groups)
        else:
            norm_layer = None
        layers: List[nn.Module] = []
        for i in range(len(nkernels) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode=padding_mode,
                )
            )
            if norm_layer is not None:
                layers.append(norm_layer(nkernels[i + 1]))
            if last_relu or i < len(nkernels) - 2:
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConvBlock(TemporallySharedBlock):
    def __init__(self, nkernels, pad_value=None, norm="batch", last_relu=True, padding_mode="reflect"):
        super().__init__(pad_value=pad_value)
        self.conv = ConvLayer(nkernels=nkernels, norm=norm, last_relu=last_relu, padding_mode=padding_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownConvBlock(TemporallySharedBlock):
    def __init__(self, d_in, d_out, k, s, p, pad_value=None, norm="batch", padding_mode="reflect"):
        super().__init__(pad_value=pad_value)
        self.down = ConvLayer(nkernels=[d_in, d_in], norm=norm, k=k, s=s, p=p, padding_mode=padding_mode)
        self.conv1 = ConvLayer(nkernels=[d_in, d_out], norm=norm, padding_mode=padding_mode)
        self.conv2 = ConvLayer(nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.down(x)
        out = self.conv1(out)
        return out + self.conv2(out)


class UpConvBlock(nn.Module):
    def __init__(self, d_in, d_out, k, s, p, norm="batch", d_skip=None, padding_mode="reflect"):
        super().__init__()
        d = d_out if d_skip is None else d_skip
        self.skip_conv = nn.Sequential(nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1), nn.BatchNorm2d(d), nn.ReLU())
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=d_in, out_channels=d_out, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(d_out),
            nn.ReLU(),
        )
        self.conv1 = ConvLayer(nkernels=[d_out + d, d_out], norm=norm, padding_mode=padding_mode)
        self.conv2 = ConvLayer(nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        out = self.up(x)
        out = torch.cat([out, self.skip_conv(skip)], dim=1)
        out = self.conv1(out)
        return out + self.conv2(out)


class TemporalAggregator(nn.Module):
    """Collapses the time axis of skip connections using the L-TAE attention masks."""

    def __init__(self, mode: str = "mean"):
        super().__init__()
        if mode not in {"att_group", "att_mean", "mean"}:
            raise ValueError(f"agg_mode must be 'att_group', 'att_mean' or 'mean', got {mode!r}")
        self.mode = mode

    def forward(self, x, pad_mask=None, attn_mask=None):
        padded = pad_mask is not None and bool(pad_mask.any())
        if self.mode == "att_group":
            n_heads, b, t, h, w = attn_mask.shape
            attn = attn_mask.view(n_heads * b, t, h, w)
            if x.shape[-2] > w:
                attn = nn.functional.interpolate(attn, size=x.shape[-2:], mode="bilinear", align_corners=False)
            else:
                attn = nn.functional.avg_pool2d(attn, kernel_size=w // x.shape[-2])
            attn = attn.view(n_heads, b, t, *x.shape[-2:])
            if padded:
                attn = attn * (~pad_mask).float()[None, :, :, None, None]
            out = torch.stack(x.chunk(n_heads, dim=2))
            out = (attn[:, :, :, None, :, :] * out).sum(dim=2)
            return torch.cat([group for group in out], dim=1)
        if self.mode == "att_mean":
            attn = attn_mask.mean(dim=0)
            attn = nn.functional.interpolate(attn, size=x.shape[-2:], mode="bilinear", align_corners=False)
            if padded:
                attn = attn * (~pad_mask).float()[:, :, None, None]
            return (x * attn[:, :, None, :, :]).sum(dim=1)
        if padded:
            out = x * (~pad_mask).float()[:, :, None, None, None]
            return out.sum(dim=1) / (~pad_mask).sum(dim=1)[:, None, None, None]
        return x.mean(dim=1)


class UTAE(nn.Module):
    """U-TAE for satellite image time series segmentation.

    Input ``(batch, time, channels, height, width)`` plus ``batch_positions`` ``(batch, time)``
    holding the acquisition dates (day of year in WildfireSpreadTS and PASTIS). When
    ``batch_positions`` is omitted, ``0..T-1`` is used. Height and width must be divisible by
    ``str_conv_s ** (len(encoder_widths) - 1)``.
    """

    def __init__(
        self,
        input_dim: int,
        encoder_widths: Sequence[int] = (64, 64, 64, 128),
        decoder_widths: Optional[Sequence[int]] = (32, 32, 64, 128),
        out_conv: Sequence[int] = (32, 20),
        str_conv_k: int = 4,
        str_conv_s: int = 2,
        str_conv_p: int = 1,
        agg_mode: str = "att_group",
        encoder_norm: str = "group",
        n_head: int = 16,
        d_model: int = 256,
        d_k: int = 4,
        encoder: bool = False,
        return_maps: bool = False,
        pad_value: float = 0,
        padding_mode: str = "reflect",
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        encoder_widths = list(encoder_widths)
        decoder_widths = list(decoder_widths) if decoder_widths is not None else list(encoder_widths)
        if len(encoder_widths) != len(decoder_widths) or encoder_widths[-1] != decoder_widths[-1]:
            raise ValueError("decoder_widths must match encoder_widths in length and in the last entry.")
        self.input_dim = int(input_dim)
        self.n_stages = len(encoder_widths)
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.enc_dim = decoder_widths[0]
        self.stack_dim = sum(decoder_widths)
        self.pad_value = pad_value
        self.encoder = encoder
        self.return_maps = return_maps or encoder
        self.downsample_factor = str_conv_s ** (self.n_stages - 1)

        self.in_conv = ConvBlock(
            nkernels=[input_dim, encoder_widths[0], encoder_widths[0]],
            pad_value=pad_value,
            norm=encoder_norm,
            padding_mode=padding_mode,
        )
        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=encoder_widths[i],
                d_out=encoder_widths[i + 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                pad_value=pad_value,
                norm=encoder_norm,
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1)
        )
        self.up_blocks = nn.ModuleList(
            UpConvBlock(
                d_in=decoder_widths[i],
                d_out=decoder_widths[i - 1],
                d_skip=encoder_widths[i - 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                norm="batch",
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1, 0, -1)
        )
        self.temporal_encoder = LTAE2d(
            in_channels=encoder_widths[-1],
            d_model=d_model,
            n_head=n_head,
            mlp=[d_model, encoder_widths[-1]],
            return_att=True,
            d_k=d_k,
        )
        self.temporal_aggregator = TemporalAggregator(mode=agg_mode)
        self.out_conv = ConvBlock(nkernels=[decoder_widths[0]] + list(out_conv), padding_mode=padding_mode)

    def forward(self, x: torch.Tensor, batch_positions: Optional[torch.Tensor] = None, return_att: bool = False):
        if x.ndim != 5:
            raise ValueError(
                "UTAE expects input shape (batch, time, channels, height, width), "
                f"got {tuple(x.shape)}."
            )
        if x.size(2) != self.input_dim:
            raise ValueError(f"UTAE expected {self.input_dim} channels, got {x.size(2)}.")
        if x.size(-2) % self.downsample_factor or x.size(-1) % self.downsample_factor:
            raise ValueError(
                f"UTAE needs height and width divisible by {self.downsample_factor}, got {tuple(x.shape[-2:])}."
            )
        if batch_positions is None:
            batch_positions = torch.arange(x.size(1), device=x.device, dtype=x.dtype).expand(x.size(0), -1)

        pad_mask = (x == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        out = self.in_conv.smart_forward(x)
        feature_maps = [out]
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(feature_maps[-1])
            feature_maps.append(out)
        out, att = self.temporal_encoder(feature_maps[-1], batch_positions=batch_positions, pad_mask=pad_mask)
        maps = [out]
        for i in range(self.n_stages - 1):
            skip = self.temporal_aggregator(feature_maps[-(i + 2)], pad_mask=pad_mask, attn_mask=att)
            out = self.up_blocks[i](out, skip)
            maps.append(out)
        if self.encoder:
            return out, maps
        out = self.out_conv(out)
        if return_att:
            return out, att
        if self.return_maps:
            return out, maps
        return out


def utae_builder(
    task: str,
    in_channels: int,
    out_channels: int = 1,
    encoder_widths: Sequence[int] = (64, 64, 64, 128),
    decoder_widths: Sequence[int] = (32, 32, 64, 128),
    out_conv_hidden: int = 32,
    agg_mode: str = "att_group",
    encoder_norm: str = "group",
    n_head: int = 16,
    d_model: int = 256,
    d_k: int = 4,
    pad_value: float = 0,
    padding_mode: str = "reflect",
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() != "segmentation":
        raise ValueError(f"utae supports task='segmentation', got {task!r}.")
    return UTAE(
        input_dim=in_channels,
        encoder_widths=encoder_widths,
        decoder_widths=decoder_widths,
        out_conv=[out_conv_hidden, out_channels],
        agg_mode=agg_mode,
        encoder_norm=encoder_norm,
        n_head=n_head,
        d_model=d_model,
        d_k=d_k,
        pad_value=pad_value,
        padding_mode=padding_mode,
    )


__all__ = ["LTAE2d", "PositionalEncoder", "UTAE", "utae_builder"]
