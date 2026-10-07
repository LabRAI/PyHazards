"""SwinLSTM: a recurrent cell built from Swin Transformer blocks (Tang et al., ICCV 2023).

Attribution: ported from SongTang-x/SwinLSTM at commit
``4425bcfefbebfac85c9fc6c6659361b8154d5ca3`` (MIT License, Copyright (c) 2023 Song Tang):
``SwinLSTM_D.py`` (SwinLSTM-D, the deep model with patch merging/expanding), ``SwinLSTM_B.py``
(SwinLSTM-B, one or more cells at a single resolution) and the warm-up/prediction rollout of
``functions.py`` (``model_forward_multi_layer`` / ``model_forward_single_layer``). Window attention,
patch embedding, patch merging and patch expanding are the shared blocks of ``swin_blocks``
(microsoft/Swin-Transformer, MIT; timm 0.4.12 ``trunc_normal_`` and ``DropPath``), which the
reference also uses.

Module and attribute names follow the reference (``Downsample``/``Upsample`` for SwinLSTM-D, ``ST``
for SwinLSTM-B), so the official state dicts, including the released Moving-MNIST SwinLSTM-D
weights, load with ``strict=True``, and modules are created in the reference order so that the same
``torch.manual_seed`` gives identical initial weights.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .swin_blocks import (
    PatchEmbed,
    PatchExpand,
    PatchMerging,
    SwinTransformerBlock,
    stochastic_depth_rates,
    to_2tuple,
)

ImageSize = Union[int, Tuple[int, int]]
CellState = Tuple[torch.Tensor, torch.Tensor]


class SwinLSTMBlock(SwinTransformerBlock):
    """Swin block that can fuse a second token sequence before attention.

    With ``hx`` given, ``norm1(x)`` and ``norm1(hx)`` are concatenated and projected back to
    ``dim`` channels by ``red`` (the "LP" of the paper); the residual branch still starts from ``x``.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 2,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        # Created after the MLP, as in the reference (keeps seeded initialisation identical).
        self.red = nn.Linear(2 * dim, dim)

    def forward(self, x: torch.Tensor, hx: Optional[torch.Tensor] = None) -> torch.Tensor:
        self._check_tokens(x)
        h, w = self.input_resolution
        b, _, c = x.shape

        shortcut = x
        x = self.norm1(x)
        if hx is not None:
            x = self.red(torch.cat((x, self.norm1(hx)), -1))
        x = shortcut + self.drop_path(self._window_attention(x.view(b, h, w, c)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class SwinTransformerBlocks(nn.Module):
    """The Swin blocks of one SwinLSTM cell (``SwinTransformer`` in ``SwinLSTM_D.py``).

    Block 0 fuses the input ``xt`` with the hidden state, odd blocks see only the previous block's
    output, and even blocks after the first fuse the previous output with ``xt`` again.
    ``drop_path`` gives one stochastic-depth rate per block.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        drop_path: Sequence[float],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        if len(drop_path) != depth:
            raise ValueError(f"expected {depth} drop-path rates, got {len(drop_path)}")
        self.layers = nn.ModuleList(
            SwinLSTMBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        )

    def forward(self, xt: torch.Tensor, hx: torch.Tensor) -> torch.Tensor:
        x = xt
        for index, layer in enumerate(self.layers):
            if index == 0:
                x = layer(xt, hx)
            elif index % 2 == 0:
                x = layer(x, xt)
            else:
                x = layer(x, None)
        return x


class SwinLSTMCell(nn.Module):
    """SwinLSTM cell (paper Eq. 5): ``F = STB(LP(x, h))``, ``c' = sigmoid(F) * (c + tanh(F))``,
    ``h' = sigmoid(F) * tanh(c')``. Zero states are used when ``hidden_states`` is None."""

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int,
        depth: int,
        drop_path: Sequence[float],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.Swin = SwinTransformerBlocks(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            norm_layer=norm_layer,
        )

    def forward(
        self, xt: torch.Tensor, hidden_states: Optional[CellState] = None
    ) -> Tuple[torch.Tensor, CellState]:
        if hidden_states is None:
            hx = xt.new_zeros(xt.shape)
            cx = xt.new_zeros(xt.shape)
        else:
            hx, cx = hidden_states
        ft = self.Swin(xt, hx)
        gate = torch.sigmoid(ft)
        cell = torch.tanh(ft)
        cy = gate * (cx + cell)
        hy = gate * torch.tanh(cy)
        return hy, (hy, cy)


class PatchInflated(nn.Module):
    """Reconstruction layer: tokens -> image through a 3x3 ``ConvTranspose2d`` with stride 2.

    The convolution is registered as ``Conv`` (SwinLSTM-D) or ``ConvT`` (SwinLSTM-B), as in the
    reference files.
    """

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        input_resolution: Tuple[int, int],
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 1,
        conv_name: str = "Conv",
    ):
        super().__init__()
        self.input_resolution = tuple(input_resolution)
        self.conv_name = conv_name
        self.add_module(
            conv_name,
            nn.ConvTranspose2d(
                in_channels=embed_dim,
                out_channels=in_chans,
                kernel_size=(3, 3),
                stride=to_2tuple(stride),
                padding=to_2tuple(padding),
                output_padding=to_2tuple(output_padding),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = self.input_resolution
        b, length, c = x.shape
        if length != h * w:
            raise ValueError(f"PatchInflated expected {h * w} tokens, got input of shape {tuple(x.shape)}.")
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)
        return getattr(self, self.conv_name)(x)


class DownSample(nn.Module):
    """SwinLSTM-D encoder: patch embedding, then per level a SwinLSTM cell and patch merging."""

    def __init__(
        self,
        img_size: ImageSize,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        depths_downsample: Sequence[int],
        num_heads: Sequence[int],
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.num_layers = len(depths_downsample)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=nn.LayerNorm
        )
        patches_resolution = self.patch_embed.patches_resolution
        dpr = stochastic_depth_rates(drop_path_rate, depths_downsample)

        self.layers = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for i_layer in range(self.num_layers):
            resolution = (patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer))
            dim = int(embed_dim * 2 ** i_layer)
            # Patch merging is created before the cell, as in the reference.
            downsample = PatchMerging(input_resolution=resolution, dim=dim)
            layer = SwinLSTMCell(
                dim=dim,
                input_resolution=resolution,
                depth=depths_downsample[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths_downsample[:i_layer]) : sum(depths_downsample[: i_layer + 1])],
                norm_layer=norm_layer,
            )
            self.layers.append(layer)
            self.downsample.append(downsample)

    def forward(
        self, x: torch.Tensor, y: Sequence[Optional[CellState]]
    ) -> Tuple[List[CellState], torch.Tensor]:
        x = self.patch_embed(x)
        hidden_states_down = []
        for index, layer in enumerate(self.layers):
            x, hidden_state = layer(x, y[index])
            x = self.downsample[index](x)
            hidden_states_down.append(hidden_state)
        return hidden_states_down, x


class UpSample(nn.Module):
    """SwinLSTM-D decoder: per level a SwinLSTM cell and patch expanding, then reconstruction.

    The reference indexes ``depths_upsample`` and ``num_heads`` from the deepest level upwards and
    reverses the stochastic-depth rates inside each cell (``flag=0``); both are reproduced here.
    ``patch_embed`` is never used in the forward pass but is part of the official state dict.
    """

    def __init__(
        self,
        img_size: ImageSize,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        depths_upsample: Sequence[int],
        num_heads: Sequence[int],
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer=nn.LayerNorm,
        out_chans: Optional[int] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_layers = len(depths_upsample)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        # Unused by the reference forward pass; kept so official checkpoints load strictly.
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=nn.LayerNorm
        )
        patches_resolution = self.patch_embed.patches_resolution
        self.Unembed = PatchInflated(
            in_chans=in_chans if out_chans is None else out_chans,
            embed_dim=embed_dim,
            input_resolution=patches_resolution,
        )
        dpr = stochastic_depth_rates(drop_path_rate, depths_upsample)

        self.layers = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i_layer in range(self.num_layers):
            level = self.num_layers - 1 - i_layer
            scale = 2 ** (self.num_layers - i_layer)
            resolution = (patches_resolution[0] // scale, patches_resolution[1] // scale)
            dim = int(embed_dim * scale)
            upsample = PatchExpand(input_resolution=resolution, dim=dim)
            rates = dpr[sum(depths_upsample[:level]) : sum(depths_upsample[: level + 1])]
            layer = SwinLSTMCell(
                dim=dim,
                input_resolution=resolution,
                depth=depths_upsample[level],
                num_heads=num_heads[level],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=list(reversed(rates)),
                norm_layer=norm_layer,
            )
            self.layers.append(layer)
            self.upsample.append(upsample)

    def forward(
        self, x: torch.Tensor, y: Sequence[Optional[CellState]], apply_sigmoid: bool = True
    ) -> Tuple[List[CellState], torch.Tensor]:
        hidden_states_up = []
        for index, layer in enumerate(self.layers):
            x, hidden_state = layer(x, y[index])
            x = self.upsample[index](x)
            hidden_states_up.append(hidden_state)
        x = self.Unembed(x)
        return hidden_states_up, torch.sigmoid(x) if apply_sigmoid else x


class STconvert(nn.Module):
    """SwinLSTM-B body: patch embedding, SwinLSTM cells at one resolution, reconstruction."""

    def __init__(
        self,
        img_size: ImageSize,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        depths: Sequence[int],
        num_heads: Sequence[int],
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer=nn.LayerNorm,
        out_chans: Optional[int] = None,
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=nn.LayerNorm
        )
        patches_resolution = self.patch_embed.patches_resolution
        self.PatchInflated = PatchInflated(
            in_chans=in_chans if out_chans is None else out_chans,
            embed_dim=embed_dim,
            input_resolution=patches_resolution,
            conv_name="ConvT",
        )
        # SwinLSTM-B uses the same stochastic-depth rate in every block (no linear decay).
        self.layers = nn.ModuleList(
            SwinLSTMCell(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=[drop_path_rate] * depths[i_layer],
                norm_layer=norm_layer,
            )
            for i_layer in range(self.num_layers)
        )

    def forward(
        self, x: torch.Tensor, h: Sequence[Optional[CellState]], apply_sigmoid: bool = True
    ) -> Tuple[List[CellState], torch.Tensor]:
        x = self.patch_embed(x)
        hidden_states = []
        for index, layer in enumerate(self.layers):
            x, hidden_state = layer(x, h[index])
            hidden_states.append(hidden_state)
        x = self.PatchInflated(x)
        return hidden_states, torch.sigmoid(x) if apply_sigmoid else x


def _check_geometry(img_size: Tuple[int, int], patch_size: int, window_size: int, levels: int) -> None:
    """Raise ``ValueError`` unless every token grid the model visits splits into whole windows."""
    if patch_size != 2:
        raise ValueError(
            f"patch_size must be 2, got {patch_size}: the reference reconstruction layer (a stride-2 "
            "transposed convolution) only restores the input size for 2x2 patches."
        )
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if img_size[0] % patch_size or img_size[1] % patch_size:
        raise ValueError(f"img_size {img_size} must be divisible by patch_size {patch_size}.")
    tokens = (img_size[0] // patch_size, img_size[1] // patch_size)
    if tokens[0] % (2 ** levels) or tokens[1] % (2 ** levels):
        raise ValueError(f"img_size {img_size}: the {tokens} token grid cannot be halved {levels} times.")
    for level in range(levels + 1):
        scale = 2 ** level
        grid = (tokens[0] // scale, tokens[1] // scale)
        window = min(window_size, *grid)  # a window never exceeds the grid (Swin rule)
        if grid[0] % window or grid[1] % window:
            raise ValueError(f"img_size {img_size}: the {grid} token grid is not divisible by window size {window}.")


class SwinLSTM(nn.Module):
    """SwinLSTM frame forecaster (``variant="d"``: SwinLSTM-D, ``variant="b"``: SwinLSTM-B).

    ``forward`` takes ``(batch, T_in, in_chans, H, W)`` frames and returns the next
    ``num_output_frames`` frames ``(batch, T_out, out_chans, H, W)`` in ``[0, 1]`` (sigmoid output),
    using the reference rollout: a warm-up phase feeds the first ``T_in - 1`` frames, then the last
    input frame starts an autoregressive prediction phase in which each predicted frame is the next
    input. ``include_warmup=True`` also returns the ``T_in - 1`` warm-up predictions (frames 2 to
    ``T_in``), i.e. the ``T_in - 1 + T_out`` frames the reference loss compares with
    ``cat(inputs[:, 1:], targets)``. ``step`` is the reference single-step ``forward``.
    """

    def __init__(
        self,
        img_size: ImageSize = 64,
        patch_size: int = 2,
        in_chans: int = 1,
        embed_dim: int = 128,
        depths_downsample: Sequence[int] = (2, 6),
        depths_upsample: Sequence[int] = (6, 2),
        num_heads: Sequence[int] = (4, 8),
        window_size: int = 4,
        variant: str = "d",
        depths: Sequence[int] = (12,),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        out_chans: Optional[int] = None,
        num_output_frames: int = 10,
    ):
        super().__init__()
        variant = str(variant).lower()
        if variant not in {"b", "d"}:
            raise ValueError(f"variant must be 'b' (SwinLSTM-B) or 'd' (SwinLSTM-D), got {variant!r}")
        if in_chans <= 0:
            raise ValueError(f"in_chans must be positive, got {in_chans}")
        if out_chans is not None and out_chans <= 0:
            raise ValueError(f"out_chans must be positive, got {out_chans}")
        if num_output_frames <= 0:
            raise ValueError(f"num_output_frames must be positive, got {num_output_frames}")
        size = to_2tuple(img_size)
        self.variant = variant
        self.img_size = size
        self.in_chans = int(in_chans)
        self.out_chans = self.in_chans if out_chans is None else int(out_chans)
        self.num_output_frames = int(num_output_frames)
        common = dict(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

        if variant == "d":
            depths_downsample = [int(d) for d in depths_downsample]
            depths_upsample = [int(d) for d in depths_upsample]
            if not depths_downsample or len(depths_upsample) != len(depths_downsample):
                raise ValueError("depths_downsample and depths_upsample must be non-empty and of equal length.")
            if len(num_heads) < len(depths_downsample):
                raise ValueError("num_heads needs one entry per SwinLSTM-D level.")
            _check_geometry(size, patch_size, window_size, levels=len(depths_downsample))
            self.num_layers = len(depths_downsample)
            self.Downsample = DownSample(depths_downsample=depths_downsample, num_heads=num_heads, **common)
            self.Upsample = UpSample(
                depths_upsample=depths_upsample, num_heads=num_heads, out_chans=self.out_chans, **common
            )
        else:
            depths = [int(d) for d in depths]
            if not depths or len(num_heads) < len(depths):
                raise ValueError("depths must be non-empty and num_heads needs one entry per SwinLSTM-B cell.")
            _check_geometry(size, patch_size, window_size, levels=0)
            self.num_layers = len(depths)
            self.ST = STconvert(depths=depths, num_heads=num_heads, out_chans=self.out_chans, **common)

    def _step(self, frame: torch.Tensor, states: Optional[tuple], apply_sigmoid: bool = True):
        empty = [None] * self.num_layers
        if self.variant == "d":
            states_down, states_up = states if states is not None else (empty, empty)
            states_down, x = self.Downsample(frame, states_down)
            states_up, output = self.Upsample(x, states_up, apply_sigmoid=apply_sigmoid)
            return output, (states_down, states_up)
        (cell_states,) = states if states is not None else (empty,)
        cell_states, output = self.ST(frame, cell_states, apply_sigmoid=apply_sigmoid)
        return output, (cell_states,)

    def step(self, frame: torch.Tensor, *states):
        """One reference time step on a ``(batch, in_chans, H, W)`` frame.

        SwinLSTM-D: ``step(frame, states_down, states_up) -> (output, states_down, states_up)``;
        SwinLSTM-B: ``step(frame, states) -> (output, states)``. States may be lists of ``None``
        (or omitted) for zero initial states, as in the reference.
        """
        if frame.ndim != 4 or frame.shape[1] != self.in_chans or tuple(frame.shape[-2:]) != self.img_size:
            raise ValueError(
                f"SwinLSTM.step expects a frame of shape (batch, {self.in_chans}, {self.img_size[0]}, "
                f"{self.img_size[1]}), got {tuple(frame.shape)}."
            )
        expected = 2 if self.variant == "d" else 1
        if states and len(states) != expected:
            raise ValueError(f"SwinLSTM-{self.variant.upper()} step takes {expected} state arguments, got {len(states)}.")
        output, new_states = self._step(frame, tuple(states) if states else None)
        return (output,) + tuple(new_states)

    def _check_input(self, x: torch.Tensor) -> None:
        if x.ndim != 5:
            raise ValueError(
                "SwinLSTM expects input shape (batch, time, channels, height, width), "
                f"got {tuple(x.shape)}."
            )
        if x.shape[1] < 1:
            raise ValueError("SwinLSTM needs at least one input frame.")
        if x.shape[2] != self.in_chans or tuple(x.shape[-2:]) != self.img_size:
            raise ValueError(
                f"SwinLSTM expects frames of shape ({self.in_chans}, {self.img_size[0]}, {self.img_size[1]}), "
                f"got input of shape {tuple(x.shape)}."
            )

    def forward(
        self, x: torch.Tensor, num_output_frames: Optional[int] = None, include_warmup: bool = False
    ) -> torch.Tensor:
        self._check_input(x)
        n_out = self.num_output_frames if num_output_frames is None else int(num_output_frames)
        if n_out <= 0:
            raise ValueError(f"num_output_frames must be positive, got {n_out}")
        if n_out > 1 and self.out_chans != self.in_chans:
            raise ValueError(
                "multi-frame rollout feeds each prediction back as the next input, so it needs "
                f"out_chans == in_chans (got {self.out_chans} and {self.in_chans})."
            )
        steps = x.shape[1]
        states = None
        outputs = []
        for t in range(steps - 1):
            output, states = self._step(x[:, t], states)
            outputs.append(output)
        last_input = x[:, -1]
        for _ in range(n_out):
            output, states = self._step(last_input, states)
            outputs.append(output)
            last_input = output
        frames = torch.stack(outputs, dim=1)
        return frames if include_warmup else frames[:, steps - 1 :]


class SwinLSTMSegmenter(SwinLSTM):
    """PyHazards adaptation for next-step fire masks: ``(batch, T, C, H, W)`` -> ``(batch, out_chans, H, W)``.

    The frames are fed through the reference rollout with one prediction step (warm-up on
    ``x[:, :-1]``, prediction from ``x[:, -1]``); the reconstruction layer has ``out_chans``
    channels and its output is returned before the final sigmoid, as logits. Parameter names are
    those of :class:`SwinLSTM` (no prefix).
    """

    def __init__(self, out_chans: int = 1, **kwargs):
        kwargs.pop("num_output_frames", None)
        super().__init__(out_chans=out_chans, num_output_frames=1, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        self._check_input(x)
        states = None
        for t in range(x.shape[1]):
            logits, states = self._step(x[:, t], states, apply_sigmoid=False)
        return logits


def swinlstm_builder(
    task: str,
    in_channels: int = 1,
    out_channels: Optional[int] = None,
    img_size: ImageSize = 64,
    variant: str = "d",
    patch_size: int = 2,
    embed_dim: int = 128,
    depths_downsample: Sequence[int] = (2, 6),
    depths_upsample: Sequence[int] = (6, 2),
    depths: Sequence[int] = (12,),
    num_heads: Sequence[int] = (4, 8),
    window_size: int = 4,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    num_output_frames: int = 10,
    **kwargs,
) -> nn.Module:
    """Build SwinLSTM. Defaults are the official Moving-MNIST SwinLSTM-D configuration.

    ``task="forecasting"``: :class:`SwinLSTM` (frames to frames). ``task="segmentation"``:
    :class:`SwinLSTMSegmenter`, one output frame of ``out_channels`` (default 1) logits.
    """
    _ = kwargs
    task = task.lower()
    config = dict(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_channels,
        embed_dim=embed_dim,
        depths_downsample=depths_downsample,
        depths_upsample=depths_upsample,
        num_heads=num_heads,
        window_size=window_size,
        variant=variant,
        depths=depths,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
    )
    if task == "forecasting":
        return SwinLSTM(out_chans=out_channels, num_output_frames=num_output_frames, **config)
    if task == "segmentation":
        return SwinLSTMSegmenter(out_chans=1 if out_channels is None else out_channels, **config)
    raise ValueError(f"swinlstm supports task='forecasting' or 'segmentation', got {task!r}.")


__all__ = [
    "SwinLSTM",
    "SwinLSTMBlock",
    "SwinLSTMCell",
    "SwinLSTMSegmenter",
    "swinlstm_builder",
]
