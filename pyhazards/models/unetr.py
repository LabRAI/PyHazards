"""UNETR as implemented by MONAI, with TS-SatFire's spatio-temporal (UNETR-3D) modifications.

Port of ``monai.networks.nets.UNETR`` from MONAI 1.3.2 (``monai/networks/nets/unetr.py``,
``monai/networks/nets/vit.py``, ``monai/networks/blocks/{patchembedding,transformerblock,
selfattention}.py`` and the UNETR / DynUNet blocks in :mod:`pyhazards.models.monai_blocks`;
Apache License 2.0, Copyright (c) MONAI Consortium). Changes made for PyHazards: plain PyTorch
(no MONAI or einops dependency), invalid configurations and input shapes raise ``ValueError`` up
front, three options reproduce the TS-SatFire modifications below, and :class:`TemporalUNETR`
adds the TS-SatFire input layout and temporal read-out. Module names, creation order and defaults
follow MONAI, so MONAI state dicts load with ``strict=True`` and the same seed gives the same
initial weights.

UNETR is Hatamizadeh et al., "UNETR: Transformers for 3D Medical Image Segmentation" (WACV 2022,
arXiv:2103.10504): a 12-layer ViT encoder on non-overlapping patches whose layer-3/6/9/12 tokens
are reshaped to feature maps and up-sampled by transposed convolutions into a convolutional
U-Net decoder. The authors' official code (Project-MONAI/research-contributions) is built on
MONAI's blocks.

TS-SatFire (Zhao, Gerard & Ban, Scientific Data 12:1817, 2025) uses it in two ways: "UNETR-2D"
is stock MONAI ``UNETR(spatial_dims=2, img_size=(256, 256), feature_size=16, hidden_size=384,
mlp_dim=1536, norm_name="batch")`` (23,521,186 parameters with 8 input channels, Table 3:
23.52M); "UNETR-3D" uses the repository's copy ``spatial_models/unetr/unetr.py``, which

- takes a ``patch_size`` argument -- ``(1, 16, 16)`` over (time, height, width), so patches never
  span several days (MONAI hard-codes 16 in every dimension);
- up-samples with ``(1, 2, 2)`` transposed convolutions everywhere (``kernel_size_up_down`` in the
  prediction script), so time is never up-sampled; and
- uses a ``(1, 3, 3)`` kernel in ``decoder3`` (``(3, 3, 3)`` everywhere else).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .monai_blocks import (
    MLPBlock,
    NormSpec,
    TemporalReadout,
    UnetOutBlock,
    UnetrBasicBlock,
    UnetrPrUpBlock,
    UnetrUpBlock,
    as_tuple,
    check_norm_name,
    trunc_normal_,
)

IntOrSeq = Union[int, Sequence[int]]

_CONV = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}

# TS-SatFire UNETR-3D (run_spatial_temp_model_pred.py and spatial_models/unetr/unetr.py).
TS_SATFIRE_UNETR3D_PATCH = (1, 16, 16)
TS_SATFIRE_UNETR3D_UP = (1, 2, 2)
TS_SATFIRE_UNETR3D_DECODER3_KERNEL = (1, 3, 3)


class PatchEmbeddingBlock(nn.Module):
    """ViT patch embedding with a learnable position embedding (``monai.networks.blocks``).

    ``proj_type="conv"`` uses a strided convolution, ``"perceptron"`` a linear layer on flattened
    ``(patch..., channel)`` vectors. The position embedding is initialised with a truncated normal
    (std 0.02), linear layers with a truncated normal and zero bias.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int],
        patch_size: Sequence[int],
        hidden_size: int,
        num_heads: int,
        proj_type: str = "conv",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ):
        super().__init__()
        if not 0 <= dropout_rate <= 1:
            raise ValueError(f"dropout_rate {dropout_rate} should be between 0 and 1.")
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden size {hidden_size} should be divisible by num_heads {num_heads}.")
        if proj_type not in {"conv", "perceptron"}:
            raise ValueError(f"proj_type must be 'conv' or 'perceptron', got {proj_type!r}")
        self.proj_type = proj_type
        self.spatial_dims = spatial_dims
        self.patch_size = tuple(patch_size)
        for m, p in zip(img_size, patch_size):
            if m < p:
                raise ValueError("patch_size should be smaller than img_size.")
            if self.proj_type == "perceptron" and m % p != 0:
                raise ValueError("patch_size should be divisible by img_size for perceptron.")
        self.n_patches = int(np.prod([im_d // p_d for im_d, p_d in zip(img_size, patch_size)]))
        self.patch_dim = int(in_channels * np.prod(patch_size))
        self.patch_embeddings: nn.Module
        if self.proj_type == "conv":
            self.patch_embeddings = _CONV[spatial_dims](
                in_channels=in_channels, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size
            )
        else:
            # MONAI: nn.Sequential(Rearrange("b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)"), Linear).
            self.patch_embeddings = nn.Sequential(_PatchFlatten(self.patch_size), nn.Linear(self.patch_dim, hidden_size))
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)
        trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embeddings(x)
        if self.proj_type == "conv":
            x = x.flatten(2).transpose(-1, -2)
        return self.dropout(x + self.position_embeddings)


class _PatchFlatten(nn.Module):
    """``b c (s1 p1) (s2 p2) ... -> b (s1 s2 ...) (p1 p2 ... c)`` (MONAI's einops ``Rearrange``)."""

    def __init__(self, patch_size: Tuple[int, ...]):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        dims = len(self.patch_size)
        grid = [s // p for s, p in zip(x.shape[2:], self.patch_size)]
        shape: List[int] = [b, c]
        for g, p in zip(grid, self.patch_size):
            shape += [g, p]
        x = x.reshape(shape)
        order = [0] + [2 + 2 * i for i in range(dims)] + [3 + 2 * i for i in range(dims)] + [1]
        return x.permute(order).reshape(b, int(np.prod(grid)), -1)


class SABlock(nn.Module):
    """Multi-head self-attention (``monai.networks.blocks.SABlock``)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ):
        super().__init__()
        if not 0 <= dropout_rate <= 1:
            raise ValueError("dropout_rate should be between 0 and 1.")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")
        self.num_heads = num_heads
        self.dim_head = hidden_size // num_heads
        self.inner_dim = self.dim_head * num_heads
        self.out_proj = nn.Linear(self.inner_dim, hidden_size)
        self.qkv = nn.Linear(hidden_size, self.inner_dim * 3, bias=qkv_bias)
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.scale = self.dim_head**-0.5
        self.save_attn = save_attn
        self.att_mat = torch.Tensor()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n = x.shape[:2]
        # einops "b h (qkv l d) -> qkv b l h d"
        q, k, v = self.qkv(x).reshape(b, n, 3, self.num_heads, self.dim_head).permute(2, 0, 3, 1, 4)
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1)
        if self.save_attn:
            self.att_mat = att_mat.detach()
        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        x = x.transpose(1, 2).reshape(b, n, self.inner_dim)  # "b h l d -> b l (h d)"
        return self.drop_output(self.out_proj(x))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block (``monai.networks.blocks.TransformerBlock``)."""

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ):
        super().__init__()
        if not 0 <= dropout_rate <= 1:
            raise ValueError("dropout_rate should be between 0 and 1.")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SABlock(hidden_size, num_heads, dropout_rate, qkv_bias, save_attn)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """MONAI ``ViT`` encoder without the classification head (UNETR never builds it).

    Returns the final normalised tokens and the list of every block's output.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int],
        patch_size: Sequence[int],
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        proj_type: str = "conv",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ):
        super().__init__()
        if not 0 <= dropout_rate <= 1:
            raise ValueError("dropout_rate should be between 0 and 1.")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        self.classification = False
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            proj_type=proj_type,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.patch_embedding(x)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        return self.norm(x), hidden_states_out


class UNETR(nn.Module):
    """MONAI ``UNETR`` (1.3.2) in plain PyTorch, with TS-SatFire's options.

    Args:
        in_channels: input channels.
        out_channels: output channels (logits, no activation).
        img_size: spatial input size (int or one value per dimension); fixed, because the
            position embedding has one entry per patch.
        feature_size: decoder width (MONAI default 16).
        hidden_size: ViT width (MONAI default 768; TS-SatFire uses 384).
        mlp_dim: ViT MLP width (MONAI default 3072; TS-SatFire uses 1536).
        num_heads: ViT attention heads.
        proj_type: ``"conv"`` or ``"perceptron"`` patch embedding.
        norm_name: ``"instance"`` (MONAI default) or ``"batch"`` (TS-SatFire), or ``(name, kwargs)``.
        conv_block: convolution blocks after the projection up-sampling (MONAI default True).
        res_block: residual convolution blocks (MONAI default True).
        dropout_rate: ViT dropout.
        spatial_dims: 2 or 3.
        qkv_bias: bias in the attention's qkv projection.
        save_attn: keep each block's attention matrix in ``attn.att_mat``.
        patch_size: ViT patch size (MONAI: 16 in every dimension; TS-SatFire: ``(1, 16, 16)``).
        kernel_size_up_down: kernel and stride of every transposed convolution (MONAI: 2;
            TS-SatFire: ``(1, 2, 2)``). Each patch-size entry must equal its fourth power, so that
            the decoder returns to the input resolution.
        decoder3_kernel_size: kernel of the ``decoder3`` convolution block (MONAI: 3;
            TS-SatFire: ``(1, 3, 3)``).

    Input ``(batch, in_channels, *img_size)``; output ``(batch, out_channels, *img_size)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: IntOrSeq,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        proj_type: str = "conv",
        norm_name: NormSpec = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
        patch_size: IntOrSeq = 16,
        kernel_size_up_down: IntOrSeq = 2,
        decoder3_kernel_size: IntOrSeq = 3,
    ):
        super().__init__()
        if spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
        if in_channels <= 0 or out_channels <= 0 or feature_size <= 0:
            raise ValueError("in_channels, out_channels and feature_size must be positive.")
        if not 0 <= dropout_rate <= 1:
            raise ValueError("dropout_rate should be between 0 and 1.")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        check_norm_name(norm_name)
        img_size = as_tuple(img_size, spatial_dims, "img_size")
        patch = as_tuple(patch_size, spatial_dims, "patch_size")
        up = as_tuple(kernel_size_up_down, spatial_dims, "kernel_size_up_down")
        decoder3_kernel = as_tuple(decoder3_kernel_size, spatial_dims, "decoder3_kernel_size")
        if any(k % 2 == 0 for k in decoder3_kernel):
            raise ValueError(f"decoder3_kernel_size must be odd, got {decoder3_kernel_size!r}")
        if any(u < 1 for u in up) or any(p != u**4 for p, u in zip(patch, up)):
            raise ValueError(
                f"Each patch_size entry must be kernel_size_up_down**4 (four up-sampling steps back to the "
                f"input resolution), got patch_size={patch} and kernel_size_up_down={up}."
            )
        if any(m % p for m, p in zip(img_size, patch)):
            raise ValueError(f"img_size {img_size} must be divisible by patch_size {patch}.")

        self.num_layers = 12
        self.img_size = img_size
        self.in_channels = in_channels
        self.spatial_dims = spatial_dims
        self.patch_size = patch
        self.kernel_size_up_down = up
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            proj_type=proj_type,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        for name, out_mult, num_layer in (("encoder2", 2, 2), ("encoder3", 4, 1), ("encoder4", 8, 0)):
            setattr(
                self,
                name,
                UnetrPrUpBlock(
                    spatial_dims=spatial_dims,
                    in_channels=hidden_size,
                    out_channels=feature_size * out_mult,
                    num_layer=num_layer,
                    kernel_size=3,
                    stride=1,
                    upsample_kernel_size=up,
                    norm_name=norm_name,
                    conv_block=conv_block,
                    res_block=res_block,
                ),
            )
        decoders = (
            ("decoder5", hidden_size, feature_size * 8, 3),
            ("decoder4", feature_size * 8, feature_size * 4, 3),
            ("decoder3", feature_size * 4, feature_size * 2, decoder3_kernel),  # TS-SatFire: (1, 3, 3)
            ("decoder2", feature_size * 2, feature_size, 3),
        )
        for name, in_ch, out_ch, kernel in decoders:
            setattr(
                self,
                name,
                UnetrUpBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel,
                    upsample_kernel_size=up,
                    norm_name=norm_name,
                    res_block=res_block,
                ),
            )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

    def proj_feat(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view([x.size(0)] + self.proj_view_shape)
        return x.permute(self.proj_axes).contiguous()

    def _check_input(self, x: torch.Tensor, layout: str) -> None:
        if x.ndim != self.spatial_dims + 2:
            raise ValueError(f"{type(self).__name__} expects input shape {layout}, got {tuple(x.shape)}.")
        if x.size(1) != self.in_channels or tuple(x.shape[2:]) != self.img_size:
            raise ValueError(
                f"{type(self).__name__} was built for {self.in_channels} channels and spatial size "
                f"{self.img_size} (img_size), got shape {tuple(x.shape)}."
            )

    def _forward_network(self, x_in: torch.Tensor) -> torch.Tensor:
        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(self.proj_feat(hidden_states_out[3]))
        enc3 = self.encoder3(self.proj_feat(hidden_states_out[6]))
        enc4 = self.encoder4(self.proj_feat(hidden_states_out[9]))
        dec4 = self.proj_feat(x)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        return self.out(out)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        spatial = ", ".join(["D", "H", "W"][-self.spatial_dims :])
        self._check_input(x_in, f"(batch, channels, {spatial})")
        return self._forward_network(x_in)


class TemporalUNETR(TemporalReadout, UNETR):
    """3D :class:`UNETR` over ``(time, H, W)`` for raster time series (TS-SatFire's UNETR-3D).

    Takes ``(batch, time, channels, H, W)`` with ``(time, H, W) == img_size``, runs the network on
    ``(batch, channels, time, H, W)`` and, with ``time_reduction="mean"``, averages the logits over
    time to ``(batch, out_channels, H, W)``. Parameter names are those of :class:`UNETR`.
    """

    def __init__(self, *args, time_reduction: str = "mean", **kwargs):
        super().__init__(*args, **kwargs)
        if self.spatial_dims != 3:
            raise ValueError(f"TemporalUNETR needs spatial_dims=3, got {self.spatial_dims}")
        self._set_time_reduction(time_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_network_layout(x, type(self).__name__)
        self._check_input(x, "(batch, time, channels, height, width)")
        return self._read_out(self._forward_network(x))


def unetr_builder(
    task: str,
    in_channels: int,
    out_channels: int = 2,
    spatial_dims: int = 3,
    img_size: Optional[IntOrSeq] = None,
    history: int = 6,
    image_size: int = 256,
    feature_size: int = 16,
    hidden_size: int = 384,
    mlp_dim: int = 1536,
    num_heads: int = 12,
    proj_type: str = "conv",
    norm_name: NormSpec = "batch",
    conv_block: bool = True,
    res_block: bool = True,
    dropout_rate: float = 0.0,
    qkv_bias: bool = False,
    patch_size: Optional[IntOrSeq] = None,
    kernel_size_up_down: Optional[IntOrSeq] = None,
    decoder3_kernel_size: Optional[IntOrSeq] = None,
    time_reduction: str = "mean",
    **kwargs,
) -> nn.Module:
    """Build MONAI's UNETR; the defaults are TS-SatFire's UNETR models.

    ``spatial_dims=3`` returns a :class:`TemporalUNETR` (input ``(B, T, C, H, W)``) for
    ``img_size = (history, image_size, image_size)`` with TS-SatFire's patch ``(1, 16, 16)``,
    ``(1, 2, 2)`` up-sampling and ``(1, 3, 3)`` ``decoder3`` kernel. ``spatial_dims=2`` returns a
    stock MONAI :class:`UNETR` (input ``(B, C, H, W)``, patch 16) as in TS-SatFire's UNETR-2D. The
    widths default to the TS-SatFire prediction script (feature size 16, hidden 384, MLP 1536)
    and batch norm; pass ``hidden_size=768, mlp_dim=3072, norm_name="instance"`` for MONAI's
    defaults.
    """
    _ = kwargs
    if task.lower() != "segmentation":
        raise ValueError(f"unetr supports task='segmentation', got {task!r}.")
    if spatial_dims == 3:
        if img_size is None:
            img_size = (history, image_size, image_size)
        model = TemporalUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            proj_type=proj_type,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
            dropout_rate=dropout_rate,
            spatial_dims=3,
            qkv_bias=qkv_bias,
            patch_size=TS_SATFIRE_UNETR3D_PATCH if patch_size is None else patch_size,
            kernel_size_up_down=TS_SATFIRE_UNETR3D_UP if kernel_size_up_down is None else kernel_size_up_down,
            decoder3_kernel_size=(
                TS_SATFIRE_UNETR3D_DECODER3_KERNEL if decoder3_kernel_size is None else decoder3_kernel_size
            ),
            time_reduction=time_reduction,
        )
        return model
    if spatial_dims == 2:
        return UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=(image_size, image_size) if img_size is None else img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            proj_type=proj_type,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
            dropout_rate=dropout_rate,
            spatial_dims=2,
            qkv_bias=qkv_bias,
            patch_size=16 if patch_size is None else patch_size,
            kernel_size_up_down=2 if kernel_size_up_down is None else kernel_size_up_down,
            decoder3_kernel_size=3 if decoder3_kernel_size is None else decoder3_kernel_size,
        )
    raise ValueError(f"unetr supports spatial_dims 2 or 3, got {spatial_dims}.")


__all__ = [
    "PatchEmbeddingBlock",
    "SABlock",
    "TemporalUNETR",
    "TransformerBlock",
    "UNETR",
    "ViT",
    "unetr_builder",
]
