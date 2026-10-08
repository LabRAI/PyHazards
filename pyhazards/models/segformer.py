"""SegFormer: a Mix Transformer (MiT) encoder with an all-MLP decode head.

Xie, Wang, Yu, Anandkumar, Alvarez & Luo, "SegFormer: Simple and Efficient Design for Semantic
Segmentation with Transformers", NeurIPS 2021 (https://arxiv.org/abs/2105.15203).

Ported to plain PyTorch from Hugging Face transformers 4.57.6,
``src/transformers/models/segformer/modeling_segformer.py`` (Apache-2.0, Copyright 2021 NVIDIA and
The HuggingFace Inc. team). The official NVlabs/SegFormer code is under the NVIDIA Source Code
License (non-commercial use only) and was not used.

Module and parameter names follow transformers up to 5.8.x (transformers 5.9 renamed them and maps
the old names on load; the checkpoints on the Hub keep the old ones), so:

- a ``SegformerForSemanticSegmentation`` state dict loads into :class:`SegFormer` with ``strict=True``;
- a ``SegformerModel`` state dict loads into ``SegFormer.segformer`` with ``strict=True``;
- the ImageNet-1k MiT checkpoints ``nvidia/mit-b0`` ... ``nvidia/mit-b5`` (``SegformerForImageClassification``)
  load into ``SegFormer.segformer`` with ``strict=True`` once the ``segformer.`` prefix is removed and
  the 1000-way ``classifier`` is dropped; :meth:`SegFormer.load_hf_state_dict` does this.

Weights are initialised as transformers does (PyTorch defaults at construction, then
``SegformerPreTrainedModel._init_weights``, encoder before decode head), so the same seed gives
the same initial weights as ``SegformerForSemanticSegmentation(config)``.

NVIDIA licenses its SegFormer/MiT weights for non-commercial use only; PyHazards does not bundle
them, and ``encoder_weights="imagenet"`` downloads them from the Hugging Face Hub on request.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# MiT encoder sizes (transformers configs of nvidia/mit-b0..b5, the official mit_b0..b5 classes and
# paper Table 6) and the decode-head width C (256 for the real-time B0/B1, 768 for B2-B5, paper
# Sec. 4.2). All variants share heads (1, 2, 5, 8), reduction ratios (8, 4, 2, 1) and Mix-FFN
# expansion 4; paper Table 6 lists expansion 8 in stages 1-2 of B0-B4, which matches neither the
# released code and weights nor the B1-B5 parameter counts of paper Table 1.
SEGFORMER_VARIANTS: Dict[str, Dict[str, object]] = {
    "b0": {"hidden_sizes": (32, 64, 160, 256), "depths": (2, 2, 2, 2), "decoder_hidden_size": 256},
    "b1": {"hidden_sizes": (64, 128, 320, 512), "depths": (2, 2, 2, 2), "decoder_hidden_size": 256},
    "b2": {"hidden_sizes": (64, 128, 320, 512), "depths": (3, 4, 6, 3), "decoder_hidden_size": 768},
    "b3": {"hidden_sizes": (64, 128, 320, 512), "depths": (3, 4, 18, 3), "decoder_hidden_size": 768},
    "b4": {"hidden_sizes": (64, 128, 320, 512), "depths": (3, 8, 27, 3), "decoder_hidden_size": 768},
    "b5": {"hidden_sizes": (64, 128, 320, 512), "depths": (3, 6, 40, 3), "decoder_hidden_size": 768},
}

# ImageNet-1k MiT encoders on the Hugging Face Hub: pinned revision and sha256 of pytorch_model.bin.
MIT_IMAGENET_CHECKPOINTS: Dict[str, Tuple[str, str]] = {
    "b0": ("80983a413c30d36a39c20203974ae7807835e2b4", "4af8348c2a802bf76115d34797ff5ce1d9f110bb8593c22d9b66d8bd7fa227fc"),
    "b1": ("13ddceec4e8bdf401e7cd7acf5aebc526222518c", "980b86b60db37b1b1528086f6c53d253d879e9e09ebe07397b40fd275738a3bf"),
    "b2": ("3bb39e8739149c3777d0325349b2a6c32c6413db", "4500b5665471b593e6757e15bcca5034f433fe3902fe8ec2b7230774a57f264f"),
    "b3": ("0e0522cf0515903a0d35abc5adc9df78a25fde7c", "5670aa22de1b1848b1d715cc97d8fef77761701dedc8b3d3e44525fd71012896"),
    "b4": ("3844ddaa13d9ce98816bacb904688a5284f164af", "1cddf0f9ed0b7f1639a8f5e339c177673685df3e2e3c9575a649eb15bcf68a55"),
    "b5": ("40357155205b036cf11b61f132d53d2f8861f170", "a389c0a604458fa205446fd08a2c01b74e6591a5da3e77de668c6ca5cbf75356"),
}


def mit_imagenet_url(variant: str) -> str:
    revision, _ = MIT_IMAGENET_CHECKPOINTS[variant]
    return f"https://huggingface.co/nvidia/mit-{variant}/resolve/{revision}/pytorch_model.bin"


@dataclass
class SegformerConfig:
    """The fields of ``transformers.SegformerConfig`` that change the network, with its defaults."""

    num_channels: int = 3
    num_encoder_blocks: int = 4
    depths: Sequence[int] = (2, 2, 2, 2)
    sr_ratios: Sequence[int] = (8, 4, 2, 1)
    hidden_sizes: Sequence[int] = (32, 64, 160, 256)
    patch_sizes: Sequence[int] = (7, 3, 3, 3)
    strides: Sequence[int] = (4, 2, 2, 2)
    num_attention_heads: Sequence[int] = (1, 2, 5, 8)
    mlp_ratios: Sequence[int] = (4, 4, 4, 4)
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    classifier_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    drop_path_rate: float = 0.1
    decoder_hidden_size: int = 256
    num_labels: int = 150

    @classmethod
    def from_variant(cls, variant: str, **overrides) -> "SegformerConfig":
        key = variant.lower()
        if key not in SEGFORMER_VARIANTS:
            raise ValueError(f"variant must be one of {sorted(SEGFORMER_VARIANTS)}, got {variant!r}")
        return replace(cls(**SEGFORMER_VARIANTS[key]), **overrides)


def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Stochastic depth per sample, drawing the same random numbers as transformers."""
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()
    return input.div(keep_prob) * random_tensor


class SegformerDropPath(nn.Module):
    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


class SegformerOverlapPatchEmbeddings(nn.Module):
    """Overlapping patch embedding: strided convolution followed by LayerNorm over channels."""

    def __init__(self, patch_size: int, stride: int, num_channels: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        embeddings = self.proj(pixel_values)
        _, _, height, width = embeddings.shape
        embeddings = embeddings.flatten(2).transpose(1, 2)
        embeddings = self.layer_norm(embeddings)
        return embeddings, height, width


class SegformerEfficientSelfAttention(nn.Module):
    """Multi-head self-attention whose keys and values are spatially reduced by ``sr_ratio``."""

    def __init__(self, config: SegformerConfig, hidden_size: int, num_attention_heads: int, sequence_reduction_ratio: int):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention heads ({num_attention_heads})"
            )
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.sr_ratio = sequence_reduction_ratio
        if sequence_reduction_ratio > 1:
            self.sr = nn.Conv2d(hidden_size, hidden_size, kernel_size=sequence_reduction_ratio, stride=sequence_reduction_ratio)
            self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch_size, _, _ = hidden_states.shape
        query_layer = (
            self.query(hidden_states).view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        )
        if self.sr_ratio > 1:
            num_channels = hidden_states.shape[2]
            hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
            hidden_states = self.sr(hidden_states)
            hidden_states = hidden_states.reshape(batch_size, num_channels, -1).permute(0, 2, 1)
            hidden_states = self.layer_norm(hidden_states)
        key_layer = (
            self.key(hidden_states).view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        )
        value_layer = (
            self.value(hidden_states).view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        )

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        return context_layer.view(context_layer.size()[:-2] + (self.all_head_size,))


class SegformerSelfOutput(nn.Module):
    def __init__(self, config: SegformerConfig, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.dense(hidden_states))


class SegformerAttention(nn.Module):
    def __init__(self, config: SegformerConfig, hidden_size: int, num_attention_heads: int, sequence_reduction_ratio: int):
        super().__init__()
        self.self = SegformerEfficientSelfAttention(config, hidden_size, num_attention_heads, sequence_reduction_ratio)
        self.output = SegformerSelfOutput(config, hidden_size=hidden_size)

    def forward(self, hidden_states: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return self.output(self.self(hidden_states, height, width))


class SegformerDWConv(nn.Module):
    """3x3 depth-wise convolution of the Mix-FFN, which supplies positional information."""

    def __init__(self, dim: int = 768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, hidden_states: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch_size, _, num_channels = hidden_states.shape
        hidden_states = hidden_states.transpose(1, 2).view(batch_size, num_channels, height, width)
        hidden_states = self.dwconv(hidden_states)
        return hidden_states.flatten(2).transpose(1, 2)


class SegformerMixFFN(nn.Module):
    def __init__(self, config: SegformerConfig, in_features: int, hidden_features: int, out_features: Optional[int] = None):
        super().__init__()
        out_features = out_features or in_features
        self.dense1 = nn.Linear(in_features, hidden_features)
        self.dwconv = SegformerDWConv(hidden_features)
        self.intermediate_act_fn = nn.GELU()
        self.dense2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, height: int, width: int) -> torch.Tensor:
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.dwconv(hidden_states, height, width)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        return self.dropout(hidden_states)


class SegformerLayer(nn.Module):
    """Pre-norm Transformer block: efficient self-attention and Mix-FFN, each with stochastic depth."""

    def __init__(
        self,
        config: SegformerConfig,
        hidden_size: int,
        num_attention_heads: int,
        drop_path: float,
        sequence_reduction_ratio: int,
        mlp_ratio: int,
    ):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.attention = SegformerAttention(config, hidden_size, num_attention_heads, sequence_reduction_ratio)
        self.drop_path = SegformerDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.mlp = SegformerMixFFN(config, in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio))

    def forward(self, hidden_states: torch.Tensor, height: int, width: int) -> torch.Tensor:
        attention_output = self.attention(self.layer_norm_1(hidden_states), height, width)
        hidden_states = self.drop_path(attention_output) + hidden_states
        mlp_output = self.mlp(self.layer_norm_2(hidden_states), height, width)
        return self.drop_path(mlp_output) + hidden_states


class SegformerEncoder(nn.Module):
    """The four-stage hierarchical Mix Transformer (MiT)."""

    def __init__(self, config: SegformerConfig):
        super().__init__()
        self.config = config
        # Stochastic depth grows linearly over all blocks (torch.linspace, as in transformers).
        drop_path_decays = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths), device="cpu")]

        self.patch_embeddings = nn.ModuleList(
            SegformerOverlapPatchEmbeddings(
                patch_size=config.patch_sizes[i],
                stride=config.strides[i],
                num_channels=config.num_channels if i == 0 else config.hidden_sizes[i - 1],
                hidden_size=config.hidden_sizes[i],
            )
            for i in range(config.num_encoder_blocks)
        )
        blocks = []
        cur = 0
        for i in range(config.num_encoder_blocks):
            if i != 0:
                cur += config.depths[i - 1]
            blocks.append(
                nn.ModuleList(
                    SegformerLayer(
                        config,
                        hidden_size=config.hidden_sizes[i],
                        num_attention_heads=config.num_attention_heads[i],
                        drop_path=drop_path_decays[cur + j],
                        sequence_reduction_ratio=config.sr_ratios[i],
                        mlp_ratio=config.mlp_ratios[i],
                    )
                    for j in range(config.depths[i])
                )
            )
        self.block = nn.ModuleList(blocks)
        self.layer_norm = nn.ModuleList(nn.LayerNorm(config.hidden_sizes[i]) for i in range(config.num_encoder_blocks))

    def forward(self, pixel_values: torch.Tensor) -> List[torch.Tensor]:
        batch_size = pixel_values.shape[0]
        hidden_states = pixel_values
        all_hidden_states = []
        for embedding_layer, block_layer, norm_layer in zip(self.patch_embeddings, self.block, self.layer_norm):
            hidden_states, height, width = embedding_layer(hidden_states)
            for block in block_layer:
                hidden_states = block(hidden_states, height, width)
            hidden_states = norm_layer(hidden_states)
            hidden_states = hidden_states.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
            all_hidden_states.append(hidden_states)
        return all_hidden_states


def _init_segformer_weights(module: nn.Module, std: float) -> None:
    """``SegformerPreTrainedModel._init_weights`` over every submodule.

    transformers visits modules children-first; only leaf modules carry weights here, and they come
    in the same order either way, so the random-number stream is the same.
    """
    with torch.no_grad():
        for m in module.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                m.weight.normal_(mean=0.0, std=std)
                if m.bias is not None:
                    m.bias.zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                m.bias.zero_()
                m.weight.fill_(1.0)


class SegformerModel(nn.Module):
    """The MiT encoder wrapped as ``transformers.SegformerModel`` (parameters under ``encoder.``).

    ``forward`` returns the four stage outputs ``(batch, hidden_sizes[i], H / 2**(i+2), W / 2**(i+2))``
    (sizes rounded up).
    """

    def __init__(self, config: SegformerConfig):
        super().__init__()
        self.config = config
        self.encoder = SegformerEncoder(config)
        _init_segformer_weights(self, config.initializer_range)

    def forward(self, pixel_values: torch.Tensor) -> List[torch.Tensor]:
        return self.encoder(pixel_values)


class SegformerMLP(nn.Module):
    """Linear embedding of one encoder stage to the decoder width."""

    def __init__(self, config: SegformerConfig, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, config.decoder_hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states.flatten(2).transpose(1, 2))


class SegformerDecodeHead(nn.Module):
    """All-MLP decoder: unify channels, upsample to 1/4 resolution, fuse, classify."""

    def __init__(self, config: SegformerConfig):
        super().__init__()
        self.linear_c = nn.ModuleList(
            SegformerMLP(config, input_dim=config.hidden_sizes[i]) for i in range(config.num_encoder_blocks)
        )
        # Conv-BN-ReLU: the ConvModule of the original implementation.
        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1)

    def forward(self, encoder_hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
        batch_size = encoder_hidden_states[-1].shape[0]
        target_size = encoder_hidden_states[0].size()[2:]
        all_hidden_states = []
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state).permute(0, 2, 1).reshape(batch_size, -1, height, width)
            encoder_hidden_state = F.interpolate(encoder_hidden_state, size=target_size, mode="bilinear", align_corners=False)
            all_hidden_states.append(encoder_hidden_state)
        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.dropout(self.activation(self.batch_norm(hidden_states)))
        return self.classifier(hidden_states)


StateDict = Mapping[str, torch.Tensor]


class SegFormer(nn.Module):
    """SegFormer-B0 ... B5 for dense prediction (``transformers.SegformerForSemanticSegmentation``).

    Input ``(batch, channels, height, width)``, or ``(batch, time, channels, height, width)`` which is
    flattened to ``time * channels`` input channels (data-level fusion). The decode head predicts at
    1/4 resolution; with ``upsample=True`` (default) the logits are resized bilinearly
    (``align_corners=False``) to the input size, as transformers' post-processing and loss do, so
    the output is ``(batch, num_labels, height, width)``. :meth:`raw_logits` returns the 1/4
    resolution logits that transformers returns.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_labels: int = 1,
        variant: str = "b2",
        encoder_weights: Optional[Union[str, Path]] = None,
        upsample: bool = True,
        drop_path_rate: float = 0.1,
        classifier_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
    ):
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if num_labels <= 0:
            raise ValueError(f"num_labels must be positive, got {num_labels}")
        self.variant = variant.lower()
        self.config = SegformerConfig.from_variant(
            self.variant,
            num_channels=int(in_channels),
            num_labels=int(num_labels),
            drop_path_rate=drop_path_rate,
            classifier_dropout_prob=classifier_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
        )
        self.in_channels = int(in_channels)
        self.upsample = bool(upsample)
        # Same construction and initialisation order as SegformerForSemanticSegmentation.__init__:
        # SegformerModel initialises the encoder, then the decode head is built and initialised.
        self.segformer = SegformerModel(self.config)
        self.decode_head = SegformerDecodeHead(self.config)
        _init_segformer_weights(self.decode_head, self.config.initializer_range)
        if encoder_weights is not None:
            self.load_pretrained_encoder(encoder_weights)

    # -- weights -----------------------------------------------------------------------------------

    def load_hf_state_dict(self, state_dict: StateDict, strict: bool = True):
        """Load a transformers SegFormer state dict, choosing the target from its keys.

        - ``SegformerForSemanticSegmentation`` (``segformer.*`` and ``decode_head.*``): the whole model;
        - ``SegformerForImageClassification``, e.g. ``nvidia/mit-b*`` (``segformer.*`` and
          ``classifier.*``): the ``segformer.`` prefix is removed, the ImageNet classifier is dropped,
          and the rest loads into the encoder;
        - ``SegformerModel`` (``encoder.*``): the encoder.
        """
        keys = list(state_dict)
        if any(key.startswith("decode_head.") for key in keys):
            return self.load_state_dict(state_dict, strict=strict)
        if any(key.startswith("segformer.") for key in keys):
            encoder_state = {
                key[len("segformer."):] if key.startswith("segformer.") else key: value
                for key, value in state_dict.items()
                if not key.startswith("classifier.")
            }
            return self.segformer.load_state_dict(encoder_state, strict=strict)
        return self.segformer.load_state_dict(state_dict, strict=strict)

    def load_pretrained_encoder(self, source: Union[str, Path, StateDict] = "imagenet") -> None:
        """Load ImageNet-1k MiT weights into the encoder; the decode head keeps its initialisation.

        ``source`` is ``"imagenet"`` (download ``nvidia/mit-<variant>`` from the Hugging Face Hub at a
        pinned revision, checked against its sha256 prefix and cached by ``torch.hub``), a local
        checkpoint path, or a state dict. Those checkpoints have a 3-channel first patch embedding;
        for other ``in_channels`` it is adapted the way segmentation_models_pytorch adapts ImageNet
        encoders (``patch_first_conv``): summed over RGB for one channel, otherwise the RGB filters
        are repeated cyclically over the input channels and scaled by ``3 / in_channels``.
        """
        if isinstance(source, Mapping):
            state = dict(source)
        elif str(source) == "imagenet":
            revision, sha256 = MIT_IMAGENET_CHECKPOINTS[self.variant]
            state = torch.hub.load_state_dict_from_url(
                mit_imagenet_url(self.variant),
                map_location="cpu",
                progress=False,
                check_hash=True,
                file_name=f"nvidia-mit-{self.variant}-{sha256[:8]}.bin",
                weights_only=True,
            )
        else:
            state = load_checkpoint(source)
        state = {key[len("segformer."):] if key.startswith("segformer.") else key: value for key, value in state.items()}
        state = {key: value for key, value in state.items() if not key.startswith(("classifier.", "decode_head."))}
        key = "encoder.patch_embeddings.0.proj.weight"
        if key in state:
            state[key] = adapt_first_conv_weight(state[key], self.in_channels)
        self.segformer.load_state_dict(state, strict=True)

    # -- forward -----------------------------------------------------------------------------------

    def _check_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            x = x.flatten(start_dim=1, end_dim=2)
        if x.ndim != 4:
            raise ValueError(
                "SegFormer expects input shape (batch, channels, height, width) or "
                f"(batch, time, channels, height, width), got {tuple(x.shape)}."
            )
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"SegFormer expected {self.in_channels} input channels (time * channels for 5-D input), "
                f"got shape {tuple(x.shape)}."
            )
        height, width = x.shape[-2:]
        for size in (height, width):
            for i in range(self.config.num_encoder_blocks):
                patch = self.config.patch_sizes[i]
                size = (size + 2 * (patch // 2) - patch) // self.config.strides[i] + 1
                if size < self.config.sr_ratios[i]:
                    raise ValueError(
                        f"SegFormer input of shape {tuple(x.shape)} is too small: stage {i + 1} has spatial size "
                        f"{size}, below its key/value reduction ratio {self.config.sr_ratios[i]} "
                        "(height and width must be at least 29)."
                    )
        return x

    def raw_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Decode-head logits at 1/4 of the input resolution (``SegformerForSemanticSegmentation`` output)."""
        x = self._check_input(x)
        return self.decode_head(self.segformer(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._check_input(x)
        logits = self.decode_head(self.segformer(x))
        if self.upsample:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


def load_checkpoint(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """Read a Hub checkpoint file (``pytorch_model.bin``, or ``model.safetensors`` if safetensors is installed)."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise ValueError(f"expected 'imagenet' or an existing checkpoint file, got {str(path)!r}")
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # optional: only needed for .safetensors files
            raise ImportError("reading .safetensors checkpoints requires the safetensors package") from exc
        return load_file(str(path))
    return torch.load(path, map_location="cpu", weights_only=True)


def adapt_first_conv_weight(weight: torch.Tensor, in_channels: int) -> torch.Tensor:
    """segmentation_models_pytorch's ``patch_first_conv`` rule for a pretrained RGB convolution."""
    if weight.ndim != 4:
        raise ValueError(f"expected a convolution weight of shape (out, in, kh, kw), got {tuple(weight.shape)}")
    default = weight.shape[1]
    if in_channels == default:
        return weight
    if default != 3:
        raise ValueError(f"can only adapt a 3-channel convolution, got weight shape {tuple(weight.shape)}")
    if in_channels == 1:
        return weight.sum(1, keepdim=True)
    index = torch.tensor([i % default for i in range(in_channels)])
    return weight[:, index] * (default / in_channels)


def segformer_builder(
    task: str,
    in_channels: int,
    out_channels: int = 1,
    variant: str = "b2",
    history: int = 1,
    encoder_weights: Optional[Union[str, Path]] = None,
    upsample: bool = True,
    drop_path_rate: float = 0.1,
    classifier_dropout_prob: float = 0.1,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() != "segmentation":
        raise ValueError(f"segformer supports task='segmentation', got {task!r}.")
    if history <= 0:
        raise ValueError(f"history must be positive, got {history}")
    return SegFormer(
        in_channels=in_channels * history,
        num_labels=out_channels,
        variant=variant,
        encoder_weights=encoder_weights,
        upsample=upsample,
        drop_path_rate=drop_path_rate,
        classifier_dropout_prob=classifier_dropout_prob,
    )


__all__ = [
    "MIT_IMAGENET_CHECKPOINTS",
    "SEGFORMER_VARIANTS",
    "SegFormer",
    "SegformerConfig",
    "SegformerModel",
    "adapt_first_conv_weight",
    "load_checkpoint",
    "mit_imagenet_url",
    "segformer_builder",
]
