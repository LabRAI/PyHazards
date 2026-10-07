"""Convolutional GRU encoder with a convolutional head: the Conv-GRU baseline of FireCastNet.

The recurrence is the convolutional GRU of Ballas et al. (ICLR 2016, arXiv:1511.06432, Eq. 7-8):
``z = sigmoid(W_z * x + U_z * h)``, ``r = sigmoid(W_r * x + U_r * h)``,
``h~ = tanh(W * x + U * (r . h))``, ``h' = (1 - z) . h + z . h~``, with ``W * x + U * h``
computed as one convolution over ``[x, h]``. The model is the ``ConvGRUSeg`` network that
FireCastNet (Michail et al., Scientific Reports 2025, arXiv:2502.01550) trained as its Conv-GRU
baseline: a stack of cells over ``(batch, time, channels, height, width)`` and a ``Conv2d`` head on
the last hidden state.

This file is written for PyHazards from those equations and parallels
:mod:`pyhazards.models.convlstm`. FireCastNet's implementation (SeasFire/firecastnet,
``seasfire/backbones/conv_gru.py``, "modified from TUM-LMF/MTLCC-pytorch") has no license, so none
of its code is copied; it is used only as the test oracle. Module names and creation order follow
it, so its state dicts load with ``strict=True`` and the same seed gives the same initial weights.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn

KernelSize = Union[int, Tuple[int, int]]


def _pair(kernel_size: KernelSize) -> Tuple[int, int]:
    if isinstance(kernel_size, int):
        return (kernel_size, kernel_size)
    kernel = tuple(int(k) for k in kernel_size)
    if len(kernel) != 2:
        raise ValueError(f"kernel_size must be an int or a pair, got {kernel_size!r}")
    return kernel  # type: ignore[return-value]


class ConvGRUCell(nn.Module):
    """One ConvGRU step: ``in_conv`` gives the update and reset gates, ``out_conv`` the candidate."""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: KernelSize = 3, bias: bool = True):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.kernel_size = _pair(kernel_size)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        self.in_conv = nn.Conv2d(
            self.input_dim + self.hidden_dim, 2 * self.hidden_dim, self.kernel_size, padding=self.padding, bias=bias
        )
        self.out_conv = nn.Conv2d(
            self.input_dim + self.hidden_dim, self.hidden_dim, self.kernel_size, padding=self.padding, bias=bias
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        update, reset = torch.sigmoid(self.in_conv(torch.cat([x, h], dim=1))).chunk(2, dim=1)
        candidate = torch.tanh(self.out_conv(torch.cat([x, reset * h], dim=1)))
        return (1 - update) * h + update * candidate

    def init_hidden(self, batch_size: int, height: int, width: int, like: torch.Tensor) -> torch.Tensor:
        return like.new_zeros(batch_size, self.hidden_dim, height, width)


class ConvGRU(nn.Module):
    """Stacked ConvGRU over ``(batch, time, channels, height, width)`` sequences.

    Returns the per-step hidden states of the last layer and the final hidden state of every layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Union[int, Sequence[int]],
        kernel_size: Union[KernelSize, Sequence[KernelSize]] = 3,
        num_layers: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        hidden_dims = list(hidden_dim) if isinstance(hidden_dim, (list, tuple)) else [hidden_dim] * num_layers
        if isinstance(kernel_size, (list, tuple)) and kernel_size and isinstance(kernel_size[0], (list, tuple)):
            kernel_sizes = [_pair(k) for k in kernel_size]
        else:
            kernel_sizes = [_pair(kernel_size)] * num_layers
        if not len(hidden_dims) == len(kernel_sizes) == num_layers:
            raise ValueError("hidden_dim and kernel_size lists must have num_layers entries.")
        self.input_dim = int(input_dim)
        self.hidden_dim = [int(h) for h in hidden_dims]
        self.num_layers = int(num_layers)
        self.cell_list = nn.ModuleList(
            ConvGRUCell(
                input_dim=self.input_dim if i == 0 else self.hidden_dim[i - 1],
                hidden_dim=self.hidden_dim[i],
                kernel_size=kernel_sizes[i],
                bias=bias,
            )
            for i in range(self.num_layers)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if x.ndim != 5:
            raise ValueError(
                f"ConvGRU expects input shape (batch, time, channels, height, width), got {tuple(x.shape)}."
            )
        batch, steps, _, height, width = x.shape
        layer_input = x
        last_states: List[torch.Tensor] = []
        for cell in self.cell_list:
            h = cell.init_hidden(batch, height, width, like=x)
            outputs = []
            for t in range(steps):
                h = cell(layer_input[:, t], h)
                outputs.append(h)
            layer_input = torch.stack(outputs, dim=1)
            last_states.append(h)
        return layer_input, last_states


class ConvGRUSegmenter(nn.Module):
    """ConvGRU encoder + ``Conv2d`` head on the last hidden state (FireCastNet's ``ConvGRUSeg``).

    Input ``(batch, time, channels, height, width)``; output ``(batch, num_classes, height, width)``
    logits. The head uses the cells' kernel size with padding ``(kernel - 1) // 2`` per dimension.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 1,
        hidden_dim: int = 128,
        kernel_size: KernelSize = 5,
        num_layers: int = 1,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        kernel = _pair(kernel_size)
        self.input_dim = int(input_dim)
        self.convgru_encoder = ConvGRU(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel, num_layers=num_layers)
        self.classification_layer = nn.Conv2d(
            in_channels=int(hidden_dim),
            out_channels=num_classes,
            kernel_size=kernel,
            padding=((kernel[0] - 1) // 2, (kernel[1] - 1) // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                "ConvGRUSegmenter expects input shape (batch, time, channels, height, width), "
                f"got {tuple(x.shape)}."
            )
        if x.size(2) != self.input_dim:
            raise ValueError(f"ConvGRUSegmenter expected {self.input_dim} channels, got input of shape {tuple(x.shape)}.")
        _, states = self.convgru_encoder(x)
        return self.classification_layer(states[-1])


def convgru_builder(
    task: str,
    in_channels: int = 11,
    out_channels: int = 1,
    hidden_dim: int = 128,
    kernel_size: KernelSize = 5,
    num_layers: int = 1,
    **kwargs,
) -> nn.Module:
    """FireCastNet's Conv-GRU baseline; defaults are its ``configs/conv-gru-config.yaml``."""
    _ = kwargs
    if task.lower() != "segmentation":
        raise ValueError(
            f"convgru supports task='segmentation' (the FireCastNet ConvGRUSeg model), got {task!r}. "
            "For a ConvGRU frame forecaster use build_model('trajgru', task='forecasting', layer_type='ConvGRU')."
        )
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
    return ConvGRUSegmenter(
        input_dim=in_channels,
        num_classes=out_channels,
        hidden_dim=hidden_dim,
        kernel_size=kernel_size,
        num_layers=num_layers,
    )


__all__ = ["ConvGRUCell", "ConvGRU", "ConvGRUSegmenter", "convgru_builder"]
