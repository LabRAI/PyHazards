"""Temporal Convolutional Network (Bai, Kolter & Koltun, 2018).

Port of ``TemporalConvNet``, ``TemporalBlock`` and ``Chomp1d`` from the official repository
locuslab/TCN (``TCN/tcn.py`` at 2f8c2b8, MIT License, Copyright (c) 2018 CMU Locus Lab), plus the
linear task heads of its examples (``TCN/adding_problem/model.py``, ``TCN/mnist_pixel/model.py``,
``TCN/copy_memory/model.py``, ``TCN/poly_music/model.py``).

Each residual block is two causal dilated convolutions (left padding ``(k - 1) * d`` followed by a
chomp of the same length), each with weight normalisation, ReLU and dropout, plus a 1x1
convolution on the residual path when the width changes. Level ``i`` uses dilation ``2 ** i``.

Two choices differ from the official file without changing its numbers:

* Weight normalisation uses ``torch.nn.utils.parametrizations.weight_norm`` instead of the
  deprecated hook-based ``torch.nn.utils.weight_norm``. Both compute ``g * v / ||v||`` with
  ``torch._weight_norm`` and initialise ``g = ||w||``, ``v = w`` from the same convolution weight.
  The parameters are stored as ``<conv>.parametrizations.weight.original0`` (g) and ``.original1``
  (v) instead of ``<conv>.weight_g`` / ``.weight_v``. Official state dicts load with
  ``strict=True`` regardless, because PyTorch's parametrisation renames ``weight_g`` / ``weight_v``
  while loading; :func:`to_official_state_dict` converts in the other direction.
* :class:`TCN` takes PyHazards' ``(batch, time, features)`` layout and transposes it to the
  ``(batch, channels, length)`` layout of the official code; :class:`TemporalConvNet` keeps the
  official layout.

As in the official code, ``init_weights`` draws ``N(0, 0.01)`` into the *computed* weights of the
two weight-normalised convolutions. Those values are discarded, because weight norm recomputes
the weight from ``g`` and ``v`` on every forward pass, so the effective initialisation of these
convolutions is PyTorch's default one; only the 1x1 residual convolution (and the adding-problem
style linear head) really receives ``N(0, 0.01)``. The draws are kept so that a given seed
produces the same parameters as the official code.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

_PARAMETRIZED_G = "parametrizations.weight.original0"
_PARAMETRIZED_V = "parametrizations.weight.original1"


class Chomp1d(nn.Module):
    """Drop the last ``chomp_size`` steps, which turns symmetric padding into causal padding."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Residual block of two weight-normalised causal dilated convolutions."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        if padding <= 0:
            raise ValueError(
                "TemporalBlock needs padding = (kernel_size - 1) * dilation > 0, i.e. kernel_size >= 2; "
                f"got padding={padding}."
            )
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # The official code registers the layers twice (as attributes and inside ``net``), so its
        # state dicts contain both ``conv1.*`` and ``net.0.*`` keys; keep the same structure.
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self) -> None:
        # Official initialisation. For conv1/conv2 the draws land in the computed weight, which
        # weight norm discards (see the module docstring); they are kept for seed compatibility.
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Stack of :class:`TemporalBlock` with dilations ``1, 2, 4, ...`` (official layout).

    Input ``(batch, num_inputs, length)``; output ``(batch, num_channels[-1], length)``. The output
    at step ``t`` depends only on inputs ``<= t``; the receptive field is
    ``1 + 2 * (kernel_size - 1) * (2 ** levels - 1)`` steps.
    """

    def __init__(self, num_inputs: int, num_channels: Sequence[int], kernel_size: int = 2, dropout: float = 0.2):
        super().__init__()
        if num_inputs <= 0:
            raise ValueError(f"num_inputs must be positive, got {num_inputs}")
        if len(num_channels) == 0 or any(int(c) <= 0 for c in num_channels):
            raise ValueError(f"num_channels must be a non-empty list of positive widths, got {num_channels!r}")
        if kernel_size < 2:
            raise ValueError(f"kernel_size must be >= 2 for the causal chomp, got {kernel_size}")
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]
        self.network = nn.Sequential(*layers)
        self.num_inputs = int(num_inputs)
        self.kernel_size = int(kernel_size)
        self.num_levels = num_levels

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * (self.kernel_size - 1) * (2**self.num_levels - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != self.num_inputs:
            raise ValueError(
                f"TemporalConvNet expects input shape (batch, {self.num_inputs}, length), got {tuple(x.shape)}."
            )
        return self.network(x)


class TCN(nn.Module):
    """:class:`TemporalConvNet` with the linear heads of the official examples.

    Args:
        input_size: features per time step.
        output_size: outputs (classes or regression targets).
        num_channels: width of each level.
        kernel_size: temporal kernel size.
        dropout: dropout inside the temporal blocks.
        readout: ``"last"`` applies the head to the last time step (adding problem, sequential
            MNIST) and returns ``(batch, output_size)``; ``"sequence"`` applies it to every step
            (copy memory, polyphonic music) and returns ``(batch, time, output_size)``.
        head_init: ``"normal"`` draws the head weight from ``N(0, 0.01)`` (adding problem, copy
            memory); ``"default"`` keeps PyTorch's initialisation (sequential MNIST, music).

    Input ``(batch, time, input_size)``. Outputs are raw scores: the official sequential-MNIST
    head's ``log_softmax`` and the music head's ``sigmoid`` belong in the loss.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_channels: Sequence[int],
        kernel_size: int = 2,
        dropout: float = 0.2,
        readout: str = "last",
        head_init: str = "normal",
    ):
        super().__init__()
        if output_size <= 0:
            raise ValueError(f"output_size must be positive, got {output_size}")
        if readout not in {"last", "sequence"}:
            raise ValueError(f"readout must be 'last' or 'sequence', got {readout!r}")
        if head_init not in {"normal", "default"}:
            raise ValueError(f"head_init must be 'normal' or 'default', got {head_init!r}")
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.input_size = int(input_size)
        self.readout = readout
        if head_init == "normal":
            self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(-1) != self.input_size:
            raise ValueError(
                f"TCN expects input shape (batch, time, {self.input_size}), got {tuple(x.shape)}."
            )
        y = self.tcn(x.transpose(1, 2))
        if self.readout == "last":
            return self.linear(y[:, :, -1])
        return self.linear(y.transpose(1, 2))


def to_official_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Rename weight-norm parameters to the official ``weight_g`` / ``weight_v`` keys.

    The result loads with ``strict=True`` into the locuslab/TCN modules. The reverse direction
    needs no conversion: PyHazards modules accept official state dicts directly.
    """
    converted: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.endswith(_PARAMETRIZED_G):
            key = key[: -len(_PARAMETRIZED_G)] + "weight_g"
        elif key.endswith(_PARAMETRIZED_V):
            key = key[: -len(_PARAMETRIZED_V)] + "weight_v"
        converted[key] = value
    return converted


def tcn_builder(
    task: str,
    input_dim: int = 2,
    out_dim: int = 1,
    hidden_dim: int = 30,
    num_levels: int = 8,
    num_channels: Optional[Sequence[int]] = None,
    kernel_size: int = 7,
    dropout: float = 0.0,
    readout: str = "last",
    head_init: str = "normal",
    **kwargs,
) -> nn.Module:
    """Build a :class:`TCN`; the defaults are the official adding-problem model.

    ``num_channels`` overrides ``hidden_dim`` / ``num_levels`` (``[hidden_dim] * num_levels``, as
    in the official scripts).
    """
    _ = kwargs
    if task.lower() not in {"classification", "regression"}:
        raise ValueError(f"tcn supports task='classification' or 'regression', got {task!r}.")
    if num_channels is None:
        if num_levels <= 0 or hidden_dim <= 0:
            raise ValueError("hidden_dim and num_levels must be positive.")
        num_channels = [hidden_dim] * num_levels
    return TCN(
        input_size=input_dim,
        output_size=out_dim,
        num_channels=list(num_channels),
        kernel_size=kernel_size,
        dropout=dropout,
        readout=readout,
        head_init=head_init,
    )


__all__ = ["Chomp1d", "TemporalBlock", "TemporalConvNet", "TCN", "to_official_state_dict", "tcn_builder"]
