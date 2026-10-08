"""ConvLSTM encoder with a convolutional segmentation head.

Port of the ConvLSTM used by the WildfireSpreadTS benchmark (Gerard et al., NeurIPS 2023
Datasets & Benchmarks). WildfireSpreadTS vendors it from ``VSainteuf/utae-paps``
(``src/backbones/convlstm.py``, MIT), which in turn takes it from ``TUM-LMF/MTLCC-pytorch``.
The recurrence follows Shi et al. (NeurIPS 2015) without peephole connections, and the
segmentation head reads the *cell state* of the last time step, exactly as the reference
``ConvLSTM_Seg`` does.

Parameter names match the reference modules, so reference state dicts load with
``strict=True``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

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


class ConvLSTMCell(nn.Module):
    """One ConvLSTM step: a single convolution over ``[x_t, h_{t-1}]`` produces the four gates."""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: KernelSize = 3, bias: bool = True):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.kernel_size = _pair(kernel_size)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=bias,
        )

    def forward(
        self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_cur, c_cur = state
        gates = self.conv(torch.cat([x, h_cur], dim=1))
        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size: int, height: int, width: int, like: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        zeros = like.new_zeros(batch_size, self.hidden_dim, height, width)
        return zeros, zeros.clone()


class ConvLSTM(nn.Module):
    """Stacked ConvLSTM over ``(batch, time, channels, height, width)`` sequences.

    Returns the per-step hidden states of the last layer and the final ``(h, c)`` of every layer.
    Unlike the reference code, the spatial size is taken from the input instead of being fixed
    at construction time; outputs are identical for the size the reference was built with.
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
            ConvLSTMCell(
                input_dim=self.input_dim if i == 0 else self.hidden_dim[i - 1],
                hidden_dim=self.hidden_dim[i],
                kernel_size=kernel_sizes[i],
                bias=bias,
            )
            for i in range(self.num_layers)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        if x.ndim != 5:
            raise ValueError(
                "ConvLSTM expects input shape (batch, time, channels, height, width), "
                f"got {tuple(x.shape)}."
            )
        batch, steps, _, height, width = x.shape
        layer_input = x
        last_states: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for cell in self.cell_list:
            h, c = cell.init_hidden(batch, height, width, like=x)
            outputs = []
            for t in range(steps):
                h, c = cell(layer_input[:, t], (h, c))
                outputs.append(h)
            layer_input = torch.stack(outputs, dim=1)
            last_states.append((h, c))
        return layer_input, last_states


class ConvLSTMSegmenter(nn.Module):
    """ConvLSTM encoder + ``Conv2d`` head on the last cell state (the WildfireSpreadTS ConvLSTM).

    Input ``(batch, time, channels, height, width)``; output ``(batch, num_classes, height, width)`` logits.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 1,
        hidden_dim: int = 64,
        kernel_size: KernelSize = 3,
        num_layers: int = 1,
        readout: str = "cell",
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if readout not in {"cell", "hidden"}:
            raise ValueError(f"readout must be 'cell' or 'hidden', got {readout!r}")
        kernel = _pair(kernel_size)
        self.input_dim = int(input_dim)
        self.readout = readout
        self.convlstm_encoder = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel,
            num_layers=num_layers,
        )
        self.classification_layer = nn.Conv2d(
            in_channels=int(hidden_dim),
            out_channels=num_classes,
            kernel_size=kernel,
            padding=(kernel[0] // 2, kernel[1] // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                "ConvLSTMSegmenter expects input shape (batch, time, channels, height, width), "
                f"got {tuple(x.shape)}."
            )
        if x.size(2) != self.input_dim:
            raise ValueError(f"ConvLSTMSegmenter expected {self.input_dim} channels, got {x.size(2)}.")
        _, states = self.convlstm_encoder(x)
        h, c = states[-1]
        return self.classification_layer(c if self.readout == "cell" else h)


def convlstm_builder(
    task: str,
    in_channels: int,
    out_channels: int = 1,
    hidden_dim: int = 64,
    kernel_size: KernelSize = 3,
    num_layers: int = 1,
    readout: str = "cell",
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() != "segmentation":
        raise ValueError(f"convlstm supports task='segmentation', got {task!r}.")
    return ConvLSTMSegmenter(
        input_dim=in_channels,
        num_classes=out_channels,
        hidden_dim=hidden_dim,
        kernel_size=kernel_size,
        num_layers=num_layers,
        readout=readout,
    )


__all__ = ["ConvLSTMCell", "ConvLSTM", "ConvLSTMSegmenter", "convlstm_builder"]
