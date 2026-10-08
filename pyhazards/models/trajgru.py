"""TrajGRU encoder-forecaster for precipitation nowcasting (Shi et al., NeurIPS 2017).

Attribution: ported to PyTorch from the official MXNet code, sxjscience/HKO-7 at commit
``57b987bd893bd6910996494c247359ace371d333`` (MIT License, Copyright (c) 2017-2027 Xingjian Shi
and others): ``nowcasting/operators/traj_rnn.py`` (TrajGRU cell), ``nowcasting/operators/conv_rnn.py``
(the ConvGRU cell of the paper's baselines), ``nowcasting/operators/base_rnn.py`` (stacked RNN
blocks), ``nowcasting/encoder_forecaster.py`` and ``nowcasting/prediction_base_factory.py``
(encoder-forecaster wiring and input coordinates) and ``nowcasting/ops.py`` (activations,
down/up-sampling modules).

Module names are chosen so that a parameter's PyTorch key with ``.`` replaced by ``_`` is its MXNet
name (``ebrnn1.0.i2h.weight`` <-> ``ebrnn1_0_i2h_weight``); :func:`load_hko7_params` loads the
official ``encoder_net-*.params`` / ``forecaster_net-*.params`` arrays.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

Pair = Tuple[int, int]
IntOrPair = Union[int, Sequence[int]]

LEAKY_SLOPE = 0.2  # nowcasting/ops.py: LeakyReLU(slope=0.2) for act_type="leaky"
FLOW_CHANNELS = 32  # hidden channels of the TrajGRU flow generator (traj_rnn.py)
FLOW_KERNEL = 5


def _pair(value: IntOrPair) -> Pair:
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"expected an int or a pair, got {value!r}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def activation(x: torch.Tensor, act_type: str) -> torch.Tensor:
    """``nowcasting.ops.activation``: leaky (slope 0.2), identity, or an ``mx.sym.Activation`` type."""
    if act_type == "leaky":
        return F.leaky_relu(x, LEAKY_SLOPE)
    if act_type == "identity":
        return x
    if act_type == "tanh":
        return torch.tanh(x)
    if act_type == "relu":
        return F.relu(x)
    if act_type == "sigmoid":
        return torch.sigmoid(x)
    raise ValueError(f"unsupported act_type {act_type!r}")


def _check_act(act_type: str) -> str:
    if act_type not in {"leaky", "identity", "tanh", "relu", "sigmoid"}:
        raise ValueError(f"unsupported act_type {act_type!r}")
    return act_type


def warp(data: torch.Tensor, flows: torch.Tensor) -> torch.Tensor:
    """Warp ``data`` (B, C, H, W) by each of the L flow fields in ``flows`` (B, 2L, H, W).

    As in ``traj_rnn.flow_conv``: ``GridGenerator(data=-flow, transform_type="warp")`` followed by
    ``BilinearSampler``, i.e. output pixel ``(y, x)`` bilinearly samples ``data`` at
    ``(y - flow_y, x - flow_x)`` (channel 0 of each flow is x, channel 1 is y), with zeros outside
    the map. Returns the L warped maps concatenated along channels: (B, L * C, H, W).
    """
    if data.ndim != 4 or flows.ndim != 4 or flows.shape[1] % 2:
        raise ValueError(
            f"warp expects data (B, C, H, W) and flows (B, 2L, H, W), got {tuple(data.shape)} and {tuple(flows.shape)}"
        )
    b, c, h, w = data.shape
    links = flows.shape[1] // 2
    flows = flows.view(b, links, 2, h, w)
    xs = torch.arange(w, dtype=data.dtype, device=data.device).view(1, 1, 1, w)
    ys = torch.arange(h, dtype=data.dtype, device=data.device).view(1, 1, h, 1)
    # MXNet normalises to [-1, 1] with the corner-aligned convention: (pos) / ((size - 1) / 2) - 1.
    grid_x = (xs - flows[:, :, 0]) / ((w - 1) / 2.0) - 1.0
    grid_y = (ys - flows[:, :, 1]) / ((h - 1) / 2.0) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).view(b * links, h, w, 2)
    repeated = data.unsqueeze(1).expand(b, links, c, h, w).reshape(b * links, c, h, w)
    warped = F.grid_sample(repeated, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return warped.view(b, links * c, h, w)


class TrajGRUCell(nn.Module):
    """TrajGRU cell (paper Eq. 4; ``traj_rnn.TrajGRU``).

    A small network ``f_out(act(i2f_conv1(x) + h2f_conv1(h)))`` predicts L flow fields; the
    previous state is warped along each and mixed by the 1x1 ``h2h`` convolution. Gates:
    ``r = sigmoid(i2h_r + h2h_r)``, ``u = sigmoid(i2h_u + h2h_u)``,
    ``h' = act(i2h_h + r * h2h_h)``, ``h_t = u * h_{t-1} + (1 - u) * h'``. A cell built with
    ``input_channels=None`` takes no input (the top forecaster block) and has no ``i2h`` or
    ``i2f_conv1``.
    """

    def __init__(
        self,
        input_channels: Optional[int],
        num_filter: int,
        L: int = 5,
        i2h_kernel: IntOrPair = 3,
        i2h_pad: IntOrPair = 1,
        act_type: str = "leaky",
        init_grid: bool = True,
    ):
        super().__init__()
        if num_filter <= 0 or L <= 0:
            raise ValueError(f"num_filter and L must be positive, got {num_filter} and {L}")
        self.input_channels = input_channels
        self.num_filter = int(num_filter)
        self.L = int(L)
        self.act_type = _check_act(act_type)
        self.init_grid = bool(init_grid)
        pad = FLOW_KERNEL // 2
        if input_channels is not None:
            self.i2h = nn.Conv2d(input_channels, 3 * num_filter, _pair(i2h_kernel), padding=_pair(i2h_pad))
            self.i2f_conv1 = nn.Conv2d(input_channels, FLOW_CHANNELS, FLOW_KERNEL, padding=pad)
        self.h2f_conv1 = nn.Conv2d(num_filter, FLOW_CHANNELS, FLOW_KERNEL, padding=pad)
        self.f_out = nn.Conv2d(FLOW_CHANNELS, 2 * L, FLOW_KERNEL, padding=pad)
        self.h2h = nn.Conv2d(L * num_filter, 3 * num_filter, 1)

    def flows(self, inputs: Optional[torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        """The L flow fields ``(B, 2L, H, W)`` (x then y offset per link)."""
        f = self.h2f_conv1(state)
        if inputs is not None:
            f = self.i2f_conv1(inputs) + f
        return self.f_out(activation(f, self.act_type))

    def forward(self, inputs: Optional[torch.Tensor], state: Optional[torch.Tensor] = None) -> torch.Tensor:
        if (inputs is None) != (self.input_channels is None):
            raise ValueError("this TrajGRU cell was built " + ("without" if self.input_channels is None else "with") + " an input")
        i2h = self.i2h(inputs) if inputs is not None else None
        if state is None:
            if i2h is None:
                raise ValueError("a TrajGRU cell without input needs an explicit state")
            state = i2h.new_zeros(i2h.shape[0], self.num_filter, i2h.shape[2], i2h.shape[3])
        h2h = self.h2h(warp(state, self.flows(inputs, state)))
        h2h_r, h2h_u, h2h_h = torch.chunk(h2h, 3, dim=1)
        if i2h is not None:
            i2h_r, i2h_u, i2h_h = torch.chunk(i2h, 3, dim=1)
            reset = torch.sigmoid(i2h_r + h2h_r)
            update = torch.sigmoid(i2h_u + h2h_u)
            new_mem = activation(i2h_h + reset * h2h_h, self.act_type)
        else:
            reset = torch.sigmoid(h2h_r)
            update = torch.sigmoid(h2h_u)
            new_mem = activation(reset * h2h_h, self.act_type)
        return update * state + (1 - update) * new_mem


class EFConvGRUCell(nn.Module):
    """The ConvGRU cell of the HKO-7 baselines (``conv_rnn.ConvGRU``).

    Unlike Ballas et al. (``pyhazards.models.convgru``), the reset gate multiplies the state-to-state
    convolution (``h' = act(i2h_h + r * h2h_h)``), input and state convolutions have separate biases,
    and ``h_t = u * h_{t-1} + (1 - u) * h'``.
    """

    def __init__(
        self,
        input_channels: Optional[int],
        num_filter: int,
        h2h_kernel: IntOrPair = 3,
        h2h_dilate: IntOrPair = 1,
        i2h_kernel: IntOrPair = 3,
        i2h_pad: IntOrPair = 1,
        act_type: str = "leaky",
    ):
        super().__init__()
        kernel, dilate = _pair(h2h_kernel), _pair(h2h_dilate)
        if kernel[0] % 2 == 0 or kernel[1] % 2 == 0:
            raise ValueError(f"h2h_kernel must be odd, got {kernel}")
        self.input_channels = input_channels
        self.num_filter = int(num_filter)
        self.act_type = _check_act(act_type)
        if input_channels is not None:
            self.i2h = nn.Conv2d(input_channels, 3 * num_filter, _pair(i2h_kernel), padding=_pair(i2h_pad))
        padding = (dilate[0] * (kernel[0] - 1) // 2, dilate[1] * (kernel[1] - 1) // 2)
        self.h2h = nn.Conv2d(num_filter, 3 * num_filter, kernel, padding=padding, dilation=dilate)

    def forward(self, inputs: Optional[torch.Tensor], state: Optional[torch.Tensor] = None) -> torch.Tensor:
        if (inputs is None) != (self.input_channels is None):
            raise ValueError("this ConvGRU cell was built " + ("without" if self.input_channels is None else "with") + " an input")
        i2h = self.i2h(inputs) if inputs is not None else None
        if state is None:
            if i2h is None:
                raise ValueError("a ConvGRU cell without input needs an explicit state")
            state = i2h.new_zeros(i2h.shape[0], self.num_filter, i2h.shape[2], i2h.shape[3])
        h2h_r, h2h_u, h2h_h = torch.chunk(self.h2h(state), 3, dim=1)
        if i2h is not None:
            i2h_r, i2h_u, i2h_h = torch.chunk(i2h, 3, dim=1)
            reset = torch.sigmoid(i2h_r + h2h_r)
            update = torch.sigmoid(i2h_u + h2h_u)
            new_mem = activation(i2h_h + reset * h2h_h, self.act_type)
        else:
            reset = torch.sigmoid(h2h_r)
            update = torch.sigmoid(h2h_u)
            new_mem = activation(reset * h2h_h, self.act_type)
        return update * state + (1 - update) * new_mem


class _ConvAct(nn.Module):
    """``conv2d_act``: convolution with bias, then the CNN activation (``edown*_conv``)."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int, pad: int, act_type: str):
        super().__init__()
        self.act_type = act_type
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return activation(self.conv(x), self.act_type)


class _DeconvAct(nn.Module):
    """``deconv2d_act``: transposed convolution *without* bias, then the CNN activation (``fup*_deconv``)."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int, pad: int, act_type: str):
        super().__init__()
        self.act_type = act_type
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=stride, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return activation(self.deconv(x), self.act_type)


def _conv_out(size: int, kernel: int, stride: int, pad: int) -> int:
    return (size + 2 * pad - kernel) // stride + 1


def _deconv_out(size: int, kernel: int, stride: int, pad: int) -> int:
    return (size - 1) * stride - 2 * pad + kernel


# The architecture fields of the official configurations (HKO-7 experiments/*/configurations).
TRAJGRU_CONFIGS: Dict[str, Dict[str, object]] = {
    # experiments/hko/configurations/trajgru_55_55_33_1_64_1_192_1_192_13_13_9_b4.yml (480x480 radar maps)
    "hko7": dict(
        num_filter=(64, 192, 192),
        L=(13, 13, 9),
        h2h_kernel=((5, 5), (5, 5), (3, 3)),
        h2h_dilate=((1, 1), (1, 1), (1, 1)),
        i2h_kernel=((3, 3), (3, 3), (3, 3)),
        i2h_pad=((1, 1), (1, 1), (1, 1)),
        stack_num=(1, 1, 1),
        first_conv=(8, 7, 5, 1),
        last_deconv=(8, 7, 5, 1),
        downsample=((5, 3, 1), (3, 2, 1)),
        upsample=((5, 3, 1), (4, 2, 1)),
        num_output_frames=20,
    ),
    # experiments/movingmnist/configurations/trajgru_1_64_1_96_1_96_L13.yml (64x64 MovingMNIST++)
    "movingmnist": dict(
        num_filter=(64, 96, 96),
        L=(13, 13, 13),
        h2h_kernel=((5, 5), (5, 5), (5, 5)),
        h2h_dilate=((1, 1), (1, 1), (1, 1)),
        i2h_kernel=((3, 3), (3, 3), (3, 3)),
        i2h_pad=((1, 1), (1, 1), (1, 1)),
        stack_num=(1, 1, 1),
        first_conv=(16, 3, 1, 1),
        last_deconv=(16, 3, 1, 1),
        downsample=((3, 2, 1), (3, 2, 1)),
        upsample=((4, 2, 1), (4, 2, 1)),
        num_output_frames=10,
    ),
}


def _per_block(value, n: int, name: str) -> list:
    """Broadcast a scalar (or string) setting to the ``n`` RNN blocks, or check one entry per block."""
    if isinstance(value, (str, int)):
        return [value] * n
    value = list(value)
    if len(value) != n:
        raise ValueError(f"{name} needs {n} entries (one per RNN block), got {len(value)}")
    return value


class TrajGRU(nn.Module):
    """HKO-7 encoder-forecaster with TrajGRU (or ConvGRU) blocks, frames to frames.

    Input ``(batch, T_in, in_channels, H, W)``; output ``(batch, T_out, out_channels, H, W)``.
    Each input frame is concatenated with x/y coordinates in [-1, 1] and a channel of ones
    (``_pre_encode_frame``), embedded by ``econv1`` and encoded by RNN blocks ``ebrnn1..n`` joined by
    strided convolutions ``edown*``. The forecaster blocks ``fbrnn n..1`` start from the encoder's
    final states (stack order reversed); the top block runs without input, lower blocks take the
    up-sampled outputs (``fup*``) of the block above. The bottom block's outputs go through
    ``fdeconv1``, ``conv_final`` and the 1x1 ``out`` convolution (no output activation).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_filter: Sequence[int] = (64, 192, 192),
        layer_type: Union[str, Sequence[str]] = "TrajGRU",
        L: Union[int, Sequence[int]] = (13, 13, 9),
        h2h_kernel=((5, 5), (5, 5), (3, 3)),
        h2h_dilate=((1, 1), (1, 1), (1, 1)),
        i2h_kernel=((3, 3), (3, 3), (3, 3)),
        i2h_pad=((1, 1), (1, 1), (1, 1)),
        stack_num: Union[int, Sequence[int]] = (1, 1, 1),
        first_conv: Sequence[int] = (8, 7, 5, 1),
        last_deconv: Sequence[int] = (8, 7, 5, 1),
        downsample: Sequence[Sequence[int]] = ((5, 3, 1), (3, 2, 1)),
        upsample: Sequence[Sequence[int]] = ((5, 3, 1), (4, 2, 1)),
        rnn_act_type: str = "leaky",
        cnn_act_type: str = "leaky",
        residual_connection: bool = True,
        init_grid: bool = True,
        num_output_frames: int = 20,
    ):
        super().__init__()
        n = len(num_filter)
        if n < 1:
            raise ValueError("num_filter needs at least one RNN block")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(f"in_channels and out_channels must be positive, got {in_channels} and {out_channels}")
        if num_output_frames <= 0:
            raise ValueError(f"num_output_frames must be positive, got {num_output_frames}")
        if len(first_conv) != 4 or len(last_deconv) != 4:
            raise ValueError("first_conv and last_deconv are (num_filter, kernel, stride, pad)")
        if len(downsample) != n - 1 or len(upsample) != n - 1:
            raise ValueError(f"downsample and upsample need {n - 1} (kernel, stride, pad) entries")
        layer_types = _per_block(layer_type, n, "layer_type")
        for kind in layer_types:
            if kind not in {"TrajGRU", "ConvGRU"}:
                raise ValueError(f"layer_type must be 'TrajGRU' or 'ConvGRU', got {kind!r}")
        links = _per_block(L, n, "L")
        h2h_kernels = _per_block(h2h_kernel, n, "h2h_kernel")
        h2h_dilates = _per_block(h2h_dilate, n, "h2h_dilate")
        i2h_kernels = _per_block(i2h_kernel, n, "i2h_kernel")
        i2h_pads = _per_block(i2h_pad, n, "i2h_pad")
        stacks = [int(s) for s in _per_block(stack_num, n, "stack_num")]
        if min(stacks) < 1:
            raise ValueError(f"stack_num entries must be positive, got {stacks}")
        for kernel, pad in zip(i2h_kernels, i2h_pads):
            kernel, pad = _pair(kernel), _pair(pad)
            if kernel[0] != 2 * pad[0] + 1 or kernel[1] != 2 * pad[1] + 1:
                # The encoder-forecaster wiring assumes the state has the size of the block input.
                raise ValueError(f"i2h_kernel {kernel} with i2h_pad {pad} does not preserve the feature-map size")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_filter = [int(f) for f in num_filter]
        self.layer_type = layer_types
        self.num_blocks = n
        self.first_conv = tuple(int(v) for v in first_conv)
        self.last_deconv = tuple(int(v) for v in last_deconv)
        self.downsample = [tuple(int(v) for v in d) for d in downsample]
        self.upsample = [tuple(int(v) for v in u) for u in upsample]
        self.rnn_act_type = _check_act(rnn_act_type)
        self.cnn_act_type = _check_act(cnn_act_type)
        self.residual_connection = bool(residual_connection)
        self.num_output_frames = int(num_output_frames)

        def make_cell(i: int, input_channels: Optional[int]) -> nn.Module:
            if layer_types[i] == "TrajGRU":
                return TrajGRUCell(
                    input_channels, self.num_filter[i], L=int(links[i]), i2h_kernel=i2h_kernels[i],
                    i2h_pad=i2h_pads[i], act_type=rnn_act_type, init_grid=init_grid,
                )
            return EFConvGRUCell(
                input_channels, self.num_filter[i], h2h_kernel=h2h_kernels[i], h2h_dilate=h2h_dilates[i],
                i2h_kernel=i2h_kernels[i], i2h_pad=i2h_pads[i], act_type=rnn_act_type,
            )

        def make_block(i: int, input_channels: Optional[int]) -> nn.ModuleList:
            # Stacked cells after the first take the previous cell's output (num_filter[i] channels).
            return nn.ModuleList(
                make_cell(i, input_channels if j == 0 else self.num_filter[i]) for j in range(stacks[i])
            )

        # Encoder: 3 coordinate channels (x, y, ones) are appended to every frame.
        c, k, s, p = self.first_conv
        self.econv1 = nn.Conv2d(self.in_channels + 3, c, k, stride=s, padding=p)
        for i in range(n):
            setattr(self, f"ebrnn{i + 1}", make_block(i, c if i == 0 else self.num_filter[i]))
            if i < n - 1:
                k, s, p = self.downsample[i]
                setattr(self, f"edown{i + 1}", _ConvAct(self.num_filter[i], self.num_filter[i + 1], k, s, p, cnn_act_type))
        # Forecaster: the top block has no input; block i < n - 1 takes fup{i + 1}'s output.
        for i in reversed(range(n)):
            setattr(self, f"fbrnn{i + 1}", make_block(i, None if i == n - 1 else self.num_filter[i + 1]))
            if i > 0:
                k, s, p = self.upsample[i - 1]
                setattr(self, f"fup{i}", _DeconvAct(self.num_filter[i], self.num_filter[i], k, s, p, cnn_act_type))
        c, k, s, p = self.last_deconv
        self.fdeconv1 = nn.ConvTranspose2d(self.num_filter[0], c, k, stride=s, padding=p, bias=False)
        self.conv_final = nn.Conv2d(c, c, 3, padding=1)
        self.out = nn.Conv2d(c, self.out_channels, 1)
        self.reset_parameters()

    def reset_parameters(self, slope: float = 0.2) -> None:
        """Reference initialisation: ``mx.init.MSRAPrelu(slope=0.2)`` for every weight (Gaussian,
        ``std = sqrt(2 / (1 + slope**2) / fan_avg)``), zero biases, and (``init_grid``) zeros for
        the flow output layer ``f_out`` so that every link starts as the identity warp."""
        magnitude = 2.0 / (1.0 + slope ** 2)
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                weight = module.weight
                receptive = weight[0][0].numel()
                fan_avg = (weight.shape[0] + weight.shape[1]) * receptive / 2.0
                with torch.no_grad():
                    weight.normal_(0.0, math.sqrt(magnitude / fan_avg))
                    if module.bias is not None:
                        module.bias.zero_()
        for module in self.modules():
            if isinstance(module, TrajGRUCell) and module.init_grid:
                with torch.no_grad():
                    module.f_out.weight.zero_()
                    module.f_out.bias.zero_()

    def _sizes(self, size: int) -> List[int]:
        _, k, s, p = self.first_conv
        sizes = [_conv_out(size, k, s, p)]
        for k, s, p in self.downsample:
            sizes.append(_conv_out(sizes[-1], k, s, p))
        return sizes

    def check_input_size(self, height: int, width: int) -> None:
        """Raise ``ValueError`` unless the decoder restores every encoder feature map and the input size."""
        for size in (height, width):
            sizes = self._sizes(size)
            if min(sizes) < 1:
                raise ValueError(f"input size {size} is too small for this configuration")
            for i in range(1, self.num_blocks):
                k, s, p = self.upsample[i - 1]
                if _deconv_out(sizes[i], k, s, p) != sizes[i - 1]:
                    raise ValueError(
                        f"input size {(height, width)}: up-sampling block {i + 1} gives "
                        f"{_deconv_out(sizes[i], k, s, p)} pixels, block {i} expects {sizes[i - 1]} "
                        f"(encoder feature sizes {sizes})."
                    )
            _, k, s, p = self.last_deconv
            if _deconv_out(sizes[0], k, s, p) != size:
                raise ValueError(
                    f"input size {(height, width)}: the output layer gives {_deconv_out(sizes[0], k, s, p)} pixels."
                )

    def _coordinates(self, n: int, height: int, width: int, like: torch.Tensor) -> torch.Tensor:
        x = torch.arange(width, dtype=like.dtype, device=like.device) / float(width - 1) * 2.0 - 1.0
        y = torch.arange(height, dtype=like.dtype, device=like.device) / float(height - 1) * 2.0 - 1.0
        grid = torch.stack(
            (x.view(1, width).expand(height, width), y.view(height, 1).expand(height, width), like.new_ones(height, width))
        )
        return grid.unsqueeze(0).expand(n, 3, height, width)

    def _unroll(
        self,
        block: nn.ModuleList,
        inputs: Optional[List[torch.Tensor]],
        length: int,
        begin_states: Optional[Sequence[torch.Tensor]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """``BaseStackRNN.unroll``: returns the last cell's outputs and every cell's final state."""
        finals: List[torch.Tensor] = []
        outputs: List[torch.Tensor] = []
        for j, cell in enumerate(block):
            state = begin_states[j] if begin_states is not None else None
            outputs = []
            for t in range(length):
                state = cell(inputs[t] if inputs is not None else None, state)
                outputs.append(state)
            if self.residual_connection and j > 0:
                outputs = [o + x for o, x in zip(outputs, inputs)]
            inputs = outputs
            finals.append(state)
        return outputs, finals

    def encode(
        self, x: torch.Tensor, initial_states: Optional[Sequence[Sequence[torch.Tensor]]] = None
    ) -> List[List[torch.Tensor]]:
        """Final states of every encoder block (one tensor per stacked cell)."""
        self._check_input(x)
        b, t, c, h, w = x.shape
        frames = x.transpose(0, 1).reshape(t * b, c, h, w)
        frames = torch.cat((frames, self._coordinates(t * b, h, w, x)), dim=1)
        seq = list(activation(self.econv1(frames), self.cnn_act_type).split(b, dim=0))
        states: List[List[torch.Tensor]] = []
        for i in range(self.num_blocks):
            begin = initial_states[i] if initial_states is not None else None
            outputs, finals = self._unroll(getattr(self, f"ebrnn{i + 1}"), seq, t, begin)
            states.append(finals)
            if i < self.num_blocks - 1:
                seq = list(getattr(self, f"edown{i + 1}")(torch.cat(outputs, dim=0)).split(b, dim=0))
        return states

    def forecast(self, states: Sequence[Sequence[torch.Tensor]], num_output_frames: int) -> torch.Tensor:
        """Run the forecaster from encoder states; returns ``(batch, T_out, out_channels, H, W)``."""
        inputs: Optional[List[torch.Tensor]] = None
        outputs: List[torch.Tensor] = []
        b = states[0][0].shape[0]
        for i in reversed(range(self.num_blocks)):
            outputs, _ = self._unroll(getattr(self, f"fbrnn{i + 1}"), inputs, num_output_frames, list(states[i])[::-1])
            if i > 0:
                inputs = list(getattr(self, f"fup{i}")(torch.cat(outputs, dim=0)).split(b, dim=0))
        y = activation(self.fdeconv1(torch.cat(outputs, dim=0)), self.cnn_act_type)
        y = self.out(activation(self.conv_final(y), self.cnn_act_type))
        return y.view(num_output_frames, b, *y.shape[1:]).transpose(0, 1)

    def _check_input(self, x: torch.Tensor) -> None:
        if x.ndim != 5:
            raise ValueError(
                f"TrajGRU expects input shape (batch, time, channels, height, width), got {tuple(x.shape)}."
            )
        if x.shape[2] != self.in_channels:
            raise ValueError(f"TrajGRU expected {self.in_channels} input channels, got input of shape {tuple(x.shape)}.")
        if x.shape[1] < 1:
            raise ValueError("TrajGRU needs at least one input frame.")
        self.check_input_size(x.shape[3], x.shape[4])

    def forward(
        self,
        x: torch.Tensor,
        num_output_frames: Optional[int] = None,
        initial_states: Optional[Sequence[Sequence[torch.Tensor]]] = None,
        return_states: bool = False,
    ):
        """Forecast ``num_output_frames`` frames (default: the configuration's ``OUT_LEN``).

        ``initial_states`` (per block, per stacked cell) replaces the zero initial encoder state;
        ``return_states=True`` also returns the final encoder states, which the official training
        loop carries over between batches (see the model card).
        """
        n_out = self.num_output_frames if num_output_frames is None else int(num_output_frames)
        if n_out <= 0:
            raise ValueError(f"num_output_frames must be positive, got {n_out}")
        states = self.encode(x, initial_states)
        prediction = self.forecast(states, n_out)
        return (prediction, states) if return_states else prediction


class TrajGRUSegmenter(TrajGRU):
    """PyHazards adaptation for next-step fire masks: ``(batch, T, C, H, W)`` -> ``(batch, out_channels, H, W)``.

    The encoder-forecaster predicts a single frame; its (linear) output is returned as logits.
    Parameter names are those of :class:`TrajGRU` (no prefix).
    """

    def __init__(self, **kwargs):
        kwargs.pop("num_output_frames", None)
        super().__init__(num_output_frames=1, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.forecast(self.encode(x), 1)[:, 0]


def load_hko7_params(model: TrajGRU, params: Mapping[str, object], strict: bool = True) -> None:
    """Load official HKO-7 parameters (``{"arg:<name>": array}`` or ``{"<name>": array}``, as read by
    ``mx.nd.load`` from ``encoder_net-*.params`` and ``forecaster_net-*.params``) into ``model``."""
    arrays = {}
    for name, value in params.items():
        name = name.split(":", 1)[1] if ":" in name else name
        array = value.asnumpy() if hasattr(value, "asnumpy") else value
        arrays[name] = torch.as_tensor(array)
    state = {}
    for key, tensor in model.state_dict().items():
        name = key.replace(".", "_")
        if name not in arrays:
            if strict:
                raise KeyError(f"parameter {name!r} (for {key!r}) is missing from the HKO-7 checkpoint")
            continue
        value = arrays.pop(name)
        if tuple(value.shape) != tuple(tensor.shape):
            raise ValueError(f"{name}: checkpoint shape {tuple(value.shape)} != model shape {tuple(tensor.shape)}")
        state[key] = value.to(tensor.dtype)
    if strict and arrays:
        raise KeyError(f"unexpected HKO-7 parameters: {sorted(arrays)}")
    model.load_state_dict(state, strict=strict)


def trajgru_builder(
    task: str,
    config: str = "hko7",
    in_channels: int = 1,
    out_channels: int = 1,
    layer_type: Union[str, Sequence[str]] = "TrajGRU",
    num_output_frames: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """Build the HKO-7 TrajGRU encoder-forecaster.

    ``config``: ``"hko7"`` (HKO-7 benchmark model, 480x480 input, 5 -> 20 frames) or
    ``"movingmnist"`` (TrajGRU-L13 of the MovingMNIST++ experiments, 64x64, 10 -> 10). Any
    architecture field (``num_filter``, ``L``, ``h2h_kernel``, ``h2h_dilate``, ``i2h_kernel``,
    ``i2h_pad``, ``stack_num``, ``first_conv``, ``last_deconv``, ``downsample``, ``upsample``,
    activations) can be overridden; ``layer_type="ConvGRU"`` gives the paper's ConvGRU baseline.
    ``task="forecasting"``: :class:`TrajGRU`; ``task="segmentation"``: :class:`TrajGRUSegmenter`.
    """
    task = task.lower()
    if config not in TRAJGRU_CONFIGS:
        raise ValueError(f"config must be one of {sorted(TRAJGRU_CONFIGS)}, got {config!r}")
    allowed = {
        "num_filter", "L", "h2h_kernel", "h2h_dilate", "i2h_kernel", "i2h_pad", "stack_num", "first_conv",
        "last_deconv", "downsample", "upsample", "rnn_act_type", "cnn_act_type", "residual_connection", "init_grid",
    }
    settings = dict(TRAJGRU_CONFIGS[config])
    settings.update({key: value for key, value in kwargs.items() if key in allowed})
    frames = settings.pop("num_output_frames")
    if num_output_frames is not None:
        frames = num_output_frames
    common = dict(in_channels=in_channels, out_channels=out_channels, layer_type=layer_type, **settings)
    if task == "forecasting":
        return TrajGRU(num_output_frames=frames, **common)
    if task == "segmentation":
        return TrajGRUSegmenter(**common)
    raise ValueError(f"trajgru supports task='forecasting' or 'segmentation', got {task!r}.")


__all__ = [
    "EFConvGRUCell",
    "TRAJGRU_CONFIGS",
    "TrajGRU",
    "TrajGRUCell",
    "TrajGRUSegmenter",
    "load_hko7_params",
    "trajgru_builder",
    "warp",
]
