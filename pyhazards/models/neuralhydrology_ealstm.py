"""Entity-Aware LSTM (EA-LSTM) of Kratzert et al. (HESS 2019), as implemented in NeuralHydrology.

Port of NeuralHydrology ``neuralhydrology/modelzoo/ealstm.py`` (``EALSTM``, ``_DynamicGates``) and
``head.py`` (``Regression``) at commit ea94a40 (https://github.com/neuralhydrology/neuralhydrology,
BSD-3-Clause, Copyright (c) 2021, NeuralHydrology), which is the same network as the paper code
(kratzert/ealstm_regional_modeling, ``papercode/ealstm.py``, Apache-2.0).

The input gate is computed once from the static catchment attributes, ``i = sigmoid(W_s x_s + b_s)``,
and used at every step; the forget, output and cell gates come from the dynamic inputs and the previous
hidden state (``f, o, g`` chunks of ``h W_hh + x W_ih + b``):
``c_t = sigmoid(f) * c_{t-1} + i * tanh(g)``, ``h_t = sigmoid(o) * tanh(c_t)``.
Parameter names (``input_gate.*``, ``dynamic_gates.weight_ih/weight_hh/bias``, ``head.net.0.*``),
initialisation (orthogonal ``weight_ih``, identity-tiled ``weight_hh``, zero bias with the forget slice
set to ``initial_forget_bias``) and the order of creation follow the reference. Official Kratzert et al.
(2019) checkpoints load after :func:`~pyhazards.models.neuralhydrology_lstm.convert_kratzert2019_state_dict`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn

from .neuralhydrology_lstm import Regression, load_kratzert2019_state_dict, streamflow_inputs


class _DynamicGates(nn.Module):
    """Forget, output and cell gates of the EA-LSTM (``_DynamicGates`` of the reference)."""

    def __init__(self, input_size: int, hidden_size: int, initial_forget_bias: Optional[float]):
        super().__init__()
        self.hidden_size = hidden_size
        self.initial_forget_bias = initial_forget_bias
        self.weight_ih = nn.Parameter(torch.empty(input_size, 3 * hidden_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, 3 * hidden_size))
        self.bias = nn.Parameter(torch.empty(3 * hidden_size))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.orthogonal_(self.weight_ih.data)
        self.weight_hh.data = torch.eye(self.hidden_size).repeat(1, 3)
        nn.init.constant_(self.bias.data, val=0)
        if self.initial_forget_bias is not None:
            self.bias.data[: self.hidden_size] = self.initial_forget_bias

    def forward(self, h: torch.Tensor, x_d: torch.Tensor) -> torch.Tensor:
        return h @ self.weight_hh + x_d @ self.weight_ih + self.bias


class NeuralHydrologyEALSTM(nn.Module):
    """EA-LSTM with a regression head.

    ``forward({"x_d": (B, T, n_dynamic), "x_s": (B, n_static)})`` returns ``y_hat`` (B, T, n_targets) and
    the hidden and cell states of every step, ``h_n`` and ``c_n`` (B, T, hidden_size), like the reference.
    """

    def __init__(
        self,
        n_dynamic: int = 5,
        n_static: int = 27,
        hidden_size: int = 256,
        n_targets: int = 1,
        output_dropout: float = 0.4,
        initial_forget_bias: Optional[float] = 5.0,
        output_activation: str = "linear",
    ):
        super().__init__()
        for label, value in (
            ("n_dynamic", n_dynamic),
            ("n_static", n_static),
            ("hidden_size", hidden_size),
            ("n_targets", n_targets),
        ):
            if int(value) <= 0:
                raise ValueError(f"{label} must be positive (the EA-LSTM input gate needs static inputs), got {value}.")
        if not 0.0 <= output_dropout < 1.0:
            raise ValueError(f"output_dropout must be in [0, 1), got {output_dropout}.")
        self.n_dynamic = int(n_dynamic)
        self.n_static = int(n_static)
        self._hidden_size = int(hidden_size)
        # Reference order: InputLayer (no parameters), input gate, dynamic gates, dropout, head.
        self.input_gate = nn.Linear(self.n_static, self._hidden_size)
        self.dynamic_gates = _DynamicGates(self.n_dynamic, self._hidden_size, initial_forget_bias)
        self.dropout = nn.Dropout(p=output_dropout)
        self.head = Regression(n_in=self._hidden_size, n_out=int(n_targets), activation=output_activation)

    def _cell(self, x: torch.Tensor, i: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor]):
        h_0, c_0 = states
        gates = self.dynamic_gates(h_0, x)
        f, o, g = gates.chunk(3, 1)
        c_1 = torch.sigmoid(f) * c_0 + i * torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        return h_1, c_1

    def forward(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x_d, x_s = streamflow_inputs(batch, self.n_dynamic, self.n_static, "NeuralHydrologyEALSTM")
        x_d = x_d.transpose(0, 1)  # [seq_length, batch_size, n_features]
        h_t = x_d.new_zeros(x_d.shape[1], self._hidden_size)
        c_t = x_d.new_zeros(x_d.shape[1], self._hidden_size)
        h_n, c_n = [], []
        i = torch.sigmoid(self.input_gate(x_s))  # computed once: the inputs are static
        for x_dt in x_d:
            h_t, c_t = self._cell(x_dt, i, (h_t, c_t))
            h_n.append(h_t)
            c_n.append(c_t)
        h_n = torch.stack(h_n, 0).transpose(0, 1)
        c_n = torch.stack(c_n, 0).transpose(0, 1)
        pred = {"h_n": h_n, "c_n": c_n}
        pred.update(self.head(self.dropout(h_n)))
        return pred


def neuralhydrology_ealstm_builder(
    task: str,
    n_dynamic: int = 5,
    n_static: int = 27,
    hidden_size: int = 256,
    n_targets: int = 1,
    output_dropout: float = 0.4,
    initial_forget_bias: Optional[float] = 5.0,
    output_activation: str = "linear",
    checkpoint: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> nn.Module:
    """Build the EA-LSTM; defaults are the Kratzert et al. (2019) configuration (5 dynamic, 27 static, 256).

    ``checkpoint`` is an official 2019 EA-LSTM ``model_epoch30.pt``, loaded with ``strict=True``.
    """
    kwargs.pop("name", None)
    if kwargs:
        raise ValueError(f"Unexpected arguments for neuralhydrology_ealstm: {sorted(kwargs)}")
    if task.lower() not in ("regression", "streamflow"):
        raise ValueError(f"neuralhydrology_ealstm supports task='regression' (streamflow), got {task!r}.")
    model = NeuralHydrologyEALSTM(
        n_dynamic=n_dynamic,
        n_static=n_static,
        hidden_size=hidden_size,
        n_targets=n_targets,
        output_dropout=output_dropout,
        initial_forget_bias=initial_forget_bias,
        output_activation=output_activation,
    )
    if checkpoint is not None:
        model.load_state_dict(load_kratzert2019_state_dict(checkpoint), strict=True)
    return model


__all__ = ["NeuralHydrologyEALSTM", "neuralhydrology_ealstm_builder"]
