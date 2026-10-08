"""NeuralHydrology LSTM (``CudaLSTM``): the regional rainfall-runoff LSTM with static catchment inputs.

Port of NeuralHydrology ``neuralhydrology/modelzoo/cudalstm.py`` (``CudaLSTM``), ``head.py``
(``Regression``) and the input concatenation of ``inputlayer.py`` at commit ea94a40
(https://github.com/neuralhydrology/neuralhydrology, BSD-3-Clause, Copyright (c) 2021, NeuralHydrology).

The model is a single-layer ``nn.LSTM`` over daily inputs, where the static catchment attributes ``x_s``
are concatenated to the dynamic inputs ``x_d`` at every time step (Kratzert et al., HESS 2019, "LSTM
with static inputs"; the LSTM rainfall-runoff model itself is Kratzert et al., HESS 2018), followed by
dropout and a linear regression head applied to every time step. Parameter names (``lstm.*``,
``head.net.0.*``), initialisation order and the forget-gate bias initialisation of the reference are kept,
so NeuralHydrology state dicts load with ``strict=True`` and the same seed gives the same weights.

:func:`convert_kratzert2019_state_dict` maps the official checkpoints of Kratzert et al. (2019) (written by
``papercode/lstm.py`` of kratzert/ealstm_regional_modeling, Apache-2.0: one bias vector, gate order f, i,
o, g, transposed weights) onto this layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn


def streamflow_inputs(batch: Any, n_dynamic: int, n_static: int, model_name: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """``x_d`` (batch, time, n_dynamic) and ``x_s`` (batch, n_static) from a streamflow batch mapping."""
    if not isinstance(batch, Mapping) or "x_d" not in batch:
        raise ValueError(
            f"{model_name} expects a mapping with 'x_d' of shape (batch, time, n_dynamic) and, with static "
            "attributes, 'x_s' of shape (batch, n_static)."
        )
    x_d = batch["x_d"]
    if not isinstance(x_d, torch.Tensor) or x_d.ndim != 3 or x_d.shape[-1] != n_dynamic:
        shape = tuple(x_d.shape) if isinstance(x_d, torch.Tensor) else type(x_d).__name__
        raise ValueError(f"{model_name} expects x_d of shape (batch, time, {n_dynamic}), got {shape}.")
    x_s = batch.get("x_s")
    if n_static:
        if not isinstance(x_s, torch.Tensor) or x_s.ndim != 2 or tuple(x_s.shape) != (x_d.shape[0], n_static):
            shape = tuple(x_s.shape) if isinstance(x_s, torch.Tensor) else None
            raise ValueError(f"{model_name} expects x_s of shape ({x_d.shape[0]}, {n_static}), got {shape}.")
    elif x_s is not None and x_s.numel():
        raise ValueError(f"{model_name} was built without static attributes (n_static=0) but received x_s.")
    else:
        x_s = None
    return x_d, x_s


class Regression(nn.Module):
    """NeuralHydrology's single-layer regression head (``head.Regression``)."""

    def __init__(self, n_in: int, n_out: int, activation: str = "linear"):
        super().__init__()
        layers = [nn.Linear(n_in, n_out)]
        if activation != "linear":
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "softplus":
                layers.append(nn.Softplus())
            else:
                raise ValueError(f"output_activation must be 'linear', 'relu' or 'softplus', got {activation!r}.")
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"y_hat": self.net(x)}


class NeuralHydrologyLSTM(nn.Module):
    """``CudaLSTM`` of NeuralHydrology with a regression head.

    ``forward({"x_d": (B, T, n_dynamic), "x_s": (B, n_static)})`` returns, like the reference,
    ``y_hat`` (B, T, n_targets), ``lstm_output`` (B, T, hidden_size), ``h_n`` and ``c_n`` (B, 1, hidden_size).
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
        for label, value in (("n_dynamic", n_dynamic), ("hidden_size", hidden_size), ("n_targets", n_targets)):
            if int(value) <= 0:
                raise ValueError(f"{label} must be positive, got {value}.")
        if int(n_static) < 0:
            raise ValueError(f"n_static must be >= 0, got {n_static}.")
        if not 0.0 <= output_dropout < 1.0:
            raise ValueError(f"output_dropout must be in [0, 1), got {output_dropout}.")
        self.n_dynamic = int(n_dynamic)
        self.n_static = int(n_static)
        self.hidden_size = int(hidden_size)
        self.initial_forget_bias = initial_forget_bias
        # Reference order: InputLayer (no parameters without embedding networks), LSTM, dropout, head.
        self.lstm = nn.LSTM(input_size=self.n_dynamic + self.n_static, hidden_size=self.hidden_size)
        self.dropout = nn.Dropout(p=output_dropout)
        self.head = Regression(n_in=self.hidden_size, n_out=int(n_targets), activation=output_activation)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[self.hidden_size: 2 * self.hidden_size] = self.initial_forget_bias

    def forward(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x_d, x_s = streamflow_inputs(batch, self.n_dynamic, self.n_static, "NeuralHydrologyLSTM")
        x = x_d.transpose(0, 1)  # [seq_length, batch_size, n_features], as the reference InputLayer
        if x_s is not None:
            x = torch.cat([x, x_s.unsqueeze(0).repeat(x.shape[0], 1, 1)], dim=-1)
        lstm_output, (h_n, c_n) = self.lstm(input=x)
        lstm_output = lstm_output.transpose(0, 1)
        h_n = h_n.transpose(0, 1)
        c_n = c_n.transpose(0, 1)
        pred = {"lstm_output": lstm_output, "h_n": h_n, "c_n": c_n}
        pred.update(self.head(self.dropout(lstm_output)))
        return pred


def _reorder_gates(tensor: torch.Tensor, hidden_size: int, order: Tuple[int, ...]) -> torch.Tensor:
    chunks = tensor.split(hidden_size, dim=0)
    return torch.cat([chunks[i] for i in order], dim=0)


def convert_kratzert2019_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Official Kratzert et al. (2019) checkpoint -> state dict of this module or the EA-LSTM port.

    LSTM checkpoints (``lstm.weight_ih`` (in, 4H), ``lstm.weight_hh`` (H, 4H), ``lstm.bias`` (4H), gates
    f, i, o, g) become ``nn.LSTM`` tensors (gates i, f, g, o; the single bias goes to ``bias_ih_l0`` and
    ``bias_hh_l0`` is zero). EA-LSTM checkpoints (with ``lstm.weight_sh``) keep their gate layout (f, o, g,
    as in NeuralHydrology); ``weight_sh`` (S, H) becomes ``input_gate.weight`` (H, S). ``fc.*`` becomes
    ``head.net.0.*``.
    """
    state = dict(state_dict)
    out: Dict[str, torch.Tensor] = {
        "head.net.0.weight": state["fc.weight"],
        "head.net.0.bias": state["fc.bias"],
    }
    if "lstm.weight_sh" in state:
        out.update(
            {
                "input_gate.weight": state["lstm.weight_sh"].t().contiguous(),
                "input_gate.bias": state["lstm.bias_s"],
                "dynamic_gates.weight_ih": state["lstm.weight_ih"],
                "dynamic_gates.weight_hh": state["lstm.weight_hh"],
                "dynamic_gates.bias": state["lstm.bias"],
            }
        )
        return out
    hidden = state["lstm.weight_hh"].shape[0]
    order = (1, 0, 3, 2)  # (f, i, o, g) -> (i, f, g, o)
    out.update(
        {
            "lstm.weight_ih_l0": _reorder_gates(state["lstm.weight_ih"].t(), hidden, order).contiguous(),
            "lstm.weight_hh_l0": _reorder_gates(state["lstm.weight_hh"].t(), hidden, order).contiguous(),
            "lstm.bias_ih_l0": _reorder_gates(state["lstm.bias"], hidden, order).contiguous(),
            "lstm.bias_hh_l0": torch.zeros_like(state["lstm.bias"]),
        }
    )
    return out


def load_kratzert2019_state_dict(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """Read a ``model_epoch30.pt`` of the official 2019 runs (HydroShare, CC BY 4.0) and convert it."""
    return convert_kratzert2019_state_dict(torch.load(path, map_location="cpu", weights_only=True))


def neuralhydrology_lstm_builder(
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
    """Build the LSTM; defaults are the Kratzert et al. (2019) "LSTM with static inputs" configuration.

    ``checkpoint`` is an official 2019 ``model_epoch30.pt`` (LSTM with or without static inputs),
    converted with :func:`convert_kratzert2019_state_dict` and loaded with ``strict=True``.
    """
    kwargs.pop("name", None)
    if kwargs:
        raise ValueError(f"Unexpected arguments for neuralhydrology_lstm: {sorted(kwargs)}")
    if task.lower() not in ("regression", "streamflow"):
        raise ValueError(f"neuralhydrology_lstm supports task='regression' (streamflow), got {task!r}.")
    model = NeuralHydrologyLSTM(
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


__all__ = [
    "NeuralHydrologyLSTM",
    "Regression",
    "convert_kratzert2019_state_dict",
    "load_kratzert2019_state_dict",
    "neuralhydrology_lstm_builder",
    "streamflow_inputs",
]
