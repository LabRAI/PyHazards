"""NeuralHydrology LSTM and EA-LSTM without the reference code: sizes, outputs, 2019 checkpoints, validation."""

from __future__ import annotations

import pytest
import torch

from pyhazards.models import build_model
from pyhazards.models.neuralhydrology_lstm import convert_kratzert2019_state_dict


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize(
    "name, kwargs, expected",
    [
        ("neuralhydrology_lstm", {}, 297_217),  # Kratzert et al. (2019) LSTM with static inputs
        ("neuralhydrology_lstm", {"n_static": 0}, 269_569),  # ... without static inputs
        ("neuralhydrology_ealstm", {}, 208_641),  # EA-LSTM
    ],
)
def test_parameter_counts(name, kwargs, expected):
    assert _n_params(build_model(name, task="regression", **kwargs)) == expected


def test_lstm_outputs_and_forget_bias():
    model = build_model("neuralhydrology_lstm", task="regression", hidden_size=16).eval()
    assert torch.all(model.lstm.bias_hh_l0[16:32] == 5.0)
    out = model({"x_d": torch.randn(3, 40, 5), "x_s": torch.randn(3, 27)})
    assert out["y_hat"].shape == (3, 40, 1)
    assert out["lstm_output"].shape == (3, 40, 16)
    assert out["h_n"].shape == out["c_n"].shape == (3, 1, 16)
    torch.testing.assert_close(out["h_n"][:, 0], out["lstm_output"][:, -1])
    assert list(model.state_dict()) == ["lstm.weight_ih_l0", "lstm.weight_hh_l0", "lstm.bias_ih_l0", "lstm.bias_hh_l0",
                                        "head.net.0.weight", "head.net.0.bias"]


def test_ealstm_input_gate_uses_only_static_inputs():
    model = build_model("neuralhydrology_ealstm", task="regression", hidden_size=8, output_dropout=0.0).eval()
    assert torch.equal(model.dynamic_gates.weight_hh, torch.eye(8).repeat(1, 3))
    assert torch.all(model.dynamic_gates.bias[:8] == 5.0) and torch.all(model.dynamic_gates.bias[8:] == 0)
    x_d, x_s = torch.randn(2, 25, 5), torch.randn(2, 27)
    out = model({"x_d": x_d, "x_s": x_s})
    assert out["y_hat"].shape == (2, 25, 1) and out["h_n"].shape == out["c_n"].shape == (2, 25, 8)
    # Hand-written recurrence of Kratzert et al. (2019), Eqs. 7-10.
    i = torch.sigmoid(x_s @ model.input_gate.weight.t() + model.input_gate.bias)
    h = c = torch.zeros(2, 8)
    for t in range(25):
        gates = h @ model.dynamic_gates.weight_hh + x_d[:, t] @ model.dynamic_gates.weight_ih + model.dynamic_gates.bias
        f, o, g = gates.chunk(3, 1)
        c = torch.sigmoid(f) * c + i * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
    torch.testing.assert_close(out["h_n"][:, -1], h)
    torch.testing.assert_close(out["y_hat"][:, -1], model.head.net(h))


def test_kratzert2019_lstm_checkpoint_conversion():
    """A checkpoint in the 2019 layout (single bias, gates f, i, o, g) gives the 2019 recurrence."""
    torch.manual_seed(0)
    hidden, n_in = 6, 32
    state = {
        "lstm.weight_ih": torch.randn(n_in, 4 * hidden) * 0.3,
        "lstm.weight_hh": torch.randn(hidden, 4 * hidden) * 0.3,
        "lstm.bias": torch.randn(4 * hidden),
        "fc.weight": torch.randn(1, hidden),
        "fc.bias": torch.randn(1),
    }
    model = build_model("neuralhydrology_lstm", task="regression", hidden_size=hidden).eval()
    model.load_state_dict(convert_kratzert2019_state_dict(state), strict=True)
    x_d, x_s = torch.randn(3, 12, 5), torch.randn(3, 27)
    x = torch.cat([x_d, x_s.unsqueeze(1).repeat(1, 12, 1)], dim=-1)
    h = c = torch.zeros(3, hidden)
    for t in range(12):  # papercode/lstm.py
        gates = torch.addmm(state["lstm.bias"].expand(3, -1), h, state["lstm.weight_hh"]) + x[:, t] @ state["lstm.weight_ih"]
        f, i, o, g = gates.chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
    expected = h @ state["fc.weight"].t() + state["fc.bias"]
    torch.testing.assert_close(model({"x_d": x_d, "x_s": x_s})["y_hat"][:, -1], expected, rtol=1e-5, atol=1e-6)


def test_kratzert2019_ealstm_checkpoint_conversion():
    torch.manual_seed(1)
    hidden = 4
    state = {
        "lstm.weight_ih": torch.randn(5, 3 * hidden),
        "lstm.weight_hh": torch.randn(hidden, 3 * hidden),
        "lstm.weight_sh": torch.randn(27, hidden),
        "lstm.bias": torch.randn(3 * hidden),
        "lstm.bias_s": torch.randn(hidden),
        "fc.weight": torch.randn(1, hidden),
        "fc.bias": torch.randn(1),
    }
    model = build_model("neuralhydrology_ealstm", task="regression", hidden_size=hidden)
    model.load_state_dict(convert_kratzert2019_state_dict(state), strict=True)
    x_s = torch.randn(2, 27)
    torch.testing.assert_close(model.input_gate(x_s), x_s @ state["lstm.weight_sh"] + state["lstm.bias_s"])


@pytest.mark.parametrize("name", ["neuralhydrology_lstm", "neuralhydrology_ealstm"])
def test_bad_inputs_raise(name):
    model = build_model(name, task="regression", hidden_size=8)
    with pytest.raises(ValueError, match="shape"):
        model({"x_d": torch.randn(2, 10, 4), "x_s": torch.randn(2, 27)})
    with pytest.raises(ValueError, match="x_s"):
        model({"x_d": torch.randn(2, 10, 5)})
    with pytest.raises(ValueError, match="mapping"):
        model(torch.randn(2, 10, 5))
    with pytest.raises(ValueError):
        build_model(name, task="classification")


def test_ealstm_needs_static_inputs():
    with pytest.raises(ValueError, match="static"):
        build_model("neuralhydrology_ealstm", task="regression", n_static=0)
