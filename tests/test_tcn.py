"""TCN: configurations, shapes, causality, state-dict layout and input validation.

Numerical equivalence with locuslab/TCN lives in tests/oracle/test_tcn_oracle.py.
"""

from __future__ import annotations

import copy

import pytest
import torch

from pyhazards.models import build_model
from pyhazards.models.tcn import TCN, TemporalConvNet, to_official_state_dict


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, 96_001),  # official adding-problem script defaults (8 x 30, kernel 7)
        (dict(hidden_dim=24, num_levels=8, kernel_size=8), 70_369),  # paper Table 2, adding T=600 (~70K)
        (dict(input_dim=1, out_dim=10, hidden_dim=25, num_levels=8, kernel_size=7), 66_910),  # seq. MNIST
    ],
)
def test_parameter_counts(kwargs, expected):
    assert _n_params(build_model("tcn", task="regression", **kwargs)) == expected


def test_forward_shapes():
    model = build_model("tcn", task="classification", input_dim=5, out_dim=3, hidden_dim=8, num_levels=3, kernel_size=2).eval()
    x = torch.randn(4, 6, 5)  # (batch, time, features)
    with torch.no_grad():
        assert model(x).shape == (4, 3)
    model.readout = "sequence"
    with torch.no_grad():
        assert model(x).shape == (4, 6, 3)


def test_last_step_readout_matches_sequence_readout():
    model = TCN(4, 2, [8, 8], kernel_size=3, dropout=0.0).eval()
    x = torch.randn(2, 9, 4)
    with torch.no_grad():
        last = model(x)
        model.readout = "sequence"
        torch.testing.assert_close(last, model(x)[:, -1])


def test_outputs_are_causal_with_the_documented_receptive_field():
    torch.manual_seed(0)
    net = TemporalConvNet(2, [4, 4, 4], kernel_size=3, dropout=0.0).eval()
    field = net.receptive_field
    assert field == 1 + 2 * 2 * (2**3 - 1)
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.abs_()  # positive weights and inputs keep every ReLU active, so no path is cut
    length, t = 64, 50
    x = torch.rand(1, 2, length)
    with torch.no_grad():
        base = net(x)
        future = x.clone()
        future[:, :, t + 1 :] += 1.0
        torch.testing.assert_close(net(future)[:, :, : t + 1], base[:, :, : t + 1], rtol=0, atol=0)
        too_old = x.clone()
        too_old[:, :, t - field] += 1.0  # just outside the receptive field of step t
        assert torch.equal(net(too_old)[:, :, t], base[:, :, t])
        oldest = x.clone()
        oldest[:, :, t - field + 1] += 1.0  # the oldest input step t can see
        assert not torch.equal(net(oldest)[:, :, t], base[:, :, t])


def test_official_state_dict_layout_round_trips():
    model = build_model("tcn", task="regression", input_dim=3, hidden_dim=6, num_levels=2, kernel_size=2)
    official = to_official_state_dict(model.state_dict())
    assert list(official)[:3] == ["tcn.network.0.conv1.bias", "tcn.network.0.conv1.weight_g", "tcn.network.0.conv1.weight_v"]
    assert "tcn.network.0.net.0.weight_v" in official  # the official code registers each conv twice
    assert "tcn.network.0.downsample.weight" in official
    assert "tcn.network.1.downsample.weight" not in official  # same width: identity residual
    restored = build_model("tcn", task="regression", input_dim=3, hidden_dim=6, num_levels=2, kernel_size=2)
    restored.load_state_dict(official, strict=True)
    x = torch.randn(2, 5, 3)
    model.eval()
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(x), model(x))


def test_model_can_be_deep_copied():
    model = build_model("tcn", task="regression", input_dim=3, hidden_dim=6, num_levels=2)
    clone = copy.deepcopy(model)
    x = torch.randn(1, 4, 3)
    model.eval()
    clone.eval()
    with torch.no_grad():
        torch.testing.assert_close(clone(x), model(x))


@pytest.mark.parametrize("shape", [(2, 5), (2, 5, 4), (2, 1, 5, 3)])
def test_bad_input_shapes_raise(shape):
    model = build_model("tcn", task="regression", input_dim=3, hidden_dim=4, num_levels=2)
    with pytest.raises(ValueError):
        model(torch.randn(*shape))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(task="segmentation"),
        dict(task="regression", kernel_size=1),
        dict(task="regression", readout="mean"),
        dict(task="regression", head_init="xavier"),
        dict(task="regression", num_channels=[]),
    ],
)
def test_bad_configurations_raise(kwargs):
    with pytest.raises(ValueError):
        build_model("tcn", **kwargs)
