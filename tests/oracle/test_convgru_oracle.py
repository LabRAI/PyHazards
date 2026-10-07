"""ConvGRU checked against FireCastNet's Conv-GRU baseline.

Reference (pinned in repos.yaml): SeasFire/firecastnet ``seasfire/backbones/conv_gru.py``
(``ConvGRUSeg``), the network its ``configs/conv-gru-config.yaml`` trains. The repository has no
license; it is used only here and never vendored. ``seasfire.backbones.conv_gru`` imports only
torch, so this file runs in the ``default`` suite.
"""

from __future__ import annotations

import pytest
import torch
import yaml

from oracle_utils import import_from, oracle_repo
from pyhazards.models import build_model


@pytest.fixture()
def conv_gru():
    repo = oracle_repo("firecastnet")
    return repo, import_from(repo, "seasfire.backbones.conv_gru")


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _assert_same_state(reference: torch.nn.Module, port: torch.nn.Module) -> None:
    ref_state, port_state = reference.state_dict(), port.state_dict()
    assert list(ref_state) == list(port_state)
    for key, value in ref_state.items():
        assert torch.equal(value, port_state[key]), key


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_official_config_values(conv_gru):
    repo, _ = conv_gru
    config = yaml.safe_load((repo / "configs" / "conv-gru-config.yaml").read_text())
    model = config["model"]
    assert model["class_path"] == "seasfire.conv_gru_lit.ConvGRULit"
    assert model["init_args"] == {
        "input_dim_grid_nodes": 11, "hidden_layers": 1, "hidden_dim": 128, "kernel_size": [5, 5],
        "lr": 0.001, "weight_decay": 1e-08,
    }
    # ConvGRULit builds ConvGRUSeg(input_dim, hidden_dim, kernel_size, num_layers, num_classes=1) and
    # trains it with BCEWithLogitsLoss on every pixel of the 64x64 patch.
    source = (repo / "seasfire" / "conv_gru_lit.py").read_text()
    assert "num_classes=1" in source and "nn.BCEWithLogitsLoss()" in source
    assert (config["data"]["lat_dim"], config["data"]["lon_dim"]) == (64, 64)


@pytest.mark.parametrize("hidden_dim, expected", [(128, 1_337_985), (64, 361_793)])
def test_firecastnet_convgru_matches_reference(conv_gru, hidden_dim, expected):
    # 128 hidden channels: configs/conv-gru-config.yaml; 64: the paper (Sec. 5.1) and ConvGRULit's default.
    _, module = conv_gru
    torch.manual_seed(0)
    reference = module.ConvGRUSeg(input_dim=11, hidden_dim=hidden_dim, kernel_size=(5, 5), num_layers=1, num_classes=1)
    torch.manual_seed(0)
    port = build_model("convgru", task="segmentation", hidden_dim=hidden_dim)
    _assert_same_state(reference, port)
    assert _n_params(port) == expected

    torch.manual_seed(1)
    x = torch.randn(2, 12, 11, 64, 64)  # (batch, time, variables, lat, lon) after ConvGRULit's permute
    reference.eval()
    port.eval()
    with torch.no_grad():
        _assert_close(port(x), reference(x))
        # A sequence that starts with an all-zero (padding) step takes the reference's pad-mask
        # branch, which never changes the output of the one-layer model.
        x[:, :2] = 0
        _assert_close(port(x), reference(x))


def test_multilayer_convgru_matches_reference(conv_gru):
    _, module = conv_gru
    torch.manual_seed(2)
    reference = module.ConvGRUSeg(input_dim=5, hidden_dim=16, kernel_size=(3, 3), num_layers=3, num_classes=2)
    port = build_model("convgru", task="segmentation", in_channels=5, out_channels=2, hidden_dim=16, kernel_size=3, num_layers=3)
    port.load_state_dict(reference.state_dict(), strict=True)
    torch.manual_seed(3)
    x = torch.randn(3, 4, 5, 20, 24, requires_grad=True)
    expected = reference(x)
    actual = port(x)
    _assert_close(actual, expected)
    grad_ref, = torch.autograd.grad(expected.square().sum(), x)
    grad_port, = torch.autograd.grad(actual.square().sum(), x)
    _assert_close(grad_port, grad_ref)
