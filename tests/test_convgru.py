"""ConvGRU (FireCastNet's Conv-GRU baseline) without the reference code.

The comparison with FireCastNet's ``ConvGRUSeg`` is in tests/oracle/test_convgru_oracle.py.
"""

from __future__ import annotations

import pytest
import torch

from pyhazards.models import build_model
from pyhazards.models.convgru import ConvGRU, ConvGRUCell


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, 1_337_985),  # configs/conv-gru-config.yaml: 11 inputs, one layer of 128, 5x5
        (dict(hidden_dim=64), 361_793),  # the paper's "single hidden layer with 64 hidden units"
    ],
)
def test_parameter_counts(kwargs, expected):
    assert _n_params(build_model("convgru", task="segmentation", **kwargs)) == expected


def test_state_dict_layout():
    model = build_model("convgru", task="segmentation")
    assert list(model.state_dict()) == [
        "convgru_encoder.cell_list.0.in_conv.weight",
        "convgru_encoder.cell_list.0.in_conv.bias",
        "convgru_encoder.cell_list.0.out_conv.weight",
        "convgru_encoder.cell_list.0.out_conv.bias",
        "classification_layer.weight",
        "classification_layer.bias",
    ]


def test_forward_shape_and_gradients():
    model = build_model("convgru", task="segmentation", in_channels=4, hidden_dim=8, kernel_size=3, num_layers=2)
    logits = model(torch.randn(2, 5, 4, 16, 12))
    assert logits.shape == (2, 1, 16, 12)
    logits.mean().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_cell_follows_ballas_equations():
    torch.manual_seed(0)
    cell = ConvGRUCell(input_dim=3, hidden_dim=4, kernel_size=3)
    x, h = torch.randn(2, 3, 6, 6), torch.randn(2, 4, 6, 6)
    gates = torch.sigmoid(cell.in_conv(torch.cat([x, h], 1)))
    update, reset = gates[:, :4], gates[:, 4:]
    candidate = torch.tanh(cell.out_conv(torch.cat([x, reset * h], 1)))
    torch.testing.assert_close(cell(x, h), (1 - update) * h + update * candidate)


def test_encoder_returns_sequence_and_final_states():
    encoder = ConvGRU(input_dim=3, hidden_dim=[4, 5], kernel_size=[(3, 3), (1, 1)], num_layers=2)
    outputs, states = encoder(torch.randn(2, 6, 3, 8, 8))
    assert outputs.shape == (2, 6, 5, 8, 8)
    assert [s.shape for s in states] == [(2, 4, 8, 8), (2, 5, 8, 8)]
    torch.testing.assert_close(outputs[:, -1], states[-1])


def test_input_validation():
    model = build_model("convgru", task="segmentation", in_channels=3, hidden_dim=4)
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(2, 3, 8, 8))
    with pytest.raises(ValueError, match="channels"):
        model(torch.randn(2, 4, 5, 8, 8))
    with pytest.raises(ValueError, match="trajgru"):
        build_model("convgru", task="forecasting")
    with pytest.raises(ValueError, match="num_layers"):
        build_model("convgru", task="segmentation", num_layers=0)
