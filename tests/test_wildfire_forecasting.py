"""Kondylatos et al. (2022) LSTM / ConvLSTM: configurations, shapes and input validation.

Numerical equivalence with the official code lives in tests/oracle/test_wildfire_forecasting_oracle.py.
"""

from __future__ import annotations

import pytest
import torch

from pyhazards.datasets import load_dataset
from pyhazards.models import SimpleConvLSTM, SimpleLSTM, WILDFIRE_FORECASTING_VARIANTS, build_model

# Computed from the official fire_modules.py at the paper configuration (25 input features,
# hidden size 64 for the LSTM and 32 for the ConvLSTM).
EXPECTED_PARAMS = {"lstm": 29_652, "convlstm": 372_212}
EXAMPLE_INPUT = {"lstm": (3, 10, 25), "convlstm": (3, 10, 25, 25, 25)}


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("variant", WILDFIRE_FORECASTING_VARIANTS)
def test_paper_parameter_counts(variant):
    model = build_model("wildfire_forecasting", task="classification", variant=variant)
    assert _n_params(model) == EXPECTED_PARAMS[variant]


def test_default_variant_is_lstm():
    model = build_model("wildfire_forecasting", task="classification")
    assert isinstance(model, SimpleLSTM)
    assert model.lstm.hidden_size == 64
    convlstm = build_model("wildfire_forecasting", task="classification", variant="convlstm")
    assert isinstance(convlstm, SimpleConvLSTM)
    assert convlstm.convlstm.hidden_dim == [32]
    assert convlstm.fc1.in_features == 12 * 12 * 32


@pytest.mark.parametrize("variant", WILDFIRE_FORECASTING_VARIANTS)
def test_outputs_are_log_probabilities(variant):
    model = build_model("wildfire_forecasting", task="classification", variant=variant).eval()
    with torch.no_grad():
        out = model(torch.randn(*EXAMPLE_INPUT[variant]))
    assert out.shape == (3, 2)
    assert (out <= 0).all()
    torch.testing.assert_close(out.exp().sum(dim=1), torch.ones(3))


def test_lstm_keeps_reference_state_dict_layout():
    model = SimpleLSTM()
    keys = list(model.state_dict())
    # The reference registers fc1/fc2/fc3 twice (also inside fc_nn), so both key sets exist.
    assert keys[:2] == ["ln1.weight", "ln1.bias"]
    assert {"fc1.weight", "fc_nn.0.weight", "fc2.weight", "fc_nn.3.weight", "fc3.weight", "fc_nn.6.weight"} <= set(keys)
    assert model.fc_nn[0] is model.fc1


def test_convlstm_keeps_reference_state_dict_layout():
    keys = list(SimpleConvLSTM().state_dict())
    assert keys == [
        "ln1.weight",
        "ln1.bias",
        "convlstm.cell_list.0.conv.weight",
        "convlstm.cell_list.0.conv.bias",
        "conv1.weight",
        "conv1.bias",
        "fc1.weight",
        "fc1.bias",
        "fc2.weight",
        "fc2.bias",
        "fc3.weight",
        "fc3.bias",
    ]


def test_hyperparameters_are_configurable():
    lstm = build_model("wildfire_forecasting", task="classification", hidden_size=16, lstm_layers=2, input_dim=12)
    assert lstm(torch.randn(2, 7, 12)).shape == (2, 2)
    convlstm = build_model(
        "wildfire_forecasting", task="classification", variant="convlstm", hidden_size=8, patch_size=9, input_dim=4
    )
    assert convlstm.fc1.in_features == 4 * 4 * 8
    assert convlstm(torch.randn(2, 3, 4, 9, 9)).shape == (2, 2)


def test_invalid_inputs_raise_value_error():
    lstm = SimpleLSTM()
    with pytest.raises(ValueError, match="shape"):
        lstm(torch.randn(2, 10, 25, 1))
    with pytest.raises(ValueError, match="features"):
        lstm(torch.randn(2, 10, 24))
    convlstm = SimpleConvLSTM()
    with pytest.raises(ValueError, match="shape"):
        convlstm(torch.randn(2, 10, 25, 25))
    with pytest.raises(ValueError, match="features"):
        convlstm(torch.randn(2, 10, 24, 25, 25))
    with pytest.raises(ValueError, match="25x25"):
        convlstm(torch.randn(2, 10, 25, 32, 32))


def test_invalid_builder_arguments():
    with pytest.raises(ValueError, match="classification"):
        build_model("wildfire_forecasting", task="forecasting")
    with pytest.raises(ValueError, match="variant"):
        build_model("wildfire_forecasting", task="classification", variant="gru")
    with pytest.raises(ValueError, match="dropout"):
        build_model("wildfire_forecasting", task="classification", dropout=1.0)


@pytest.mark.parametrize(
    "variant,access_mode",
    [("lstm", "temporal"), ("convlstm", "spatiotemporal")],
)
def test_synthetic_danger_dataset_feeds_each_variant(variant, access_mode):
    bundle = load_dataset("wildfire_danger_synthetic", micro=True, access_mode=access_mode).load()
    train = bundle.get_split("train")
    assert train.inputs.shape[1:] == EXAMPLE_INPUT[variant][1:]
    assert train.targets.dtype == torch.long
    assert set(train.targets.tolist()) == {0, 1}
    assert bundle.label_spec.task_type == "classification"
    model = build_model("wildfire_forecasting", task="classification", variant=variant).eval()
    with torch.no_grad():
        assert model(train.inputs[:2]).shape == (2, 2)


def test_synthetic_danger_dataset_is_deterministic_with_two_to_one_negatives():
    first = load_dataset("wildfire_danger_synthetic", samples=30).load()
    second = load_dataset("wildfire_danger_synthetic", samples=30).load()
    torch.testing.assert_close(first.get_split("train").inputs, second.get_split("train").inputs)
    labels = torch.cat([first.get_split(name).targets for name in ("train", "val", "test")])
    assert int(labels.sum()) == 10
    with pytest.raises(ValueError, match="access_mode"):
        load_dataset("wildfire_danger_synthetic", access_mode="spatial")
