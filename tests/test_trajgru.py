"""TrajGRU without the reference code: sizes, the warp, shapes, states and validation.

The comparison with the official MXNet implementation and its released weights is in
tests/oracle/test_trajgru_oracle.py.
"""

from __future__ import annotations

import pytest
import torch

from pyhazards.models import build_model
from pyhazards.models.trajgru import TrajGRU, TrajGRUCell, TrajGRUSegmenter, load_hko7_params, warp

SMALL = dict(config="movingmnist", num_filter=(8, 12, 12), L=3, first_conv=(4, 3, 1, 1), last_deconv=(4, 3, 1, 1))


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, 12_150_053),  # HKO-7 TrajGRU (trajgru_55_55_33_1_64_1_192_1_192_13_13_9_b4.yml)
        (dict(layer_type="ConvGRU"), 13_649_081),  # HKO-7 ConvGRU baseline (convgru_55_55_33_..._b4.yml)
        (dict(config="movingmnist"), 4_000_397),  # MovingMNIST++ TrajGRU-L13: paper Table 1, 4.00M
        (dict(config="movingmnist", L=9), 3_421_277),  # Traj-L9, 3.42M
        (dict(config="movingmnist", layer_type="ConvGRU", h2h_kernel=5), 4_767_505),  # Conv-K5-D1, 4.77M
        (dict(config="movingmnist", layer_type="ConvGRU", h2h_kernel=7), 8_011_537),  # Conv-K7-D1, 8.01M
    ],
)
def test_parameter_counts(kwargs, expected):
    assert _n_params(build_model("trajgru", task="forecasting", **kwargs)) == expected


def test_parameter_names_follow_mxnet_names():
    model = build_model("trajgru", task="forecasting")
    names = {key.replace(".", "_") for key in model.state_dict()}
    assert {"econv1_weight", "ebrnn1_0_i2h_weight", "ebrnn1_0_f_out_bias", "edown1_conv_weight",
            "fbrnn3_0_h2h_weight", "fup2_deconv_weight", "fdeconv1_weight", "conv_final_bias", "out_weight"} <= names
    # The top forecaster block runs without input; deconvolutions have no bias.
    assert not any(name.startswith(("fbrnn3_0_i2h", "fbrnn3_0_i2f")) for name in names)
    assert "fup1_deconv_bias" not in names and "fdeconv1_bias" not in names
    assert model.econv1.in_channels == 4  # frame + x, y and ones channels


def test_flow_layers_start_at_zero():
    model = build_model("trajgru", task="forecasting", config="movingmnist")
    for module in model.modules():
        if isinstance(module, TrajGRUCell):
            assert not module.f_out.weight.any() and not module.f_out.bias.any()
    assert model.ebrnn1[0].h2h.weight.std() > 0
    assert not model.ebrnn1[0].h2h.bias.any()


def test_warp_with_hand_computed_flows():
    data = torch.arange(16.0).view(1, 1, 4, 4)
    zero = torch.zeros(1, 2, 4, 4)
    torch.testing.assert_close(warp(data, zero), data)

    shift_x = zero.clone()
    shift_x[:, 0] = 1.0  # output (y, x) samples the input at (y, x - 1); zeros enter at the border
    expected = torch.tensor([[0.0, 0, 1, 2], [0, 4, 5, 6], [0, 8, 9, 10], [0, 12, 13, 14]]).view(1, 1, 4, 4)
    torch.testing.assert_close(warp(data, shift_x), expected)

    half_y = zero.clone()
    half_y[:, 1] = 0.5  # halfway to the previous row
    previous = torch.cat([torch.zeros(1, 1, 1, 4), data[:, :, :-1]], dim=2)
    torch.testing.assert_close(warp(data, half_y), 0.5 * (data + previous))

    # Two links on a non-square map: outputs are concatenated link by link.
    wide = torch.arange(30.0).view(1, 2, 3, 5)
    flows = torch.zeros(1, 4, 3, 5)
    flows[:, 2] = -2.0  # second link: sample at x + 2
    out = warp(wide, flows)
    assert out.shape == (1, 4, 3, 5)
    torch.testing.assert_close(out[:, :2], wide)
    torch.testing.assert_close(out[:, 2:, :, :3], wide[:, :, :, 2:])
    assert not out[:, 2:, :, 3:].any()


def test_forecasting_and_segmentation_shapes():
    model = build_model("trajgru", task="forecasting", in_channels=2, num_output_frames=4, **SMALL).eval()
    x = torch.rand(3, 5, 2, 32, 32)
    with torch.no_grad():
        y = model(x)
        assert y.shape == (3, 4, 1, 32, 32)
        assert model(x, num_output_frames=2).shape == (3, 2, 1, 32, 32)
        prediction, states = model(x, return_states=True)
        torch.testing.assert_close(prediction, y)
        assert [len(block) for block in states] == [1, 1, 1]
        assert [block[0].shape for block in states] == [(3, 8, 32, 32), (3, 12, 16, 16), (3, 12, 8, 8)]
        # Zero initial states are the default.
        torch.testing.assert_close(model(x, initial_states=[[torch.zeros_like(s[0])] for s in states]), y)
    segmenter = build_model("trajgru", task="segmentation", in_channels=2, **SMALL).eval()
    assert isinstance(segmenter, TrajGRUSegmenter)
    segmenter.load_state_dict(model.state_dict(), strict=True)  # same keys, no prefix
    with torch.no_grad():
        torch.testing.assert_close(segmenter(x), y[:, 0])
    logits = segmenter(x)
    assert logits.shape == (3, 1, 32, 32)


def test_convgru_blocks_and_gradients():
    model = build_model("trajgru", task="segmentation", layer_type="ConvGRU", h2h_kernel=3, h2h_dilate=2, **SMALL)
    assert model.ebrnn1[0].h2h.dilation == (2, 2) and model.ebrnn1[0].h2h.padding == (2, 2)
    out = model(torch.rand(2, 3, 1, 16, 16))
    out.mean().backward()
    assert out.shape == (2, 1, 16, 16)
    assert model.econv1.weight.grad is not None


def test_load_hko7_params_round_trip():
    source = build_model("trajgru", task="forecasting", **SMALL)
    for parameter in source.parameters():
        torch.nn.init.normal_(parameter)
    params = {"arg:" + key.replace(".", "_"): value.numpy() for key, value in source.state_dict().items()}
    target = build_model("trajgru", task="forecasting", **SMALL)
    load_hko7_params(target, params)
    for key, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], value), key
    params.pop("arg:out_bias")
    with pytest.raises(KeyError, match="out_bias"):
        load_hko7_params(target, params)


def test_input_validation():
    model = build_model("trajgru", task="forecasting", **SMALL)
    with pytest.raises(ValueError, match="shape"):
        model(torch.rand(2, 1, 32, 32))
    with pytest.raises(ValueError, match="input channels"):
        model(torch.rand(2, 3, 2, 32, 32))
    with pytest.raises(ValueError, match="up-sampling"):
        model(torch.rand(1, 3, 1, 30, 30))  # 30 -> 15 -> 8 -> up 16 != 15
    with pytest.raises(ValueError, match="up-sampling"):
        build_model("trajgru", task="forecasting")(torch.rand(1, 2, 1, 256, 256))  # HKO-7 needs e.g. 480
    with pytest.raises(ValueError, match="layer_type"):
        build_model("trajgru", task="forecasting", layer_type="LSTM")
    with pytest.raises(ValueError, match="config"):
        build_model("trajgru", task="forecasting", config="sevir")
    with pytest.raises(ValueError, match="preserve"):
        TrajGRU(i2h_kernel=5, i2h_pad=1)
    with pytest.raises(ValueError, match="forecasting"):
        build_model("trajgru", task="classification")
