"""Rainformer: parameter counts, stage geometry, layouts, the segmentation adaptation, validation.

Numerical equivalence with the official code (and its KNMI weights) lives in
tests/oracle/test_rainformer_oracle.py.
"""

from __future__ import annotations

import pytest
import torch

from pyhazards.models import build_model
from pyhazards.models.rainformer import Rainformer, rainformer_stage_sizes

SMALL = dict(history=3, img_size=64, hidden_dim=8, heads=(1, 1, 1, 1), head_dim=4, window_size=2)


def _n_params(model: torch.nn.Module, trainable: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable)


def test_rainformer_default_parameter_count():
    with torch.device("meta"):
        model = build_model("rainformer", task="forecasting")
    assert model.input_channel == 9 and model.img_size == (288, 288)
    assert _n_params(model) == 187_661_990  # official Net(...) of train.py, masks included
    assert _n_params(model, trainable=True) == 187_557_014


def test_rainformer_stage_sizes_follow_the_input_size():
    assert rainformer_stage_sizes(288, (4, 2, 2, 2)) == [
        (72, 72), (36, 36), (18, 18), (9, 9), (18, 18), (36, 36), (72, 72), (288, 288)
    ]
    assert rainformer_stage_sizes((256, 128), (4, 2, 2, 2))[3] == (8, 4)
    with pytest.raises(ValueError):
        rainformer_stage_sizes(100, (4, 2, 2, 2))


def test_rainformer_256_input_needs_another_window_size():
    # Sim2Real-Fire's 256x256 frames give 64/32/16/8 maps, which window 9 does not divide.
    with pytest.raises(ValueError, match="window_size 9"):
        build_model("rainformer", task="forecasting", img_size=256)
    with torch.device("meta"):
        model = build_model("rainformer", task="forecasting", img_size=256, window_size=8)
    assert model.stage_sizes[3] == (8, 8)


def test_rainformer_official_and_time_major_layouts_agree():
    model = build_model("rainformer", task="forecasting", in_channels=2, **SMALL).eval()
    x = torch.randn(2, 3, 2, 64, 64)
    with torch.no_grad():
        out = model(x)
        torch.testing.assert_close(out.flatten(1, 2), model(x.flatten(1, 2)))
    assert out.shape == (2, 3, 2, 64, 64)


def test_rainformer_segmentation_adaptation_keeps_reference_names():
    torch.manual_seed(0)
    core = build_model("rainformer", task="forecasting", in_channels=2, **SMALL)
    torch.manual_seed(0)
    seg = build_model("rainformer", task="segmentation", in_channels=2, **SMALL)
    core_state, seg_state = core.state_dict(), seg.state_dict()
    assert list(seg_state) == list(core_state) + ["segmentation_head.weight", "segmentation_head.bias"]
    for key, value in core_state.items():
        assert torch.equal(value, seg_state[key]), key

    core.eval()
    seg.eval()
    x = torch.randn(2, 3, 2, 64, 64)
    with torch.no_grad():
        logits = seg(x)
        torch.testing.assert_close(logits, seg.segmentation_head(core(x)[:, 0]))
    assert logits.shape == (2, 1, 64, 64)
    with pytest.raises(ValueError):
        seg(x.flatten(1, 2))  # the adaptation needs the (batch, time, channels, H, W) layout


def test_rainformer_unused_gate_convolutions_get_no_gradient():
    model = build_model("rainformer", task="forecasting", **SMALL)
    model(torch.randn(1, 3, 64, 64)).sum().backward()
    gate = model.stage1.layers[0][4]
    assert gate.conv_1[0].weight.grad is not None
    assert gate.conv_2[0].weight.grad is None and gate.conv_3[0].weight.grad is None
    assert not model.stage1.layers[0][1].attention_block.fn.fn.upper_lower_mask.requires_grad


def test_rainformer_extra_stage_groups_reuse_the_stage_input():
    # Reference behaviour: with layers > 2 every group reads the stage input; the last one wins.
    torch.manual_seed(0)
    deep = build_model("rainformer", task="forecasting", **{**SMALL, "layers": (4, 2, 2, 2)}).eval()
    torch.manual_seed(1)
    shallow = build_model("rainformer", task="forecasting", **SMALL).eval()
    state = {}
    for key, value in deep.state_dict().items():  # layers[0] sets stage1 and stage8: keep group 1
        if key.startswith(("stage1.layers.0.", "stage8.layers.0.")):
            continue
        state[key.replace("stage1.layers.1.", "stage1.layers.0.").replace("stage8.layers.1.", "stage8.layers.0.")] = value
    shallow.load_state_dict(state, strict=True)
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(deep(x), shallow(x))


@pytest.mark.parametrize("shape", [(1, 4, 64, 64), (1, 3, 32, 32), (1, 2, 2, 64, 64), (3, 64, 64)])
def test_rainformer_bad_input_shapes_raise(shape):
    model = build_model("rainformer", task="forecasting", **SMALL)
    with pytest.raises(ValueError):
        model(torch.randn(*shape))


@pytest.mark.parametrize(
    "overrides",
    [dict(layers=(1, 2, 2, 2)), dict(layers=(2, 2, 2)), dict(window_size=1), dict(window_size=3), dict(hidden_dim=6), dict(history=0)],
)
def test_rainformer_rejects_unsupported_configurations(overrides):
    with pytest.raises(ValueError):
        build_model("rainformer", task="forecasting", **{**SMALL, **overrides})


def test_rainformer_rejects_unknown_task():
    with pytest.raises(ValueError):
        build_model("rainformer", task="regression", **SMALL)


def test_rainformer_class_defaults_are_the_official_configuration():
    with torch.device("meta"):
        model = Rainformer()
    attention = model.stage4.layers[0][1].attention_block.fn.fn
    assert (attention.heads, attention.head_dim, attention.window_size) == (24, 32, 9)
    assert model.stage1.patch_partition.linear.in_features == 9 * 16
    assert model.stage8.layers[0][4].conv_1[1].normalized_shape == (18, 288, 288)
