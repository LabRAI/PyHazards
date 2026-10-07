"""Attention U-Net: configurations, shapes and input validation.

Numerical equivalence with MONAI and TS-SatFire lives in tests/oracle/test_attention_unet_oracle.py.
"""

from __future__ import annotations

import pytest
import torch

from pyhazards.models import build_model
from pyhazards.models.attention_unet import AttentionUnet, TemporalAttentionUnet

SMALL = dict(channels=(8, 16, 32))


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_ts_satfire_parameter_counts():
    with torch.device("meta"):
        # Prediction model (3D, 43 channels) and the 2D detection model (8 channels).
        assert _n_params(build_model("attention_unet", task="segmentation", in_channels=43)) == 94_555_506
        assert _n_params(build_model("attention_unet", task="segmentation", in_channels=8, spatial_dims=2)) == 31_743_282


def test_default_strides_keep_time_and_halve_space():
    model = build_model("attention_unet", task="segmentation", in_channels=43)
    assert isinstance(model, TemporalAttentionUnet)
    assert model.level_strides == [(1, 2, 2)] * 4
    assert model.size_multiple() == (1, 16, 16)
    model_2d = build_model("attention_unet", task="segmentation", in_channels=8, spatial_dims=2)
    assert type(model_2d) is AttentionUnet
    assert model_2d.level_strides == [(2, 2)] * 4


def test_forward_shapes():
    model = build_model("attention_unet", task="segmentation", in_channels=4, out_channels=1, **SMALL).eval()
    x = torch.randn(2, 3, 4, 16, 16)
    with torch.no_grad():
        assert model(x).shape == (2, 1, 16, 16)
        model.time_reduction = "none"
        assert model(x).shape == (2, 1, 3, 16, 16)
    model_2d = build_model("attention_unet", task="segmentation", in_channels=4, spatial_dims=2, **SMALL).eval()
    with torch.no_grad():
        assert model_2d(torch.randn(2, 4, 12, 20)).shape == (2, 2, 12, 20)


def test_temporal_wrapper_is_the_3d_network_averaged_over_time():
    torch.manual_seed(0)
    temporal = TemporalAttentionUnet(spatial_dims=3, in_channels=3, out_channels=2, strides=(1, 2, 2), stride_mode="shared", **SMALL)
    plain = AttentionUnet(spatial_dims=3, in_channels=3, out_channels=2, strides=[(1, 2, 2)] * 2, **SMALL)
    plain.load_state_dict(temporal.state_dict(), strict=True)
    temporal.eval()
    plain.eval()
    x = torch.randn(2, 4, 3, 8, 8)
    with torch.no_grad():
        torch.testing.assert_close(temporal(x), plain(x.transpose(1, 2)).mean(2))


def test_shared_strides_equal_repeated_per_level_strides():
    torch.manual_seed(0)
    shared = AttentionUnet(spatial_dims=2, in_channels=3, out_channels=1, strides=(1, 2), stride_mode="shared", **SMALL)
    torch.manual_seed(0)
    per_level = AttentionUnet(spatial_dims=2, in_channels=3, out_channels=1, strides=[(1, 2), (1, 2)], **SMALL)
    for key, value in shared.state_dict().items():
        assert torch.equal(value, per_level.state_dict()[key]), key
    x = torch.randn(2, 3, 5, 8)
    shared.eval()
    per_level.eval()
    with torch.no_grad():
        torch.testing.assert_close(shared(x), per_level(x))


def test_monai_parameter_names():
    model = AttentionUnet(spatial_dims=2, in_channels=3, out_channels=1, strides=(2, 2), **SMALL)
    names = list(model.state_dict())
    assert names[:3] == ["model.0.conv.0.conv.weight", "model.0.conv.0.conv.bias", "model.0.conv.0.adn.N.weight"]
    assert "model.1.attention.W_g.0.conv.weight" in names
    assert "model.1.upconv.up.conv.weight" in names
    assert "model.1.merge.adn.A.weight" in names  # PReLU of the merge convolution
    assert "model.1.submodule.1.submodule.conv.1.conv.weight" in names
    assert names[-2:] == ["model.2.conv.weight", "model.2.conv.bias"]


@pytest.mark.parametrize(
    "spatial_dims, shape",
    [
        (3, (2, 4, 16, 16)),  # missing time axis
        (3, (2, 3, 5, 16, 16)),  # wrong channel count
        (3, (2, 3, 4, 14, 16)),  # height not divisible by 4
        (2, (2, 4, 16)),
        (2, (2, 4, 18, 16)),
    ],
)
def test_bad_input_shapes_raise(spatial_dims, shape):
    model = build_model("attention_unet", task="segmentation", in_channels=4, spatial_dims=spatial_dims, **SMALL)
    with pytest.raises(ValueError):
        model(torch.randn(*shape))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(strides=(2,)),  # one stride per level is needed
        dict(strides=2),  # an int needs stride_mode="shared"
        dict(strides=(2, 2), stride_mode="mixed"),
        dict(strides=[(2, 2, 2), 2]),  # stride with the wrong number of dimensions
        dict(strides=(2, 2), kernel_size=4),  # "same" padding needs odd kernels
        dict(strides=(2, 2), channels=(8,)),
    ],
)
def test_bad_configurations_raise(kwargs):
    config = dict(spatial_dims=2, in_channels=3, out_channels=1, channels=(8, 16, 32))
    config.update(kwargs)
    with pytest.raises(ValueError):
        AttentionUnet(**config)


def test_builder_validates_task_and_dims():
    with pytest.raises(ValueError):
        build_model("attention_unet", task="classification", in_channels=4)
    with pytest.raises(ValueError):
        build_model("attention_unet", task="segmentation", in_channels=4, spatial_dims=1)
