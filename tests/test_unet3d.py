"""MONAI U-Net (unet3d): configurations, shapes and input validation.

Numerical equivalence with MONAI and TS-SatFire lives in tests/oracle/test_unet3d_oracle.py.
"""

from __future__ import annotations

import pytest
import torch

from pyhazards.models import build_model
from pyhazards.models.unet3d import TemporalUNet, UNet

SMALL = dict(channels=(8, 16, 32))


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_ts_satfire_parameter_counts():
    with torch.device("meta"):
        # U-Net-3D prediction model (43 channels; Table 3: 31.7M) and the 2D U-Net (8 channels; 10.6M).
        assert _n_params(build_model("unet3d", task="segmentation", in_channels=43)) == 31_712_970
        assert _n_params(build_model("unet3d", task="segmentation", in_channels=8)) == 31_652_490
        assert _n_params(build_model("unet3d", task="segmentation", in_channels=8, spatial_dims=2)) == 10_552_458


def test_default_strides_keep_time_and_halve_space():
    model = build_model("unet3d", task="segmentation", in_channels=43)
    assert isinstance(model, TemporalUNet)
    assert model.level_strides == [(1, 2, 2)] * 4
    assert model.size_multiple() == (1, 16, 16)
    model_2d = build_model("unet3d", task="segmentation", in_channels=8, spatial_dims=2)
    assert type(model_2d) is UNet
    assert model_2d.level_strides == [(2, 2)] * 4


def test_forward_shapes():
    model = build_model("unet3d", task="segmentation", in_channels=4, out_channels=1, **SMALL).eval()
    x = torch.randn(2, 3, 4, 16, 16)
    with torch.no_grad():
        assert model(x).shape == (2, 1, 16, 16)
        model.time_reduction = "none"
        assert model(x).shape == (2, 1, 3, 16, 16)
    model_2d = build_model("unet3d", task="segmentation", in_channels=4, spatial_dims=2, num_res_units=2, **SMALL).eval()
    with torch.no_grad():
        assert model_2d(torch.randn(2, 4, 12, 20)).shape == (2, 2, 12, 20)


def test_temporal_wrapper_is_the_3d_network_averaged_over_time():
    torch.manual_seed(0)
    temporal = TemporalUNet(spatial_dims=3, in_channels=3, out_channels=2, strides=(1, 2, 2), stride_mode="shared", **SMALL)
    plain = UNet(spatial_dims=3, in_channels=3, out_channels=2, strides=[(1, 2, 2)] * 2, **SMALL)
    plain.load_state_dict(temporal.state_dict(), strict=True)
    temporal.eval()
    plain.eval()
    x = torch.randn(2, 4, 3, 8, 8)
    with torch.no_grad():
        torch.testing.assert_close(temporal(x), plain(x.transpose(1, 2)).mean(2))


def test_monai_parameter_names():
    model = UNet(spatial_dims=2, in_channels=3, out_channels=1, strides=(2, 2), **SMALL)
    names = list(model.state_dict())
    assert names[:4] == ["model.0.conv.weight", "model.0.conv.bias", "model.0.adn.A.weight", "model.1.submodule.0.conv.weight"]
    assert "model.1.submodule.1.submodule.conv.weight" in names  # bottom layer
    assert names[-2:] == ["model.2.conv.weight", "model.2.conv.bias"]  # top up-convolution: conv only
    residual = UNet(spatial_dims=2, in_channels=3, out_channels=1, strides=(2, 2), num_res_units=2, **SMALL)
    names = list(residual.state_dict())
    assert "model.0.conv.unit0.conv.weight" in names
    assert "model.0.residual.weight" in names
    assert "model.2.1.conv.unit0.conv.weight" in names


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
    model = build_model("unet3d", task="segmentation", in_channels=4, spatial_dims=spatial_dims, **SMALL)
    with pytest.raises(ValueError):
        model(torch.randn(*shape))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(strides=(2,)),
        dict(strides=2),
        dict(strides=(2, 2), stride_mode="mixed"),
        dict(strides=[(2, 2, 2), 2]),
        dict(strides=(2, 2), kernel_size=4),
        dict(strides=(2, 2), channels=(8,)),
        dict(strides=(2, 2), num_res_units=-1),
        dict(strides=(2, 2), act="gelu"),
        dict(strides=(2, 2), norm="group"),
    ],
)
def test_bad_configurations_raise(kwargs):
    config = dict(spatial_dims=2, in_channels=3, out_channels=1, channels=(8, 16, 32))
    config.update(kwargs)
    with pytest.raises(ValueError):
        UNet(**config)


def test_builder_validates_task_and_dims():
    with pytest.raises(ValueError):
        build_model("unet3d", task="classification", in_channels=4)
    with pytest.raises(ValueError):
        build_model("unet3d", task="segmentation", in_channels=4, spatial_dims=1)
