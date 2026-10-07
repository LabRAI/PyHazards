"""MONAI U-Net checked against MONAI 1.3.2 and TS-SatFire's modified copy.

References: ``monai.networks.nets.UNet`` from monai 1.3.2 and ``spatial_models/unet.py`` from
zhaoyutim/TS-SatFire (pinned in repos.yaml; no LICENSE, so it is only imported here as an
oracle). Each check builds both models from the same seed, requires identical parameter names and
initial values, then compares outputs in eval and train mode after randomising the weights.

Spatial sizes are reduced for CPU speed (the network is fully convolutional; TS-SatFire uses
256x256 tiles): 32x32 with T = 2 for the full-width 3D prediction model, 64x64 for the 2D model.
"""

from __future__ import annotations

import torch

from pyhazards.models import build_model
from pyhazards.models.unet3d import TemporalUNet, UNet
from tssatfire_oracle_helpers import (
    PRED_CHANNELS,
    TS_SATFIRE_CHANNELS,
    assert_close,
    assert_same_state,
    compare_eval_and_train,
    monai_nets,
    n_params,
    randomise,
    ts_satfire_module,
)

SMALL_CHANNELS = (8, 16, 32, 64, 128)


def test_ts_satfire_prediction_model_matches_reference():
    TSUNet = ts_satfire_module("unet").UNet
    # run_spatial_temp_model_pred.py, model "unet3d": 43 input channels, 2 classes.
    torch.manual_seed(0)
    reference = TSUNet(
        spatial_dims=3, in_channels=PRED_CHANNELS, out_channels=2, channels=TS_SATFIRE_CHANNELS, strides=(1, 2, 2)
    )
    torch.manual_seed(0)
    port = build_model("unet3d", task="segmentation", in_channels=PRED_CHANNELS)
    assert isinstance(port, TemporalUNet)
    assert_same_state(reference, port)
    assert n_params(port) == n_params(reference) == 31_712_970  # Table 3: 31.7M

    randomise(reference, port, seed=1)
    torch.manual_seed(2)
    x = torch.randn(2, 2, PRED_CHANNELS, 32, 32)  # (batch, time, channels, H, W)
    x_ref = x.transpose(1, 2).contiguous()  # TS-SatFire feeds (batch, channels, time, H, W)
    compare_eval_and_train(reference, port, x_ref, x, reduce_time=True)
    port.time_reduction = "none"
    reference.eval()
    port.eval()
    with torch.no_grad():
        assert_close(port(x), reference(x_ref))


def test_ts_satfire_shared_strides_equal_monai_per_level_strides():
    TSUNet = ts_satfire_module("unet").UNet
    MonaiUNet = monai_nets().UNet
    config = dict(spatial_dims=3, in_channels=5, out_channels=2, channels=SMALL_CHANNELS)
    torch.manual_seed(0)
    ts_model = TSUNet(**config, strides=(1, 2, 2))
    torch.manual_seed(0)
    monai_model = MonaiUNet(**config, strides=((1, 2, 2),) * 4)
    torch.manual_seed(0)
    port_shared = UNet(**config, strides=(1, 2, 2), stride_mode="shared")
    torch.manual_seed(0)
    port_per_level = UNet(**config, strides=((1, 2, 2),) * 4)
    for model in (monai_model, port_shared, port_per_level):
        assert_same_state(ts_model, model)

    torch.manual_seed(1)
    x = torch.randn(2, 5, 3, 32, 32)
    for model in (ts_model, monai_model, port_shared, port_per_level):
        model.eval()
    with torch.no_grad():
        expected = ts_model(x)
        assert expected.shape == (2, 2, 3, 32, 32)  # time is never down-sampled
        for model in (monai_model, port_shared, port_per_level):
            assert_close(model(x), expected)


def test_ts_satfire_2d_config_matches_monai():
    MonaiUNet = monai_nets().UNet
    # run_spatial_model.py, model "unet": stock MONAI, 2D, strides (2, 2, 2, 2); 8 VIIRS bands.
    torch.manual_seed(0)
    reference = MonaiUNet(spatial_dims=2, in_channels=8, out_channels=2, channels=TS_SATFIRE_CHANNELS, strides=(2, 2, 2, 2))
    torch.manual_seed(0)
    port = build_model("unet3d", task="segmentation", in_channels=8, spatial_dims=2)
    assert type(port) is UNet
    assert_same_state(reference, port)
    assert n_params(port) == 10_552_458  # Table 3: 10.6M

    randomise(reference, port, seed=3)
    torch.manual_seed(4)
    x = torch.randn(2, 8, 64, 64)
    compare_eval_and_train(reference, port, x, x)


def test_non_default_configs_match_monai():
    """Residual units, batch norm / ReLU, dropout, larger kernels and non-uniform strides in 1D-3D."""
    MonaiUNet = monai_nets().UNet
    configs = [
        dict(spatial_dims=1, in_channels=3, out_channels=1, channels=(4, 8, 16), strides=(2, 2), kernel_size=5),
        dict(
            spatial_dims=2,
            in_channels=3,
            out_channels=3,
            channels=(4, 8, 16, 32),
            strides=((2, 1), 2, (1, 2)),
            num_res_units=2,
            dropout=0.2,
        ),
        dict(
            spatial_dims=2,
            in_channels=3,
            out_channels=2,
            channels=(4, 8, 16),
            strides=(2, 2),
            num_res_units=1,
            act="relu",
            norm="batch",
            kernel_size=(3, 5),
            up_kernel_size=5,
            bias=False,
        ),
        dict(
            spatial_dims=3,
            in_channels=2,
            out_channels=1,
            channels=(4, 8, 16),
            strides=((1, 2, 2), 2),
            num_res_units=2,
            norm="batch",
            dropout=0.1,
            adn_ordering="ADN",
        ),
    ]
    inputs = [(2, 3, 16), (2, 3, 16, 16), (2, 3, 16, 16), (2, 2, 4, 16, 16)]
    for config, shape in zip(configs, inputs):
        torch.manual_seed(0)
        reference = MonaiUNet(**config)
        torch.manual_seed(0)
        port = UNet(**config)
        assert_same_state(reference, port)
        randomise(reference, port, seed=5)
        x = torch.randn(*shape)
        compare_eval_and_train(reference, port, x, x, seed=6)  # same dropout masks: same module order
