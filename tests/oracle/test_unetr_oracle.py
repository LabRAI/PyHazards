"""UNETR checked against MONAI 1.3.2 and TS-SatFire's modified copy.

References: ``monai.networks.nets.UNETR`` from monai 1.3.2 and ``spatial_models/unetr/unetr.py``
from zhaoyutim/TS-SatFire (pinned in repos.yaml; no LICENSE, so it is only imported here as an
oracle). Each check builds both models from the same seed, requires identical parameter names and
initial values, then compares outputs in eval and train mode after randomising the weights and
BatchNorm statistics.

UNETR's position embedding fixes the input size, so the forward checks build both models for a
small input -- (T, H, W) = (2, 32, 32) for the 3D prediction model, 64x64 for the 2D model --
while parameter counts are checked at TS-SatFire's 256x256 on the meta device.
"""

from __future__ import annotations

import pytest
import torch

from pyhazards.models import build_model
from pyhazards.models.unetr import UNETR, TemporalUNETR
from tssatfire_oracle_helpers import (
    PRED_CHANNELS,
    assert_close,
    assert_same_names_and_shapes,
    assert_same_state,
    compare_eval_and_train,
    monai_nets,
    n_params,
    randomise,
    ts_satfire_module,
)


def _ts_unetr():
    return ts_satfire_module("unetr.unetr").UNETR


def _ts_prediction_config(in_channels: int, img_size, feature_size: int = 16, hidden_size: int = 384, mlp_dim: int = 1536):
    # run_spatial_temp_model_pred.py, model "unetr3d" (default branch: hidden 384, MLP 1536). The
    # script also passes kernel_size_up_down=(1, 2, 2), an argument the released unetr.py does not
    # have; that file hard-codes (1, 2, 2) for every transposed convolution instead.
    return dict(
        in_channels=in_channels,
        out_channels=2,
        img_size=img_size,
        spatial_dims=3,
        norm_name="batch",
        feature_size=feature_size,
        patch_size=(1, 16, 16),
        hidden_size=hidden_size,
        mlp_dim=mlp_dim,
    )


def test_released_file_has_no_kernel_size_up_down_argument():
    """The prediction script's UNETR call does not run against the released unetr.py."""
    TSUNETR = _ts_unetr()
    with pytest.raises(TypeError):
        with torch.device("meta"):
            TSUNETR(**_ts_prediction_config(PRED_CHANNELS, (2, 32, 32)), kernel_size_up_down=(1, 2, 2))


def test_ts_satfire_prediction_model_matches_reference():
    TSUNETR = _ts_unetr()
    torch.manual_seed(0)
    reference = TSUNETR(**_ts_prediction_config(PRED_CHANNELS, (2, 32, 32)))
    torch.manual_seed(0)
    port = build_model("unetr", task="segmentation", in_channels=PRED_CHANNELS, history=2, image_size=32)
    assert isinstance(port, TemporalUNETR)
    assert_same_state(reference, port)

    randomise(reference, port, seed=1)
    torch.manual_seed(2)
    x = torch.randn(2, 2, PRED_CHANNELS, 32, 32)  # (batch, time, channels, H, W)
    x_ref = x.transpose(1, 2).contiguous()
    compare_eval_and_train(reference, port, x_ref, x, reduce_time=True)
    port.time_reduction = "none"
    reference.eval()
    port.eval()
    with torch.no_grad():
        assert_close(port(x), reference(x_ref))


def test_ts_satfire_parameter_counts_at_full_size():
    TSUNETR = _ts_unetr()
    with torch.device("meta"):
        # Prediction script configuration, 6 days of 43 channels at 256x256.
        reference = TSUNETR(**_ts_prediction_config(PRED_CHANNELS, (6, 256, 256)))
        port = build_model("unetr", task="segmentation", in_channels=PRED_CHANNELS)
        assert_same_names_and_shapes(reference, port)
        assert n_params(port) == n_params(reference) == 28_816_866
        # The 34.8M of Table 3 is the AF/BA configuration: 8 channels, feature size 36 (paper text).
        reference = TSUNETR(**_ts_prediction_config(8, (6, 256, 256), feature_size=36))
        port = build_model("unetr", task="segmentation", in_channels=8, feature_size=36)
        assert_same_names_and_shapes(reference, port)
        assert n_params(port) == n_params(reference) == 34_806_506
        # The prediction script's alternative "v0" widths.
        reference = TSUNETR(**_ts_prediction_config(PRED_CHANNELS, (6, 256, 256), hidden_size=768, mlp_dim=3072))
        port = build_model("ts_satfire", task="segmentation", baseline="unetr3d", unetr_version="v0")
        assert_same_names_and_shapes(reference, port)
        assert n_params(port) == n_params(reference) == 97_922_658


def test_ts_satfire_2d_config_matches_monai():
    MonaiUNETR = monai_nets().UNETR
    # run_spatial_model.py, model "unetr2d_half": stock MONAI, hidden 384, MLP 1536, batch norm.
    config = dict(in_channels=8, out_channels=2, spatial_dims=2, norm_name="batch", feature_size=16, hidden_size=384, mlp_dim=1536)
    with torch.device("meta"):
        reference = MonaiUNETR(img_size=(256, 256), **config)
        port = build_model("unetr", task="segmentation", in_channels=8, spatial_dims=2)
        assert type(port) is UNETR
        assert_same_names_and_shapes(reference, port)
        assert n_params(port) == 23_521_186  # Table 3: 23.52M

    torch.manual_seed(0)
    reference = MonaiUNETR(img_size=(64, 64), **config)
    torch.manual_seed(0)
    port = build_model("unetr", task="segmentation", in_channels=8, spatial_dims=2, image_size=64)
    assert_same_state(reference, port)
    randomise(reference, port, seed=3)
    torch.manual_seed(4)
    x = torch.randn(2, 8, 64, 64)
    compare_eval_and_train(reference, port, x, x)


def test_non_default_configs_match_monai():
    """Instance norm, dropout, qkv bias, perceptron patches and plain (non-residual) blocks."""
    MonaiUNETR = monai_nets().UNETR
    configs = [
        dict(in_channels=3, out_channels=2, img_size=(32, 32, 32), feature_size=8, hidden_size=48, mlp_dim=96, num_heads=4, dropout_rate=0.1, qkv_bias=True),
        dict(in_channels=3, out_channels=3, img_size=(32, 48), spatial_dims=2, feature_size=8, hidden_size=48, mlp_dim=0, num_heads=3, proj_type="perceptron", norm_name="batch"),
        dict(in_channels=2, out_channels=1, img_size=(32, 32), spatial_dims=2, feature_size=12, hidden_size=48, mlp_dim=64, num_heads=4, conv_block=False, res_block=False, save_attn=True),
        dict(in_channels=2, out_channels=1, img_size=(16, 32, 16), feature_size=8, hidden_size=32, mlp_dim=64, num_heads=2, res_block=False, norm_name=("batch", {"momentum": 0.3})),
    ]
    for config in configs:
        torch.manual_seed(0)
        reference = MonaiUNETR(**config)
        torch.manual_seed(0)
        port = UNETR(**config)
        assert_same_state(reference, port)
        randomise(reference, port, seed=5)
        torch.manual_seed(6)
        x = torch.randn(2, config["in_channels"], *config["img_size"])
        compare_eval_and_train(reference, port, x, x, seed=7)  # same dropout masks: same module order
        if config.get("save_attn"):
            for ref_block, port_block in zip(reference.vit.blocks, port.vit.blocks):
                assert_close(port_block.attn.att_mat, ref_block.attn.att_mat)
