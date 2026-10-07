"""SwinUNETR checked against MONAI 1.3.2 and TS-SatFire's modified copy.

References: ``monai.networks.nets.SwinUNETR`` from monai 1.3.2 and ``spatial_models/swinunetr/``
from zhaoyutim/TS-SatFire (pinned in repos.yaml; no LICENSE, so it is only imported here as an
oracle). Each check builds both models from the same seed, requires identical parameter names and
initial values, then compares outputs in eval and train mode after randomising the weights and
BatchNorm statistics.

Forward checks use 32x32 (3D) or 64x64 (2D) tiles instead of 256x256 for CPU speed; the merging
plan of TS-SatFire's copy depends only on the time length T, which is kept at the paper's values.
Parameter counts are checked at the full 256x256 size.
"""

from __future__ import annotations

import pytest
import torch

from pyhazards.models import build_model
from pyhazards.models.swin_unetr import SwinUNETR, TemporalSwinUNETR
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


def _ts_swinunetr():
    return ts_satfire_module("swinunetr.swinunetr").SwinUNETR


def _ts_config(in_channels: int, history: int, size: int, feature_size: int = 36, num_heads: int = 3, attn_version: str = "v1"):
    # run_spatial_temp_model_pred.py, model "swinunetr3d" (-ed = feature size, -nh = heads).
    return dict(
        image_size=(history, size, size),
        patch_size=(1, 2, 2),
        window_size=(history, 4, 4),
        in_channels=in_channels,
        out_channels=2,
        depths=(2, 2, 2, 2),
        num_heads=(num_heads,) * 4,
        feature_size=feature_size,
        norm_name="batch",
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        attn_version=attn_version,
        normalize=True,
        use_checkpoint=False,
        spatial_dims=3,
    )


def test_ts_satfire_prediction_model_matches_reference():
    TSSwinUNETR = _ts_swinunetr()
    torch.manual_seed(0)
    reference = TSSwinUNETR(**_ts_config(PRED_CHANNELS, history=6, size=32))
    torch.manual_seed(0)
    port = build_model("swin_unetr", task="segmentation", in_channels=PRED_CHANNELS, history=6, image_size=32)
    assert isinstance(port, TemporalSwinUNETR)
    assert_same_state(reference, port)
    assert port.swinViT.resamples == reference.swinViT.resamples == [[1, 2, 2], [1, 2, 2], [1, 2, 2], [2, 2, 2]]

    randomise(reference, port, seed=1)
    torch.manual_seed(2)
    x = torch.randn(2, 6, PRED_CHANNELS, 32, 32)  # (batch, time, channels, H, W)
    x_ref = x.transpose(1, 2).contiguous()
    compare_eval_and_train(reference, port, x_ref, x, reduce_time=True)
    port.time_reduction = "none"
    reference.eval()
    port.eval()
    with torch.no_grad():
        assert_close(port(x), reference(x_ref))


@pytest.mark.parametrize("history", [2, 3, 4])
def test_ts_satfire_merging_plans_match_reference(history):
    """T = 2 and 4 (Table 5) merge time once and twice; an odd T never merges it."""
    TSSwinUNETR = _ts_swinunetr()
    torch.manual_seed(0)
    reference = TSSwinUNETR(**_ts_config(7, history=history, size=32, feature_size=12))
    torch.manual_seed(0)
    port = build_model("swin_unetr", task="segmentation", in_channels=7, history=history, image_size=32, feature_size=12)
    assert_same_state(reference, port)
    assert port.swinViT.resamples == reference.swinViT.resamples
    randomise(reference, port, seed=3)
    x = torch.randn(2, history, 7, 32, 32)
    compare_eval_and_train(reference, port, x.transpose(1, 2).contiguous(), x, reduce_time=True)


def test_ts_satfire_v2_attention_matches_reference():
    """``attn_version="v2"`` (an option of the AF/BA script): Swin-V2 cosine attention."""
    TSSwinUNETR = _ts_swinunetr()
    torch.manual_seed(0)
    reference = TSSwinUNETR(**_ts_config(8, history=4, size=32, feature_size=24, num_heads=4, attn_version="v2"))
    torch.manual_seed(0)
    port = build_model(
        "swin_unetr", task="segmentation", in_channels=8, history=4, image_size=32, feature_size=24, num_heads=4, attn_version="v2"
    )
    assert_same_state(reference, port)
    randomise(reference, port, seed=4)
    x = torch.randn(2, 4, 8, 32, 32)
    compare_eval_and_train(reference, port, x.transpose(1, 2).contiguous(), x, reduce_time=True)


def test_ts_satfire_parameter_counts_at_full_size():
    TSSwinUNETR = _ts_swinunetr()
    # The reference calls Tensor.item() while building, so it cannot be built on the meta device.
    reference = TSSwinUNETR(**_ts_config(PRED_CHANNELS, history=6, size=256))
    with torch.device("meta"):
        port = build_model("swin_unetr", task="segmentation", in_channels=PRED_CHANNELS)
    assert_same_names_and_shapes(reference, port)
    assert n_params(port) == n_params(reference) == 33_191_942  # Table 3: 33.2M
    del reference


def test_ts_satfire_2d_config_matches_monai():
    MonaiSwinUNETR = monai_nets().SwinUNETR
    # run_spatial_model.py, model "swinunetr2d": stock MONAI, feature size 48, batch norm.
    config = dict(in_channels=8, out_channels=2, spatial_dims=2, feature_size=48, norm_name="batch")
    torch.manual_seed(0)
    reference = MonaiSwinUNETR(img_size=(64, 64), **config)
    torch.manual_seed(0)
    port = build_model("swin_unetr", task="segmentation", in_channels=8, spatial_dims=2, image_size=64)
    assert type(port) is SwinUNETR
    assert_same_state(reference, port)
    assert n_params(port) == 25_151_996  # Table 3: 25.2M (the 2D model's size does not depend on the tile)
    randomise(reference, port, seed=5)
    torch.manual_seed(6)
    x = torch.randn(2, 8, 64, 64)
    compare_eval_and_train(reference, port, x, x)


def test_non_default_configs_match_monai():
    """MONAI's "merging" (with its repeated slices), "mergingv2", SwinUNETR-v2 blocks, dropout,
    stochastic depth, instance norm, gradient checkpointing and unnormalised features."""
    MonaiSwinUNETR = monai_nets().SwinUNETR
    configs = [
        dict(img_size=(32, 32, 32), in_channels=3, out_channels=2, feature_size=12, norm_name="batch", drop_rate=0.1, attn_drop_rate=0.1, dropout_path_rate=0.2),
        dict(img_size=(32, 64, 32), in_channels=3, out_channels=2, feature_size=12, num_heads=(1, 2, 2, 4), downsample="mergingv2", use_v2=True),
        dict(img_size=(64, 32), in_channels=2, out_channels=3, feature_size=24, spatial_dims=2, normalize=False, use_v2=True, depths=(2, 1, 2, 1)),
        dict(img_size=(32, 32), in_channels=2, out_channels=1, feature_size=12, spatial_dims=2, norm_name="batch", downsample="mergingv2", use_checkpoint=True, dropout_path_rate=0.1),
    ]
    for config in configs:
        torch.manual_seed(0)
        reference = MonaiSwinUNETR(**config)
        torch.manual_seed(0)
        port = SwinUNETR(**config)
        assert_same_state(reference, port)
        randomise(reference, port, seed=7)
        torch.manual_seed(8)
        x = torch.randn(2, config["in_channels"], *config["img_size"])
        compare_eval_and_train(reference, port, x, x, seed=9)  # same dropout / drop-path draws
