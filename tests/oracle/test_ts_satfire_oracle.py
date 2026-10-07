"""The ts_satfire preset checked against TS-SatFire's prediction-script models.

Each baseline is built the way ``run_spatial_temp_model_pred.py`` (zhaoyutim/TS-SatFire, pinned in
repos.yaml; no LICENSE, test oracle only) builds it -- 43 input channels, T input days, 256x256
tiles, two classes, logits averaged over time -- and compared with ``build_model("ts_satfire",
baseline=...)``: parameter counts at the full size, then names, seeded initial values and eval /
train-mode outputs on 32x32 tiles with T = 2 (CPU speed; U-Nets are fully convolutional and
UNETR / SwinUNETR are built for the smaller size on both sides).
"""

from __future__ import annotations

import pytest
import torch

from pyhazards.models import build_model
from tssatfire_oracle_helpers import (
    PRED_CHANNELS,
    TS_SATFIRE_CHANNELS,
    assert_same_names_and_shapes,
    assert_same_state,
    compare_eval_and_train,
    n_params,
    randomise,
    ts_satfire_module,
)

EXPECTED_PARAMETERS = {
    "unet3d": 31_712_970,  # Table 3: 31.7M
    "attention_unet3d": 94_555_506,  # Table 3: 94.5M
    "unetr3d": 28_816_866,  # Table 3 lists 34.8M, the AF/BA configuration (see test_unetr_oracle.py)
    "swinunetr3d": 33_191_942,  # Table 3: 33.2M
}


def _reference(baseline: str, history: int, size: int) -> torch.nn.Module:
    """The model of run_spatial_temp_model_pred.py (n_channel=43, -ed 36 and -nh 3 for SwinUNETR)."""
    n_channel, num_classes = PRED_CHANNELS, 2
    if baseline == "unet3d":
        UNet = ts_satfire_module("unet").UNet
        return UNet(spatial_dims=3, in_channels=n_channel, out_channels=num_classes, channels=TS_SATFIRE_CHANNELS, strides=(1, 2, 2))
    if baseline == "attention_unet3d":
        AttentionUnet = ts_satfire_module("attentionunet").AttentionUnet
        return AttentionUnet(
            spatial_dims=3, in_channels=n_channel, out_channels=num_classes, channels=TS_SATFIRE_CHANNELS, strides=(1, 2, 2)
        )
    image_size = (history, size, size)
    if baseline == "unetr3d":
        UNETR = ts_satfire_module("unetr.unetr").UNETR
        # The script's kernel_size_up_down=(1, 2, 2) is hard-coded in the released file.
        return UNETR(
            in_channels=n_channel,
            out_channels=num_classes,
            img_size=image_size,
            spatial_dims=3,
            norm_name="batch",
            feature_size=16,
            patch_size=(1, 16, 16),
            hidden_size=384,
            mlp_dim=1536,
        )
    SwinUNETR = ts_satfire_module("swinunetr.swinunetr").SwinUNETR
    return SwinUNETR(
        image_size=image_size,
        patch_size=(1, 2, 2),
        window_size=(history, 4, 4),
        in_channels=n_channel,
        out_channels=2,
        depths=(2, 2, 2, 2),
        num_heads=(3, 3, 3, 3),
        feature_size=36,
        norm_name="batch",
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        attn_version="v1",
        normalize=True,
        use_checkpoint=False,
        spatial_dims=3,
    )


@pytest.mark.parametrize("baseline", sorted(EXPECTED_PARAMETERS))
def test_parameter_counts_at_full_size(baseline):
    if baseline == "swinunetr3d":
        reference = _reference(baseline, history=6, size=256)  # calls Tensor.item(): not on meta
    else:
        with torch.device("meta"):
            reference = _reference(baseline, history=6, size=256)
    with torch.device("meta"):
        port = build_model("ts_satfire", task="segmentation", baseline=baseline)
    assert_same_names_and_shapes(reference, port)
    assert n_params(port) == n_params(reference) == EXPECTED_PARAMETERS[baseline]


@pytest.mark.parametrize("baseline", sorted(EXPECTED_PARAMETERS))
def test_baselines_match_reference(baseline):
    torch.manual_seed(0)
    reference = _reference(baseline, history=2, size=32)
    torch.manual_seed(0)
    port = build_model("ts_satfire", task="segmentation", baseline=baseline, history=2, image_size=32)
    assert_same_state(reference, port)
    randomise(reference, port, seed=1)
    torch.manual_seed(2)
    x = torch.randn(2, 2, PRED_CHANNELS, 32, 32)  # (batch, days, channels, H, W)
    compare_eval_and_train(reference, port, x.transpose(1, 2).contiguous(), x, reduce_time=True)
