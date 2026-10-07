"""TS-SatFire prediction baselines: configurations, parameter counts and shapes.

The comparison with TS-SatFire's own models lives in tests/oracle/test_ts_satfire_oracle.py.
"""

from __future__ import annotations

import pytest
import torch

from pyhazards.models import TS_SATFIRE_BASELINES, build_model
from pyhazards.models.attention_unet import TemporalAttentionUnet
from pyhazards.models.swin_unetr import TemporalSwinUNETR
from pyhazards.models.unet3d import TemporalUNet
from pyhazards.models.unetr import TemporalUNETR

EXPECTED = {
    "unet3d": (TemporalUNet, 31_712_970),
    "attention_unet3d": (TemporalAttentionUnet, 94_555_506),
    "unetr3d": (TemporalUNETR, 28_816_866),
    "swinunetr3d": (TemporalSwinUNETR, 33_191_942),
}


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_baselines_are_complete():
    assert set(TS_SATFIRE_BASELINES) == set(EXPECTED)


@pytest.mark.parametrize("baseline", TS_SATFIRE_BASELINES)
def test_default_configuration_parameter_counts(baseline):
    cls, count = EXPECTED[baseline]
    with torch.device("meta"):
        model = build_model("ts_satfire", task="segmentation", baseline=baseline)
    assert isinstance(model, cls)
    assert _n_params(model) == count


def test_default_is_swinunetr3d():
    model = build_model("ts_satfire", task="segmentation", history=2, image_size=32)
    assert isinstance(model, TemporalSwinUNETR)
    assert model.window_size == (2, 4, 4)


# attention_unet3d (94.5M parameters) is left out for speed; tests/test_attention_unet.py covers its shapes.
@pytest.mark.parametrize("baseline", ["unet3d", "unetr3d", "swinunetr3d"])
def test_forward_shapes(baseline):
    model = build_model("ts_satfire", task="segmentation", baseline=baseline, in_channels=6, history=2, image_size=32)
    model.eval()
    with torch.no_grad():
        assert model(torch.randn(1, 2, 6, 32, 32)).shape == (1, 2, 32, 32)


def test_options():
    with torch.device("meta"):
        v0 = build_model("ts_satfire", task="segmentation", baseline="unetr3d", unetr_version="v0")
        assert _n_params(v0) == 97_922_658
        wide = build_model("ts_satfire", task="segmentation", baseline="unetr3d", feature_size=36)
        assert _n_params(wide) == 38_282_426
        heads = build_model("ts_satfire", task="segmentation", baseline="swinunetr3d", num_heads=6)
        assert _n_params(heads) == 33_204_878
        short = build_model("ts_satfire", task="segmentation", baseline="swinunetr3d", history=2)
        assert short.swinViT.stage_scales[0] == (2, 2, 2)


def test_bad_arguments_raise():
    with pytest.raises(ValueError):
        build_model("ts_satfire", task="classification")
    with pytest.raises(ValueError):
        build_model("ts_satfire", task="segmentation", baseline="segmenter")
    with pytest.raises(ValueError):
        build_model("ts_satfire", task="segmentation", baseline="unetr3d", unetr_version="v1")
    with pytest.raises(ValueError):
        build_model("ts_satfire", task="segmentation", history=0)
