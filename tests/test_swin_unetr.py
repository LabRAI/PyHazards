"""SwinUNETR: configurations, shapes, TS-SatFire's merging plan and input validation.

Numerical equivalence with MONAI and TS-SatFire lives in tests/oracle/test_swin_unetr_oracle.py.
"""

from __future__ import annotations

import pytest
import torch

from pyhazards.models import build_model
from pyhazards.models.swin_unetr import AdaptivePatchMerging, PatchMerging, PatchMergingV2, SwinUNETR, TemporalSwinUNETR

SMALL = dict(feature_size=12, num_heads=3)


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_ts_satfire_parameter_counts():
    with torch.device("meta"):
        # SwinUNETR-3D prediction model (43 channels, 6 days, feature size 36, 3 heads): Table 3's 33.2M.
        assert _n_params(build_model("swin_unetr", task="segmentation", in_channels=43)) == 33_191_942
        # SwinUNETR-2D (stock MONAI, feature size 48, 8 channels): Table 3's 25.2M.
        assert _n_params(build_model("swin_unetr", task="segmentation", in_channels=8, spatial_dims=2)) == 25_151_996


@pytest.mark.parametrize(
    "history, scales",
    [
        (6, [(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)]),
        (4, [(2, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2)]),
        (2, [(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)]),
        (3, [(1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)]),
    ],
)
def test_ts_satfire_merging_plan(history, scales):
    model = build_model("swin_unetr", task="segmentation", in_channels=4, history=history, image_size=32, **SMALL)
    assert isinstance(model, TemporalSwinUNETR)
    assert model.swinViT.stage_scales == scales
    assert model.window_size == (history, 4, 4)
    assert model.patch_size == (1, 2, 2)
    x = torch.randn(1, history, 4, 32, 32)
    model.eval()
    with torch.no_grad():
        assert model(x).shape == (1, 2, 32, 32)
        model.time_reduction = "none"
        assert model(x).shape == (1, 2, history, 32, 32)


def test_patch_merging_orders():
    x = torch.arange(8.0).reshape(1, 2, 2, 2, 1)  # value = 4 d + 2 h + w
    with torch.no_grad():
        for module, expected in (
            (PatchMergingV2(dim=1), [0, 1, 2, 3, 4, 5, 6, 7]),
            (PatchMerging(dim=1), [0, 4, 2, 1, 5, 2, 1, 7]),  # MONAI's v0.9.0 order, with repeats
            (AdaptivePatchMerging(dim=1, merge=(True, True, True)), [0, 4, 2, 1, 6, 5, 3, 7]),
            (AdaptivePatchMerging(dim=1, merge=(False, True, True)), [[0, 2, 1, 3], [4, 6, 5, 7]]),
        ):
            module.norm = torch.nn.Identity()
            module.reduction = torch.nn.Identity()
            assert module(x).flatten().tolist() == torch.tensor(expected, dtype=torch.float32).flatten().tolist()


def test_ts_satfire_naming_and_attention_versions():
    model = build_model("swin_unetr", task="segmentation", in_channels=4, history=2, image_size=32, **SMALL)
    names = list(model.state_dict())
    assert names[-2:] == ["out.0.conv.conv.weight", "out.0.conv.conv.bias"]  # TS-SatFire wraps the output block
    assert "swinViT.layers1.0.blocks.0.attn.relative_position_bias_table" in names
    assert "swinViT.layers1.0.downsample.norm.weight" in names
    v2 = build_model("swin_unetr", task="segmentation", in_channels=4, history=2, image_size=32, attn_version="v2", **SMALL)
    assert "swinViT.layers1.0.blocks.0.attn.cpb_mlp.0.weight" in v2.state_dict()
    model_2d = build_model("swin_unetr", task="segmentation", in_channels=4, spatial_dims=2, image_size=32, feature_size=12)
    assert list(model_2d.state_dict())[-2:] == ["out.conv.conv.weight", "out.conv.conv.bias"]


def test_monai_2d_forward_shape():
    model = build_model("swin_unetr", task="segmentation", in_channels=3, spatial_dims=2, image_size=64, feature_size=12).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 64, 32)).shape == (2, 2, 64, 32)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 4, 32, 32),  # missing time axis
        (2, 2, 5, 32, 32),  # wrong channel count
        (2, 2, 4, 30, 32),  # not divisible by the patch / merges
        (2, 3, 4, 32, 32),  # 3 days do not fit the merging plan built for 2
    ],
)
def test_bad_input_shapes_raise(shape):
    model = build_model("swin_unetr", task="segmentation", in_channels=4, history=2, image_size=32, **SMALL)
    with pytest.raises(ValueError):
        model(torch.randn(*shape))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(downsample="patchmerging"),
        dict(attn_version="ar"),  # TS-SatFire's autoregressive attention is not ported
        dict(feature_size=16),  # MONAI's feature_size % 12 check
        dict(num_heads=(5, 5, 5, 5)),
        dict(depths=(2, 2, 2)),
        dict(img_size=(30, 32)),
        dict(norm_name="group"),
        dict(drop_rate=2.0),
        dict(downsample="ts_satfire"),  # 3D only
    ],
)
def test_bad_configurations_raise(kwargs):
    config = dict(img_size=(32, 32), in_channels=3, out_channels=1, feature_size=12, spatial_dims=2)
    config.update(kwargs)
    with pytest.raises(ValueError):
        SwinUNETR(**config)


def test_ts_satfire_configuration_errors():
    with pytest.raises(ValueError):  # the merging plan needs img_size
        SwinUNETR(None, 3, 2, feature_size=12, downsample="ts_satfire", patch_size=(1, 2, 2))
    with pytest.raises(ValueError):  # TS-SatFire's img_size check: 48 is not divisible by 2**5
        SwinUNETR((2, 48, 32), 3, 2, feature_size=12, downsample="ts_satfire", patch_size=(1, 2, 2))
    with pytest.raises(ValueError):  # Swin-V2 attention divides by window - 1
        SwinUNETR((2, 32, 32), 3, 2, feature_size=12, downsample="ts_satfire", patch_size=(1, 2, 2), window_size=(1, 4, 4), attn_version="v2")


def test_builder_validates_task_and_dims():
    with pytest.raises(ValueError):
        build_model("swin_unetr", task="classification", in_channels=4)
    with pytest.raises(ValueError):
        build_model("swin_unetr", task="segmentation", in_channels=4, spatial_dims=1)
