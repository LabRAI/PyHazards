"""UNETR: configurations, shapes and input validation.

Numerical equivalence with MONAI and TS-SatFire lives in tests/oracle/test_unetr_oracle.py.
"""

from __future__ import annotations

import pytest
import torch

from pyhazards.models import build_model
from pyhazards.models.unetr import UNETR, TemporalUNETR

SMALL = dict(feature_size=8, hidden_size=32, mlp_dim=64, num_heads=4)


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_ts_satfire_parameter_counts():
    with torch.device("meta"):
        # Prediction script (43 channels, 6 days, feature size 16, hidden 384, MLP 1536).
        assert _n_params(build_model("unetr", task="segmentation", in_channels=43)) == 28_816_866
        # AF/BA configuration of the paper text (8 channels, feature size 36): Table 3's 34.8M.
        assert _n_params(build_model("unetr", task="segmentation", in_channels=8, feature_size=36)) == 34_806_506
        # UNETR-2D (stock MONAI, 8 channels): Table 3's 23.52M.
        assert _n_params(build_model("unetr", task="segmentation", in_channels=8, spatial_dims=2)) == 23_521_186


def test_ts_satfire_modifications():
    model = build_model("unetr", task="segmentation", in_channels=43)
    assert isinstance(model, TemporalUNETR)
    assert model.patch_size == (1, 16, 16)
    assert model.kernel_size_up_down == (1, 2, 2)
    assert model.feat_size == (6, 16, 16)
    assert model.decoder3.conv_block.conv1.conv.weight.shape[-3:] == (1, 3, 3)
    assert model.decoder4.conv_block.conv1.conv.weight.shape[-3:] == (3, 3, 3)
    assert model.decoder2.transp_conv.conv.weight.shape[-3:] == (1, 2, 2)


def test_forward_shapes():
    model = build_model("unetr", task="segmentation", in_channels=5, history=3, image_size=32, **SMALL).eval()
    x = torch.randn(2, 3, 5, 32, 32)
    with torch.no_grad():
        assert model(x).shape == (2, 2, 32, 32)
        model.time_reduction = "none"
        assert model(x).shape == (2, 2, 3, 32, 32)
    model_2d = build_model("unetr", task="segmentation", in_channels=5, spatial_dims=2, image_size=32, **SMALL).eval()
    with torch.no_grad():
        assert model_2d(torch.randn(2, 5, 32, 32)).shape == (2, 2, 32, 32)


def test_temporal_wrapper_is_the_3d_network_averaged_over_time():
    torch.manual_seed(0)
    temporal = build_model("unetr", task="segmentation", in_channels=3, history=2, image_size=16, **SMALL)
    plain = UNETR(
        in_channels=3,
        out_channels=2,
        img_size=(2, 16, 16),
        norm_name="batch",
        patch_size=(1, 16, 16),
        kernel_size_up_down=(1, 2, 2),
        decoder3_kernel_size=(1, 3, 3),
        **SMALL,
    )
    plain.load_state_dict(temporal.state_dict(), strict=True)
    temporal.eval()
    plain.eval()
    x = torch.randn(2, 2, 3, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(temporal(x), plain(x.transpose(1, 2)).mean(2))


def test_monai_parameter_names():
    model = UNETR(in_channels=2, out_channels=1, img_size=32, spatial_dims=2, **SMALL)
    names = list(model.state_dict())
    assert names[:2] == ["vit.patch_embedding.position_embeddings", "vit.patch_embedding.patch_embeddings.weight"]
    assert "vit.blocks.11.attn.qkv.weight" in names
    assert "vit.blocks.0.mlp.linear1.weight" in names
    assert "encoder2.blocks.1.1.conv1.conv.weight" in names
    assert "decoder5.transp_conv.conv.weight" in names
    assert names[-2:] == ["out.conv.conv.weight", "out.conv.conv.bias"]


@pytest.mark.parametrize(
    "shape",
    [
        (2, 3, 32, 32),  # missing time axis
        (2, 3, 4, 32, 32),  # wrong channel count
        (2, 2, 3, 16, 32),  # not the img_size the model was built for
        (2, 3, 3, 32, 32),  # wrong number of days
    ],
)
def test_bad_input_shapes_raise(shape):
    model = build_model("unetr", task="segmentation", in_channels=3, history=2, image_size=32, **SMALL)
    with pytest.raises(ValueError):
        model(torch.randn(*shape))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(img_size=40),  # not divisible by the patch
        dict(img_size=32, num_heads=5),
        dict(img_size=32, dropout_rate=1.5),
        dict(img_size=32, patch_size=8),  # patch must be kernel_size_up_down ** 4
        dict(img_size=32, decoder3_kernel_size=2),
        dict(img_size=32, norm_name="group"),
        dict(img_size=32, proj_type="linear"),
        dict(img_size=32, spatial_dims=1),
        dict(img_size=(32, 32, 32)),
    ],
)
def test_bad_configurations_raise(kwargs):
    config = dict(in_channels=3, out_channels=1, spatial_dims=2, feature_size=8, hidden_size=32, mlp_dim=64, num_heads=4)
    config.update(kwargs)
    with pytest.raises(ValueError):
        UNETR(**config)


def test_builder_validates_task_and_dims():
    with pytest.raises(ValueError):
        build_model("unetr", task="classification", in_channels=4)
    with pytest.raises(ValueError):
        build_model("unetr", task="segmentation", in_channels=4, spatial_dims=1)
