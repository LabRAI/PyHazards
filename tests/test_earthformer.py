"""Earthformer: presets, parameter counts, shapes, layouts and input validation.

Numerical equivalence with the official code and checkpoints lives in
tests/oracle/test_earthformer_oracle.py.
"""

from __future__ import annotations

import pytest
import torch

from pyhazards.models import build_model
from pyhazards.models.earthformer import (
    EARTHFORMER_CONFIGS,
    CuboidTransformerModel,
    Earthformer,
    EarthformerSegmenter,
    earthformer_config,
)

SMALL = dict(history=4, in_channels=3, img_size=24)


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _count(config: str, **overrides) -> int:
    with torch.device("meta"):
        return _n_params(build_model("earthformer", task="forecasting", config=config, **overrides))


@pytest.mark.parametrize(
    "config, expected",
    [("sevir", 8_659_677), ("sevir_lr", 1_505_069), ("moving_mnist", 6_702_109), ("nbody", 6_702_109), ("enso", 1_394_325)],
)
def test_official_config_parameter_counts(config, expected):
    assert _count(config) == expected


PAPER_GLOBAL = dict(
    num_global_vectors=8,
    use_dec_self_global=False,
    use_dec_cross_global=False,
    use_global_vector_ffn=False,
    use_global_self_attn=False,
    separate_global_qkv=False,
)


def test_paper_parameter_counts():
    # Earthformer paper Tables 4-7: 15.1M / 13.1M (SEVIR), 7.61M / 6.61M (Moving-MNIST), 7.6M / 6.6M (ENSO).
    assert _count("sevir", enc_depth=[2, 2], dec_depth=[2, 2], **PAPER_GLOBAL) == 15_082_069
    assert _count("sevir", enc_depth=[2, 2], dec_depth=[2, 2], num_global_vectors=0) == 13_075_029
    assert _count("moving_mnist", pos_embed_type="t+h+w", **PAPER_GLOBAL) == 7_610_781
    assert _count("moving_mnist", pos_embed_type="t+h+w") == 6_611_997
    assert _count("enso", enc_depth=[4, 4], dec_depth=[4, 4], **PAPER_GLOBAL) == 7_600_077
    assert _count("enso", enc_depth=[4, 4], dec_depth=[4, 4]) == 6_601_293


def test_default_build_is_the_released_sevir_model():
    with torch.device("meta"):
        model = build_model("earthformer", task="forecasting")
    assert isinstance(model, Earthformer)
    assert model.input_shape == (13, 384, 384, 1)
    assert model.target_shape == (12, 384, 384, 1)
    assert model.input_shape_after_initial_downsample == (13, 32, 32, 128)
    assert model.encoder.get_mem_shapes() == [(13, 32, 32, 128), (13, 16, 16, 256)]
    assert _n_params(model) == 8_659_677


def test_config_expands_patterns_like_the_training_scripts():
    kwargs = earthformer_config("sevir")
    assert kwargs["enc_attn_patterns"] == ["axial", "axial"]
    assert kwargs["dec_self_attn_patterns"] == ["axial", "axial"]
    assert kwargs["dec_cross_attn_patterns"] == ["cross_1x1", "cross_1x1"]
    assert "self_pattern" not in kwargs
    kwargs = earthformer_config("moving_mnist", self_pattern="video_swin_2x8", enc_attn_patterns=["full", "axial"])
    assert kwargs["enc_attn_patterns"] == ["full", "axial"]
    assert kwargs["dec_self_attn_patterns"] == ["axial", "axial"]
    assert EARTHFORMER_CONFIGS["nbody"] == EARTHFORMER_CONFIGS["moving_mnist"]


def test_forecasting_shapes():
    model = build_model("earthformer", task="forecasting", horizon=2, out_channels=2, **SMALL).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 4, 3, 24, 24)).shape == (2, 2, 2, 24, 24)
    # Sizes that the 12x initial down-sampling does not divide are padded inside the patch merging.
    model = build_model("earthformer", task="forecasting", history=3, img_size=(26, 31)).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 1, 26, 31)).shape == (1, 12, 1, 26, 31)
    model = build_model("earthformer", task="forecasting", config="moving_mnist", img_size=16, history=3, horizon=2).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 1, 16, 16)).shape == (2, 2, 1, 16, 16)


def test_segmentation_returns_one_mask_logit_map():
    model = build_model("earthformer", task="segmentation", **SMALL)
    assert isinstance(model, EarthformerSegmenter)
    assert model.target_shape == (1, 24, 24, 1)
    model.eval()
    with torch.no_grad():
        assert model(torch.randn(2, 4, 3, 24, 24)).shape == (2, 1, 24, 24)
    model.train()
    loss = model(torch.randn(2, 4, 3, 24, 24)).square().mean()
    loss.backward()
    assert model.encoder.blocks[0][0].attn_l[0].qkv.weight.grad is not None


def test_wrappers_keep_core_parameter_names_and_layout():
    torch.manual_seed(0)
    kwargs = earthformer_config("sevir", input_shape=[4, 24, 24, 3], target_shape=[2, 24, 24, 2])
    core = CuboidTransformerModel(**kwargs)
    torch.manual_seed(0)
    wrapped = Earthformer(**kwargs)
    assert list(core.state_dict()) == list(wrapped.state_dict())
    for key, value in core.state_dict().items():
        assert torch.equal(value, wrapped.state_dict()[key]), key
    core.eval()
    wrapped.eval()
    x = torch.randn(2, 4, 3, 24, 24)
    with torch.no_grad():
        expected = core(x.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        torch.testing.assert_close(wrapped(x), expected)

    segmenter = build_model("earthformer", task="segmentation", **SMALL)
    reference = CuboidTransformerModel(**earthformer_config("sevir", input_shape=[4, 24, 24, 3], target_shape=[1, 24, 24, 1]))
    assert list(segmenter.state_dict()) == list(reference.state_dict())


def test_local_state_dict_loads_strictly(tmp_path):
    torch.manual_seed(0)
    source = build_model("earthformer", task="forecasting", **SMALL)
    path = tmp_path / "earthformer.pt"
    torch.save(source.state_dict(), path)
    torch.manual_seed(1)
    loaded = build_model("earthformer", task="forecasting", pretrained=path, **SMALL)
    for key, value in source.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key]), key


def test_input_validation():
    model = build_model("earthformer", task="forecasting", **SMALL)
    with pytest.raises(ValueError, match=r"\(batch, time, channels, height, width\)"):
        model(torch.randn(2, 4, 24, 24, 3))  # channels-last input
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(4, 3, 24, 24))
    with pytest.raises(ValueError, match="shape"):
        CuboidTransformerModel.forward(model, torch.randn(2, 4, 3, 24, 24))  # the core takes (B, T, H, W, C)


def test_builder_validation():
    with pytest.raises(ValueError, match="task"):
        build_model("earthformer", task="classification")
    with pytest.raises(ValueError, match="config"):
        build_model("earthformer", task="forecasting", config="sevir_v2")
    with pytest.raises(ValueError, match="Unknown Earthformer arguments"):
        build_model("earthformer", task="forecasting", base_unit=64)
    with pytest.raises(ValueError, match="input_shape"):
        build_model("earthformer", task="forecasting", input_shape=[13, 384, 384, 1])
    with pytest.raises(ValueError, match="one frame"):
        build_model("earthformer", task="segmentation", horizon=2)
    with pytest.raises(ValueError, match="positive"):
        build_model("earthformer", task="forecasting", history=0)
    with pytest.raises(ValueError, match="pattern"):
        build_model("earthformer", task="forecasting", self_pattern="diagonal", **SMALL)
    # Official checkpoints only fit their own configuration (checked before any download).
    with pytest.raises(ValueError, match="checkpoint"):
        build_model("earthformer", task="forecasting", pretrained="sevir", **SMALL)
    with pytest.raises(ValueError, match="checkpoint"):
        build_model("earthformer", task="segmentation", pretrained="sevir")
    with pytest.raises(ValueError, match="checkpoint"):
        build_model("earthformer", task="forecasting", config="moving_mnist", pretrained="icarenso2021")
