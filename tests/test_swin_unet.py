"""Swin-Unet: configurations, shapes, input validation and the Swin-T initialisation mapping.

Numerical equivalence with the official Swin-Unet lives in tests/oracle/test_swin_oracle.py.
"""

from __future__ import annotations

import pytest
import torch

from pyhazards.models import build_model
from pyhazards.models.swin_unet import SwinUnet

SMALL = dict(img_size=64, window_size=4)


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize(
    "in_channels, out_channels, history, expected",
    [
        (40, 1, 1, 27_224_964),  # WSTS+ "All" features, 1 day (paper: 27.2M)
        (3, 9, 1, 27_168_900),  # official Synapse configuration
        (40, 1, 5, 27_470_724),  # WSTS+ 5-day data-level fusion (200 input channels)
    ],
)
def test_swin_unet_parameter_counts(in_channels, out_channels, history, expected):
    model = build_model(
        "swin_unet", task="segmentation", in_channels=in_channels, out_channels=out_channels, history=history
    )
    assert _n_params(model) == expected


def test_swin_unet_forward_shapes_and_time_flattening():
    model = build_model("swin_unet", task="segmentation", in_channels=3, history=2, **SMALL).eval()
    x = torch.randn(2, 2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
        torch.testing.assert_close(out, model(x.flatten(1, 2)))
    assert out.shape == (2, 1, 64, 64)


def test_swin_unet_default_input_size_is_224():
    model = build_model("swin_unet", task="segmentation", in_channels=7).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 7, 224, 224)).shape == (1, 1, 224, 224)


@pytest.mark.parametrize("shape", [(2, 4, 64, 64), (2, 3, 32, 32), (3, 64, 64), (2, 3, 3, 3, 64, 64)])
def test_swin_unet_bad_input_shapes_raise(shape):
    model = build_model("swin_unet", task="segmentation", in_channels=3, **SMALL)
    with pytest.raises(ValueError):
        model(torch.randn(*shape))


def test_swin_unet_rejects_incompatible_geometry_and_task():
    with pytest.raises(ValueError):
        build_model("swin_unet", task="segmentation", in_channels=3, img_size=100)
    with pytest.raises(ValueError):
        build_model("swin_unet", task="segmentation", in_channels=3, img_size=112)  # odd 7x7 grid at stage 2
    with pytest.raises(ValueError):
        build_model("swin_unet", task="regression", in_channels=3)


def test_swin_unet_decoder_mirrors_encoder_depths():
    model = SwinUnet(in_chans=3, depths=(2, 2, 6, 2))
    assert [len(layer.blocks) for layer in model.layers] == [2, 2, 6, 2]
    assert [len(layer.blocks) for layer in list(model.layers_up)[1:]] == [6, 2, 2]


def test_swin_pretrained_mapping_fills_encoder_and_mirrored_decoder():
    torch.manual_seed(0)
    donor = SwinUnet(in_chans=3, **SMALL)
    swin_state = {
        key: torch.randn_like(value) if value.is_floating_point() else value
        for key, value in donor.state_dict().items()
        if key.startswith(("patch_embed.", "layers.", "norm."))
    }
    swin_state["head.weight"] = torch.randn(1000, 768)  # classifier of the Swin checkpoint, ignored

    model = SwinUnet(in_chans=5, **SMALL)  # 5-channel stem: shape differs, keeps its random init
    stem_before = model.patch_embed.proj.weight.detach().clone()
    model.load_swin_pretrained({"model": swin_state})
    state = model.state_dict()

    assert torch.equal(state["patch_embed.proj.weight"], stem_before)
    assert torch.equal(state["patch_embed.norm.weight"], swin_state["patch_embed.norm.weight"])
    for stage in range(4):
        encoder_key = f"layers.{stage}.blocks.0.mlp.fc1.weight"
        assert torch.equal(state[encoder_key], swin_state[encoder_key])
        if stage < 3:  # decoder step 3 - stage holds the mirrored blocks; step 0 has none
            assert torch.equal(state[f"layers_up.{3 - stage}.blocks.0.mlp.fc1.weight"], swin_state[encoder_key])
    assert torch.equal(state["norm.weight"], swin_state["norm.weight"])
