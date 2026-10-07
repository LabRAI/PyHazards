"""Earthfarseer: parameter counts, shapes, the segmentation adaptation and input validation.

Numerical equivalence with the official code lives in tests/oracle/test_earthfarseer_oracle.py.
"""

from __future__ import annotations

import pytest
import torch

from pyhazards.models import build_model
from pyhazards.models.earthfarseer import Earthfarseer

# Small enough for CPU tests (the fixed-size SimVP skip branch still has 13.9M parameters).
SMALL = dict(history=3, img_size=32, hid_S=8, hid_T=16, N_S=4, N_T=2, gf_embed_dim=32, gf_depth=1)


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_earthfarseer_default_parameter_count():
    with torch.device("meta"):
        model = build_model("earthfarseer", task="forecasting")
    assert model.shape_in == (10, 1, 64, 64)
    assert _n_params(model) == 157_518_863  # official Earthfarseer_model(shape_in=(10, 1, 64, 64))
    parts = {name: _n_params(child) for name, child in model.named_children()}
    assert parts == {
        "fotf_encoder": 78_246_509,
        "skip_conneciton": 13_866_401,
        "latent_projection": 7_088_640,
        "enc": 7_088_640,
        "TeDev_block": 39_425_536,
        "dec": 11_803_137,
    }


def test_earthfarseer_forecasting_shapes():
    model = build_model("earthfarseer", task="forecasting", in_channels=2, **SMALL).eval()
    x = torch.randn(2, 3, 2, 32, 32)
    with torch.no_grad():
        assert model(x).shape == (2, 3, 2, 32, 32)


def test_earthfarseer_segmentation_adaptation_keeps_reference_names():
    torch.manual_seed(0)
    core = build_model("earthfarseer", task="forecasting", in_channels=2, **SMALL)
    torch.manual_seed(0)
    seg = build_model("earthfarseer", task="segmentation", in_channels=2, **SMALL)
    core_state, seg_state = core.state_dict(), seg.state_dict()
    assert list(seg_state) == list(core_state) + ["segmentation_head.weight", "segmentation_head.bias"]
    for key, value in core_state.items():
        assert torch.equal(value, seg_state[key]), key

    core.eval()
    seg.eval()
    x = torch.randn(2, 3, 2, 32, 32)
    with torch.no_grad():
        logits = seg(x)
        torch.testing.assert_close(logits, seg.segmentation_head(core(x)[:, 0]))
    assert logits.shape == (2, 1, 32, 32)


def test_earthfarseer_unused_reference_modules_get_no_gradient():
    model = build_model("earthfarseer", task="forecasting", **SMALL)
    model(torch.randn(1, 3, 1, 32, 32)).sum().backward()
    assert all(p.grad is None for p in model.enc.parameters())  # second encoder, never called
    assert model.fotf_encoder.gf_block.blocks[0].mlp.fc2.weight.grad is None  # replaced by avg pool
    assert model.fotf_encoder.gf_block.blocks[0].mlp.fc1.weight.grad is not None


def test_earthfarseer_runs_where_the_reference_token_grid_formula_fails():
    # The reference sizes the TeDev grid as int(H / 4) + 1 when H % 3 == 0 (48 -> 13, latent 12)
    # and only supports square inputs; the port uses the true latent size.
    model = build_model("earthfarseer", task="forecasting", **{**SMALL, "img_size": 48}).eval()
    assert (model.H1, model.W1) == (12, 12)
    rect = build_model("earthfarseer", task="forecasting", **{**SMALL, "img_size": (32, 48)}).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 1, 48, 48)).shape == (1, 3, 1, 48, 48)
        assert rect(torch.randn(1, 3, 1, 32, 48)).shape == (1, 3, 1, 32, 48)


@pytest.mark.parametrize("shape", [(1, 3, 1, 32, 48), (1, 4, 1, 32, 32), (1, 3, 2, 32, 32), (3, 1, 32, 32)])
def test_earthfarseer_bad_input_shapes_raise(shape):
    model = build_model("earthfarseer", task="forecasting", **SMALL)
    with pytest.raises(ValueError):
        model(torch.randn(*shape))


@pytest.mark.parametrize(
    "overrides", [dict(img_size=40), dict(img_size=(32, 24)), dict(N_T=1), dict(N_S=0), dict(history=0)]
)
def test_earthfarseer_rejects_unsupported_configurations(overrides):
    with pytest.raises(ValueError):
        build_model("earthfarseer", task="forecasting", **{**SMALL, **overrides})


def test_earthfarseer_rejects_unknown_task():
    with pytest.raises(ValueError):
        build_model("earthfarseer", task="classification", **SMALL)


def test_earthfarseer_class_defaults_are_the_official_defaults():
    with torch.device("meta"):
        model = Earthfarseer()
    assert model.shape_in == (10, 1, 64, 64)
    assert len(model.TeDev_block.enc) == len(model.TeDev_block.dec) == 8
    assert len(model.TeDev_block.blocks) == 12
    assert len(model.fotf_encoder.gf_block.blocks) == 12
    assert model.fotf_encoder.gf_block.embed_dim == 768
    assert model.fotf_encoder.num_interactions == 3
