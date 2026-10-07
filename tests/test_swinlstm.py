"""SwinLSTM without the reference code: sizes, shapes, rollout semantics and input validation.

The comparison with the official implementation and weights is in tests/oracle/test_swinlstm_oracle.py.
"""

from __future__ import annotations

import pytest
import torch

from pyhazards.models import build_model
from pyhazards.models.swinlstm import SwinLSTM, SwinLSTMSegmenter

SMALL = dict(img_size=16, embed_dim=16, depths_downsample=[2, 2], depths_upsample=[2, 2], num_heads=[2, 4], window_size=2)


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_official_configurations():
    deep = build_model("swinlstm", task="forecasting")  # Moving-MNIST SwinLSTM-D (official weights)
    assert _n_params(deep) == 20_191_969
    assert len(deep.state_dict()) == 286
    assert [len(cell.Swin.layers) for cell in deep.Downsample.layers] == [2, 6]
    # Upsample indexes depths_upsample=(6, 2) from the deepest level: 2 blocks at 8x8, then 6 at 16x16.
    assert [len(cell.Swin.layers) for cell in deep.Upsample.layers] == [2, 6]
    assert [cell.Swin.layers[0].input_resolution for cell in deep.Upsample.layers] == [(8, 8), (16, 16)]
    base = build_model("swinlstm", task="forecasting", variant="b")
    assert _n_params(base) == 2_778_417
    assert sorted({k.split(".")[0] for k in base.state_dict()}) == ["ST"]


def test_stochastic_depth_rates_follow_the_reference():
    deep = build_model("swinlstm", task="forecasting")
    rates = lambda cell: [getattr(b.drop_path, "drop_prob", 0.0) for b in cell.Swin.layers]  # noqa: E731
    down = [rates(cell) for cell in deep.Downsample.layers]
    assert down[0][0] == 0.0 and down[1][-1] == pytest.approx(0.1)
    # Decoder cells use their level's slice of the schedule in reverse order (flag=0 in the reference).
    up = [rates(cell) for cell in deep.Upsample.layers]
    assert up[0] == sorted(up[0], reverse=True) and up[1][-1] == 0.0
    base = build_model("swinlstm", task="forecasting", variant="b")
    assert {rate for rate in rates(base.ST.layers[0])} == {0.1}


@pytest.mark.parametrize("variant", ["d", "b"])
def test_forecasting_shapes_and_rollout(variant):
    kwargs = dict(SMALL, depths=[2]) if variant == "b" else SMALL
    model = build_model("swinlstm", task="forecasting", variant=variant, in_channels=2, num_output_frames=3, **kwargs).eval()
    x = torch.rand(2, 4, 2, 16, 16)
    with torch.no_grad():
        future = model(x)
        everything = model(x, include_warmup=True)
        assert future.shape == (2, 3, 2, 16, 16)
        assert everything.shape == (2, 6, 2, 16, 16)
        torch.testing.assert_close(everything[:, 3:], future)
        assert ((future > 0) & (future < 1)).all()  # sigmoid output
        # The first prediction only depends on the input frames.
        torch.testing.assert_close(model(x, num_output_frames=1), future[:, :1])
        # step() is the reference single-step forward.
        states = ()
        for t in range(4):
            output, *states = model.step(x[:, t], *states)
        torch.testing.assert_close(output, future[:, 0])


def test_segmentation_is_the_one_step_rollout_before_the_sigmoid():
    forecaster = build_model("swinlstm", task="forecasting", in_channels=1, **SMALL).eval()
    segmenter = build_model("swinlstm", task="segmentation", in_channels=1, **SMALL).eval()
    assert isinstance(segmenter, SwinLSTMSegmenter)
    segmenter.load_state_dict(forecaster.state_dict(), strict=True)  # same keys, no prefix
    x = torch.rand(2, 3, 1, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(torch.sigmoid(segmenter(x)), forecaster(x, num_output_frames=1)[:, 0])


def test_segmentation_with_many_input_channels():
    model = build_model("swinlstm", task="segmentation", in_channels=7, **SMALL)
    assert model.Upsample.Unembed.Conv.out_channels == 1
    assert model.Upsample.patch_embed.proj.in_channels == 7
    logits = model(torch.randn(2, 3, 7, 16, 16))
    assert logits.shape == (2, 1, 16, 16)
    logits.mean().backward()


def test_input_validation():
    model = build_model("swinlstm", task="forecasting", in_channels=2, **SMALL)
    with pytest.raises(ValueError, match="shape"):
        model(torch.rand(2, 2, 16, 16))
    with pytest.raises(ValueError, match="frames of shape"):
        model(torch.rand(2, 3, 1, 16, 16))
    with pytest.raises(ValueError, match="frames of shape"):
        model(torch.rand(2, 3, 2, 32, 32))
    with pytest.raises(ValueError, match="num_output_frames"):
        model(torch.rand(2, 3, 2, 16, 16), num_output_frames=0)
    with pytest.raises(ValueError, match="out_chans == in_chans"):
        SwinLSTM(in_chans=2, out_chans=1, **SMALL)(torch.rand(1, 2, 2, 16, 16), num_output_frames=2)
    with pytest.raises(ValueError, match="state arguments"):
        model.step(torch.rand(1, 2, 16, 16), None)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(patch_size=4), "patch_size must be 2"),
        (dict(img_size=18), "cannot be halved"),
        (dict(img_size=24, window_size=4), "not divisible by window size"),
        (dict(variant="c"), "variant"),
        (dict(depths_upsample=[2]), "equal length"),
        (dict(num_heads=[2]), "num_heads"),
    ],
)
def test_configuration_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_model("swinlstm", task="forecasting", **{**SMALL, **kwargs})


def test_unknown_task():
    with pytest.raises(ValueError, match="forecasting"):
        build_model("swinlstm", task="classification")
