"""ASUFM: configurations, shapes and input validation.

Numerical equivalence with the official fire-asufm code lives in tests/oracle/test_swin_oracle.py.
"""

from __future__ import annotations

import pytest
import torch

from pyhazards.models import build_model


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("in_channels, expected", [(6, 35_047_840), (12, 35_057_056)])
def test_asufm_parameter_counts(in_channels, expected):
    # get_asfum_6_configs / get_asufm_12_configs of the official code
    assert _n_params(build_model("asufm", task="segmentation", in_channels=in_channels)) == expected


def test_asufm_forward_shape():
    model = build_model("asufm", task="segmentation", in_channels=12).eval()
    x = torch.randn(2, 12, 64, 64)
    with torch.no_grad():
        out = model(x)
        torch.testing.assert_close(model(x[:, None]), out)  # a length-one time axis is accepted
    assert out.shape == (2, 1, 64, 64)


def test_asufm_decoder_modulation_is_unused():
    model = build_model("asufm", task="segmentation").eval()
    x = torch.randn(1, 6, 64, 64)
    with torch.no_grad():
        before = model(x)
        for name, param in model.named_parameters():
            if name.startswith("decoder.") and ".modulation." in name:
                param.add_(1.0)
            if name.startswith(("patch_embed.", "decoder.norm.")):
                param.add_(1.0)
        torch.testing.assert_close(model(x), before)
        model.encoder.layers[0].blocks[0].modulation.proj.weight.add_(0.1)
        assert not torch.allclose(model(x), before)


@pytest.mark.parametrize("shape", [(2, 12, 64, 64), (2, 6, 32, 32), (6, 64, 64), (2, 2, 6, 64, 64)])
def test_asufm_bad_input_shapes_raise(shape):
    model = build_model("asufm", task="segmentation", in_channels=6)
    with pytest.raises(ValueError):
        model(torch.randn(*shape))


def test_asufm_rejects_other_tasks():
    with pytest.raises(ValueError):
        build_model("asufm", task="forecasting")
