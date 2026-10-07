"""DeepLabV3 (smp port) in Shadrin et al.'s configuration; the oracle test compares it with smp."""

from __future__ import annotations

import pytest
import torch

from pyhazards.models import build_model
from pyhazards.models.deeplabv3 import DeepLabV3


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_deeplabv3_shadrin_defaults():
    model = build_model("deeplabv3", task="segmentation")
    assert model.in_channels == 58
    assert _n_params(model) == 13_220_609
    assert model.encoder.output_stride == 8
    model.eval()
    with torch.no_grad():
        assert model(torch.randn(2, 58, 32, 32)).shape == (2, 1, 32, 32)
    aspp_rates = [m.dilation[0] for m in model.decoder[0].convs.modules() if isinstance(m, torch.nn.Conv2d) and m.kernel_size == (3, 3)]
    assert aspp_rates == [12, 24, 36]


def test_deeplabv3_dilates_last_stages_like_smp():
    model = DeepLabV3(in_channels=3)
    assert _n_params(model) == 26_007_105  # smp.DeepLabV3() defaults: ResNet-34, five stages
    for layer, rate in ((model.encoder.layer3, 2), (model.encoder.layer4, 4)):
        for conv in (m for m in layer.modules() if isinstance(m, torch.nn.Conv2d)):
            assert conv.stride == (1, 1) and conv.dilation == (rate, rate)
            assert conv.padding == ((conv.kernel_size[0] // 2) * rate,) * 2
    assert model.encoder.layer2[0].conv1.stride == (2, 2)
    model.eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 64, 48)).shape == (1, 1, 64, 48)


def test_deeplabv3_validation():
    model = build_model("deeplabv3", task="segmentation", in_channels=4)
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(2, 1, 4, 32, 32))
    with pytest.raises(ValueError, match="channels"):
        model(torch.randn(2, 5, 32, 32))
    with pytest.raises(ValueError, match="divisible"):
        model(torch.randn(2, 4, 36, 32))
    with pytest.raises(ValueError):
        DeepLabV3(encoder_depth=2)
    with pytest.raises(ValueError):
        DeepLabV3(encoder_name="resnet50")
    with pytest.raises(ValueError):
        DeepLabV3(encoder_weights="ssl")
    with pytest.raises(ValueError):
        build_model("deeplabv3", task="classification")
