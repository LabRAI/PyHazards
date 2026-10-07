"""U-Net (Ronneberger et al., MICCAI 2015) rebuilt from the paper; the oracle test covers the release."""

from __future__ import annotations

import pytest
import torch

from pyhazards.models import build_model
from pyhazards.models.unet import UNet, unet_input_size, unet_output_size

SMALL = (2, 4, 8, 16, 32)


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_unet_parameter_counts():
    paper = build_model("unet", task="segmentation")
    assert _n_params(paper) == 31_030_658  # 1 input channel, 2 classes (paper Fig. 1)
    convs = [m for m in paper.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))]
    assert len(convs) == 23  # "In total the network has 23 convolutional layers"
    assert [m.out_channels for m in convs if isinstance(m, torch.nn.ConvTranspose2d)] == [512, 256, 128, 64]
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in paper.modules())
    release = build_model("unet", task="segmentation", variant="phseg_v5")
    assert _n_params(release) == 31_100_354


def test_unet_feature_map_sizes_follow_fig1():
    model = UNet(1, 2, channels=SMALL).eval()
    sizes = {}
    for name in ["conv_d0b_c", "conv_d1b_c", "conv_d2b_c", "conv_d3b_c", "conv_d4b_c", "conv_u3c_d", "conv_u2c_d", "conv_u1c_d", "conv_u0c_d"]:
        getattr(model, name).register_forward_hook(lambda m, i, o, name=name: sizes.__setitem__(name, o.shape[-1]))
    with torch.no_grad():
        out = model(torch.randn(1, 1, 572, 572))
    assert out.shape == (1, 2, 388, 388)
    assert list(sizes.values()) == [568, 280, 136, 64, 28, 52, 100, 196, 388]


def test_unet_size_helpers():
    assert unet_output_size(572) == 388
    assert unet_input_size(388) == 572
    assert unet_output_size(188) == 4 and unet_input_size(1) == 188
    for output in range(1, 600, 7):
        size = unet_input_size(output)
        assert unet_output_size(size) >= output
        assert size == 188 or unet_output_size(size - 16) < output
    for bad in (187, 189, 200, 571, 64):
        with pytest.raises(ValueError, match="16 \\* k \\+ 60"):
            unet_output_size(bad)
    assert unet_output_size(64, padding="same") == 64
    with pytest.raises(ValueError):
        unet_output_size(60, padding="same")


def test_unet_rectangular_and_same_padding():
    model = UNet(3, 1, channels=SMALL).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 188, 220)).shape == (2, 1, 4, 36)
    same = build_model("unet", task="segmentation", in_channels=12, out_channels=1, padding="same", channels=SMALL).eval()
    with torch.no_grad():
        assert same(torch.randn(2, 12, 64, 48)).shape == (2, 1, 64, 48)
    with pytest.raises(ValueError):
        same(torch.randn(2, 12, 64, 40))


def test_unet_initialisation_and_dropout():
    torch.manual_seed(0)
    model = build_model("unet", task="segmentation")
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            assert torch.count_nonzero(param) == 0, name
        elif param.numel() >= 100_000:
            fan_in = param.numel() / param.size(0)  # (out, in, k, k) conv; (in, out, k, k) up-conv
            assert abs(param.std().item() / (2.0 / fan_in) ** 0.5 - 1) < 0.02, name
    assert model.dropout_d3c.p == model.dropout_d4c.p == 0.5
    small = UNet(1, 2, channels=SMALL, padding="same")
    x = torch.randn(1, 1, 32, 32)
    small.eval()
    with torch.no_grad():
        torch.testing.assert_close(small(x), small(x))
    small.train()
    torch.manual_seed(1)
    first = small(x)
    torch.manual_seed(2)
    assert not torch.allclose(first, small(x))


def test_unet_validation():
    model = UNet(1, 2, channels=SMALL)
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(1, 188, 188))
    with pytest.raises(ValueError, match="channels"):
        model(torch.randn(1, 3, 188, 188))
    with pytest.raises(ValueError):
        model(torch.randn(1, 1, 190, 188))
    with pytest.raises(ValueError):
        UNet(1, 2, channels=(64, 128, 256, 512))
    with pytest.raises(ValueError):
        UNet(1, 2, padding="reflect")
    with pytest.raises(ValueError):
        UNet(1, 2, dropout=1.0)
    with pytest.raises(ValueError):
        build_model("unet", task="regression")
    with pytest.raises(ValueError):
        build_model("unet", task="segmentation", variant="caffe")
