"""DeepLabV3 checked against segmentation_models_pytorch 0.3.2 (and its torchvision lineage).

Shadrin et al. (Sci. Rep. 14:2606, 2024) used DeepLabV3 with "encoder backbone ResNet18, number
of stages used in encoder equals to 3 ... default values for the other hyperparameters", i.e.
``smp.DeepLabV3(encoder_name="resnet18", encoder_depth=3)``. Each check builds smp and the port
from the same seed, requires identical parameter names, initial values and convolution geometry
(stride, dilation, padding, which a state dict does not carry), and compares outputs in eval and
train mode.
"""

from __future__ import annotations

import pytest
import torch

from oracle_utils import missing, oracle_package
from pyhazards.models import build_model
from pyhazards.models.deeplabv3 import DeepLabV3

SHADRIN_PARAMETERS = 13_220_609  # 58 channels (3-day horizon); 2,727,169 of them used in forward


def _smp():
    return oracle_package("segmentation_models_pytorch", "0.3.2")


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _geometry(model: torch.nn.Module) -> list:
    return [(m.stride, m.dilation, m.padding) for m in model.modules() if isinstance(m, torch.nn.Conv2d)]


def _assert_same_model(reference: torch.nn.Module, port: torch.nn.Module) -> None:
    ref_state, port_state = reference.state_dict(), port.state_dict()
    assert list(ref_state) == list(port_state)
    for key, value in ref_state.items():
        assert torch.equal(value, port_state[key]), key
    assert _geometry(reference) == _geometry(port)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def _compare_forward(reference: torch.nn.Module, port: torch.nn.Module, x: torch.Tensor) -> None:
    reference.eval()
    port.eval()
    with torch.no_grad():
        _assert_close(port(x), reference(x))
    reference.train()
    port.train()
    torch.manual_seed(7)  # ASPP projection dropout
    expected = reference(x)
    torch.manual_seed(7)
    _assert_close(port(x), expected)


def test_deeplabv3_shadrin_config_matches_smp():
    smp = _smp()
    torch.manual_seed(0)
    reference = smp.DeepLabV3(encoder_name="resnet18", encoder_depth=3, encoder_weights=None, in_channels=58, classes=1)
    torch.manual_seed(0)
    port = build_model("deeplabv3", task="segmentation")
    _assert_same_model(reference, port)
    assert _n_params(port) == SHADRIN_PARAMETERS
    # At encoder_depth=3 the forward pass stops after layer2; layer3/layer4 stay in the model.
    unused = sum(p.numel() for n, p in port.named_parameters() if n.startswith(("encoder.layer3", "encoder.layer4")))
    assert SHADRIN_PARAMETERS - unused == 2_727_169
    assert port.encoder.output_stride == reference.encoder.output_stride == 8

    torch.manual_seed(1)
    x = torch.randn(2, 58, 32, 32)  # 32 x 32 tiles (21 x 21 km at 650 m) as in the paper
    _compare_forward(reference, port, x)


@pytest.mark.parametrize(
    "encoder_name, encoder_depth, in_channels, size",
    [("resnet34", 5, 3, 64), ("resnet18", 4, 7, 40), ("resnet18", 5, 1, 48)],
)
def test_deeplabv3_other_configs_match_smp(encoder_name, encoder_depth, in_channels, size):
    smp = _smp()
    torch.manual_seed(0)
    reference = smp.DeepLabV3(
        encoder_name=encoder_name, encoder_depth=encoder_depth, encoder_weights=None, in_channels=in_channels, classes=2
    )
    torch.manual_seed(0)
    port = DeepLabV3(in_channels=in_channels, classes=2, encoder_name=encoder_name, encoder_depth=encoder_depth)
    _assert_same_model(reference, port)
    torch.manual_seed(1)
    _compare_forward(reference, port, torch.randn(2, in_channels, size, size))


def test_deeplabv3_smp_defaults_match():
    smp = _smp()
    torch.manual_seed(0)
    reference = smp.DeepLabV3(encoder_weights=None)
    torch.manual_seed(0)
    port = DeepLabV3()
    _assert_same_model(reference, port)
    assert _n_params(port) == 26_007_105


def test_deeplabv3_imagenet_stem_matches_smp():
    smp = _smp()
    reference = smp.DeepLabV3(encoder_name="resnet18", encoder_depth=3, encoder_weights="imagenet", in_channels=58)
    port = build_model("deeplabv3", task="segmentation", encoder_weights="imagenet")
    ref_encoder, port_encoder = reference.encoder.state_dict(), port.encoder.state_dict()
    assert list(ref_encoder) == list(port_encoder)
    for key, value in ref_encoder.items():
        assert torch.equal(value, port_encoder[key]), key
    port.load_state_dict(reference.state_dict(), strict=True)
    torch.manual_seed(1)
    _compare_forward(reference, port, torch.randn(2, 58, 32, 32))


def test_deeplabv3_decoder_is_torchvision_deeplab_head():
    # smp's DeepLabV3 decoder file is taken from torchvision; its decoder plus the 1x1 head
    # convolution is torchvision's DeepLabHead (ASPP rates 12/24/36, 256 channels).
    try:
        from torchvision.models.segmentation.deeplabv3 import DeepLabHead
    except ImportError:
        missing("torchvision is not installed (the default oracle suite installs it next to torch)")

    torch.manual_seed(0)
    head = DeepLabHead(128, 1)
    port = build_model("deeplabv3", task="segmentation", in_channels=4)
    state = {f"decoder.{key}": value for key, value in head.state_dict().items() if not key.startswith("4.")}
    state["segmentation_head.0.weight"] = head.state_dict()["4.weight"]
    state["segmentation_head.0.bias"] = head.state_dict()["4.bias"]
    result = port.load_state_dict(state, strict=False)
    assert not result.unexpected_keys and all(key.startswith("encoder.") for key in result.missing_keys)
    features = torch.randn(2, 128, 6, 6)
    head.eval()
    port.eval()
    with torch.no_grad():
        _assert_close(port.segmentation_head[0](port.decoder(features)), head(features))
