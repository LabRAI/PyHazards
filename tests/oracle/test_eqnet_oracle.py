"""EQNet's feature extractor and heads checked against the author's code (test oracle only).

EQNet is rebuilt from the paper (Zhu et al., JGR Solid Earth 2022); no code of AI4EPS/EQNet was copied (its
licence is academic / non-commercial, and it has no shift-and-stack model). Two pinned commits serve as
oracles, read with ``load_definitions`` so that only the needed classes run:

- ``EQNet`` at af94a08 (current): ``eqnet/models/resnet1d.py`` ``ResNet(BasicBlock, [2, 2, 2, 2])``, the
  network of the paper's Figure 3a (input ``(batch, 3, time, stations)``, output ``{"out": (batch,
  stations, 128, time / 32)}``). Checks: 996,960 parameters, parameter names and shapes, identical seeded
  initialisation, identical outputs in evaluation and training mode, batch-norm statistics and gradients.
- ``EQNet-2022`` at a9df49a (2022-07-27, the author's first EQNet): ``PhasePicker`` and ``EventDetector``
  instantiated at the paper's widths (Figure 3b ``[64, 32, 16]``, Figure 3c ``[128, 64, 32]``). The
  current repository's heads are different networks (dilated 5-tap convolutions, 3-class softmax), so
  they are not an oracle. Checks: 7,825 and 31,009 parameters, identical seeded convolution weights,
  identical outputs with copied weights in evaluation mode and with batch statistics.

Not verifiable against any code: the shift-and-stack module and the multi-station wiring (tested from the
paper's description in tests/test_eqnet.py). No EQNet weights were ever released.
"""

from __future__ import annotations

import subprocess
import typing

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from oracle_utils import load_definitions, oracle_repo
from pyhazards.models.eqnet import EQNet, EQNetBackbone, EventDetectionHead, PhasePickingHead

EXACT = dict(rtol=0, atol=0)


@pytest.fixture(scope="module")
def resnet():
    root = oracle_repo("EQNet")
    namespace = {"torch": torch, "nn": nn, "F": F, "Tensor": torch.Tensor}
    namespace.update({name: getattr(typing, name) for name in ("Any", "Callable", "List", "Optional", "Type", "Union")})
    names = ["conv3x1", "conv1x1", "BasicBlock", "Bottleneck", "ResNet"]
    defs = load_definitions(root / "eqnet" / "models" / "resnet1d.py", names, namespace)
    return lambda: defs["ResNet"](defs["BasicBlock"], [2, 2, 2, 2])


@pytest.fixture(scope="module")
def heads():
    root = oracle_repo("EQNet-2022")
    path = root / "eqnet" / "models" / "eqnet.py"
    assert "basic version of eqnet" in subprocess.run(
        ["git", "log", "-1", "--format=%s"], cwd=root, check=True, capture_output=True, text=True
    ).stdout
    return load_definitions(path, ["EventDetector", "PhasePicker"], {"torch": torch, "nn": nn, "F": F})


def _count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _randomise_batch_norm(module: nn.Module, generator: torch.Generator) -> None:
    for bn in module.modules():
        if isinstance(bn, nn.BatchNorm1d):
            with torch.no_grad():
                bn.weight.copy_(torch.rand(bn.weight.shape, generator=generator) + 0.5)
                bn.bias.copy_(torch.randn(bn.bias.shape, generator=generator) * 0.1)
                bn.running_mean.copy_(torch.randn(bn.running_mean.shape, generator=generator) * 0.1)
                bn.running_var.copy_(torch.rand(bn.running_var.shape, generator=generator) + 0.5)


def test_backbone_matches_the_official_resnet1d(resnet):
    torch.manual_seed(0)
    reference = resnet()
    torch.manual_seed(0)
    backbone = EQNetBackbone()
    assert _count(reference) == _count(backbone) == 996_960
    ref_state, state = reference.state_dict(), backbone.state_dict()
    assert list(ref_state) == list(state)
    for key in ref_state:
        torch.testing.assert_close(state[key], ref_state[key], **EXACT, msg=key)
    # The full EQNet carries the official names under "backbone.".
    model_state = EQNet().state_dict()
    assert all(f"backbone.{key}" in model_state for key in ref_state)

    _randomise_batch_norm(reference, torch.Generator().manual_seed(1))
    backbone.load_state_dict(reference.state_dict())
    x = torch.randn(2, 3, 3072, 5)  # official layout (batch, channel, time, station)
    ours = x.permute(0, 3, 1, 2).reshape(10, 3, 3072)
    reference.eval(), backbone.eval()
    with torch.no_grad():
        expected = reference(x)["out"]
        assert expected.shape == (2, 5, 128, 96)
        torch.testing.assert_close(backbone(ours).reshape(2, 5, 128, 96), expected, **EXACT)
        # Lengths that are not multiples of 32 (e.g. 60-s STEAD windows).
        odd = torch.randn(1, 3, 6000, 2)
        torch.testing.assert_close(
            backbone(odd.permute(0, 3, 1, 2).reshape(2, 3, 6000)).reshape(1, 2, 128, -1), reference(odd)["out"], **EXACT
        )

    reference.train(), backbone.train()
    expected = reference(x)["out"]
    actual = backbone(ours).reshape(2, 5, 128, 96)
    torch.testing.assert_close(actual, expected, **EXACT)
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(backbone.state_dict()[key], value, **EXACT, msg=key)
    grad = torch.randn_like(expected)
    expected.backward(grad)
    actual.backward(grad)
    for (name, p_ref), p_port in zip(reference.named_parameters(), backbone.parameters()):
        torch.testing.assert_close(p_port.grad, p_ref.grad, rtol=1e-5, atol=1e-6, msg=name)


def _copy_head(reference: nn.Module, head: nn.Module, picker: bool) -> None:
    """Copy the port's weights into the a9df49a head (its ``bn_layers.0`` of the picker is unused)."""
    offset = 1 if picker else 0
    pairs = [
        (reference.conv_layers[0], head.conv1),
        (reference.conv_layers[1], head.conv2),
        (reference.bn_layers[offset], head.bn1),
        (reference.bn_layers[offset + 1], head.bn2),
        (reference.conv_out, head.conv_out),
    ]
    for ref_module, module in pairs:
        ref_module.load_state_dict(module.state_dict())


@pytest.mark.parametrize("picker", [True, False])
def test_heads_match_the_author_code_at_the_paper_widths(heads, picker):
    if picker:
        make_ref, make_head, channels, expected_count = (
            lambda: heads["PhasePicker"](channels=[64, 32, 16]), lambda: PhasePickingHead(64), 64, 7_825
        )
    else:
        make_ref, make_head, channels, expected_count = (
            lambda: heads["EventDetector"](channels=[128, 64, 32]), lambda: EventDetectionHead(128), 128, 31_009
        )
    torch.manual_seed(3)
    reference = make_ref()
    torch.manual_seed(3)
    head = make_head()
    assert _count(head) == expected_count
    unused = 64 * 2 if picker else 0  # PhasePicker(channels=[64, 32, 16]) also builds an unused BatchNorm1d(64)
    assert _count(reference) == expected_count + unused
    # Same creation order of the convolutions -> same seeded (PyTorch default) initial weights.
    torch.testing.assert_close(head.conv1.weight, reference.conv_layers[0].weight, **EXACT)
    torch.testing.assert_close(head.conv2.weight, reference.conv_layers[1].weight, **EXACT)
    torch.testing.assert_close(head.conv_out.weight, reference.conv_out.weight, **EXACT)
    torch.testing.assert_close(head.conv_out.bias, reference.conv_out.bias, **EXACT)

    _randomise_batch_norm(head, torch.Generator().manual_seed(4))
    _copy_head(reference, head, picker)
    features = torch.randn(2, 3, channels, 96)
    flat = features.reshape(6, channels, 96)
    for batch_statistics in (False, True):
        reference.eval(), head.eval()
        if batch_statistics:  # the a9df49a forward has no training branch that returns outputs
            for module in list(reference.modules()) + list(head.modules()):
                if isinstance(module, nn.BatchNorm1d):
                    module.train()
        with torch.no_grad():
            if picker:
                expected, _ = reference({"out": features})  # (batch, stations, 16 * T)
                torch.testing.assert_close(head(flat).reshape(2, 3, -1), expected, **EXACT)
                assert expected.shape[-1] == 16 * 96
            else:
                expected, _ = reference({"out": flat})  # (n, 1, T)
                torch.testing.assert_close(head(flat), expected.squeeze(1), **EXACT)
