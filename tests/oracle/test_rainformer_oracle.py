"""Rainformer checked against the official code (Zjut-MultimediaPlus/Rainformer, pinned in repos.yaml).

The official repository has no license: it is fetched at test time and never vendored; its KNMI
checkpoint (Google Drive, no licence stated) is used only as test data. The reference needs
einops; the Oracle workflow runs this file with requirements-farseer-rainformer.txt.

The reference hard-codes its stage feature maps for 288x288 inputs (72/36/18/9), so every
comparison runs at 288x288 with downscaling factors (4, 2, 2, 2). CI compares narrow models;
the default 187.7M-parameter model and the official weights are compared with
``PYHAZARDS_ORACLE_LARGE=1`` / the large ``rainformer_knmi_weights`` asset (run locally, recorded
in the model card). Each check requires identical parameter names and seeded initial values and
compares outputs in eval mode and in train mode (BatchNorm), gradients and running statistics.
"""

from __future__ import annotations

import importlib
import os
import sys
from importlib import metadata
from pathlib import Path

import pytest
import torch

from oracle_utils import missing, oracle_large_asset, oracle_repo
from pyhazards.models import build_model
from pyhazards.models.rainformer import Rainformer, rainformer_stage_sizes

REQUIREMENTS = Path(__file__).parent / "requirements-farseer-rainformer.txt"
KNMI_CHECKPOINT = "model1_1_100.pt"
OFFICIAL = dict(
    input_channel=9, hidden_dim=96, downscaling_factors=(4, 2, 2, 2), layers=(2, 2, 2, 2), heads=(3, 6, 12, 24),
    head_dim=32, window_size=9, relative_pos_embedding=True,
)


@pytest.fixture(autouse=True)
def _reference_stack():
    for line in REQUIREMENTS.read_text().splitlines():
        if "==" not in line or line.lstrip().startswith("#"):
            continue
        name, version = line.split("#")[0].strip().split("==")
        try:
            found = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing(f"{name} is not installed; pip install -r tests/oracle/{REQUIREMENTS.name}")
        if found != version:
            missing(f"needs {name}=={version}, found {found}")


def _import_reference(root: Path, module: str):
    """Import a module of a flat reference directory without leaking its generic module names."""
    names = {path.stem for path in root.glob("*.py")}
    saved = {name: sys.modules.pop(name) for name in list(sys.modules) if name.split(".")[0] in names}
    sys.path.insert(0, str(root))
    try:
        return importlib.import_module(module)
    finally:
        sys.path.remove(str(root))
        for name in [name for name in sys.modules if name.split(".")[0] in names]:
            del sys.modules[name]
        sys.modules.update(saved)


@pytest.fixture(scope="module")
def reference():
    return _import_reference(oracle_repo("Rainformer") / "Rainformer", "Rainformer")


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _assert_same_state(reference: torch.nn.Module, port: torch.nn.Module) -> None:
    ref_state, port_state = reference.state_dict(), port.state_dict()
    assert list(ref_state) == list(port_state)
    for key, value in ref_state.items():
        assert torch.equal(value, port_state[key]), key


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def _compare_forward(reference: torch.nn.Module, port: torch.nn.Module, x: torch.Tensor) -> None:
    reference.eval()
    port.eval()
    with torch.no_grad():
        _assert_close(port(x), reference(x))
    reference.train()
    port.train()  # BatchNorm uses batch statistics and updates its running statistics
    expected = reference(x)
    actual = port(x)
    _assert_close(actual, expected)
    expected.square().mean().backward()
    actual.square().mean().backward()
    ref_params = dict(reference.named_parameters())
    for name, param in port.named_parameters():
        ref_grad = ref_params[name].grad
        assert (param.grad is None) == (ref_grad is None), name
        if ref_grad is not None:
            torch.testing.assert_close(param.grad, ref_grad, rtol=1e-4, atol=1e-6, msg=name)
    _assert_same_state(reference, port)  # running statistics after one training step


REDUCED = [
    dict(input_channel=4, hidden_dim=8, layers=(2, 2, 2, 2), heads=(1, 1, 2, 2), head_dim=8, window_size=9, relative_pos_embedding=True),
    dict(input_channel=3, hidden_dim=8, layers=(4, 2, 2, 2), heads=(2, 1, 1, 2), head_dim=4, window_size=3, relative_pos_embedding=False),
    dict(input_channel=2, hidden_dim=16, layers=(2, 2, 4, 2), heads=(1, 2, 2, 2), head_dim=8, window_size=3, relative_pos_embedding=True),
]


@pytest.mark.parametrize("config", REDUCED)
def test_rainformer_reduced_configs_match_reference(reference, config):
    config = {**config, "downscaling_factors": (4, 2, 2, 2)}
    torch.manual_seed(0)
    ref = reference.Net(**config)
    torch.manual_seed(0)
    port = Rainformer(**config)
    _assert_same_state(ref, port)
    assert _n_params(port) == _n_params(ref)

    torch.manual_seed(1)
    x = torch.rand(2, config["input_channel"], 288, 288)
    _compare_forward(ref, port, x)


def test_rainformer_stage_sizes_match_the_hard_coded_reference(reference):
    torch.manual_seed(0)
    ref = reference.Net(**{**REDUCED[0], "downscaling_factors": (4, 2, 2, 2)})
    gate_maps = [
        tuple(getattr(ref, f"stage{i}").layers[0][4].conv_2[1].normalized_shape[1:]) for i in range(1, 9)
    ]
    assert gate_maps == rainformer_stage_sizes(288, (4, 2, 2, 2))
    # The reference cannot run Sim2Real-Fire's 256x256 frames (64/32/16/8 maps, window 9).
    with pytest.raises(Exception):
        ref.eval()(torch.rand(1, 4, 256, 256))


def test_rainformer_builder_time_major_layout(reference):
    config = {**REDUCED[0], "downscaling_factors": (4, 2, 2, 2)}
    torch.manual_seed(0)
    ref = reference.Net(**config).eval()
    model = build_model(
        "rainformer", task="forecasting", in_channels=2, history=2, hidden_dim=8, heads=(1, 1, 2, 2), head_dim=8
    ).eval()
    model.load_state_dict(ref.state_dict(), strict=True)
    x = torch.rand(1, 2, 2, 288, 288)  # (batch, time, channels) -> 4 stacked frames
    with torch.no_grad():
        _assert_close(model(x), ref(x.flatten(1, 2)).reshape(1, 2, 2, 288, 288))


def test_rainformer_default_config_matches_reference(reference):
    if os.environ.get("PYHAZARDS_ORACLE_LARGE") != "1":
        pytest.skip("default-size comparison (187.7M parameters) runs with PYHAZARDS_ORACLE_LARGE=1")
    torch.manual_seed(0)
    ref = reference.Net(**OFFICIAL)
    torch.manual_seed(0)
    port = build_model("rainformer", task="forecasting")
    _assert_same_state(ref, port)
    assert _n_params(port) == 187_661_990

    torch.manual_seed(1)
    _compare_forward(ref, port, torch.rand(2, 9, 288, 288))


def test_rainformer_knmi_weights_match_reference(reference):
    checkpoint = oracle_large_asset("rainformer_knmi_weights") / KNMI_CHECKPOINT
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    ref = reference.Net(**OFFICIAL)
    ref.load_state_dict(state, strict=True)
    port = build_model("rainformer", task="forecasting")
    port.load_state_dict(state, strict=True)
    ref.eval()
    port.eval()
    torch.manual_seed(2)
    x = torch.rand(2, 9, 288, 288) * 4  # non-negative rain-like inputs
    with torch.no_grad():
        expected = ref(x)
        _assert_close(port(x), expected)
        _assert_close(port(x.unsqueeze(2)), expected.unsqueeze(2))  # (batch, time, 1, H, W)
