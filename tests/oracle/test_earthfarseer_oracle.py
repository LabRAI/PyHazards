"""Earthfarseer checked against the official code (Alexander-wu/EarthFarseer, pinned in repos.yaml).

The official repository has no license: it is fetched at test time and never vendored. Its stack
(timm 0.9.16) conflicts with the timm pins of the other suites, so the Oracle workflow runs this
file with requirements-farseer-rainformer.txt.

Each check builds both models from the same seed, requires identical parameter names and initial
values, and compares outputs in eval and train mode (the reference has no dropout, drop path or
BatchNorm, so both modes must agree too).

The reference hard-codes the FoTF transformer at width 768 and depth 12 (about 78M parameters).
CI compares reduced models in which only those two literals are overridden in the official
``GF_Block`` call; the default 157.5M-parameter model is compared with
``PYHAZARDS_ORACLE_LARGE=1`` (run locally, recorded in the model card).
"""

from __future__ import annotations

import importlib
import os
import sys
from importlib import metadata
from pathlib import Path

import pytest
import torch

from oracle_utils import missing, oracle_repo
from pyhazards.models import build_model
from pyhazards.models.earthfarseer import Earthfarseer

REQUIREMENTS = Path(__file__).parent / "requirements-farseer-rainformer.txt"


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


def _full_size_only() -> None:
    if os.environ.get("PYHAZARDS_ORACLE_LARGE") != "1":
        pytest.skip("default-size comparison (157.5M parameters) runs with PYHAZARDS_ORACLE_LARGE=1")


def _import_reference(root: Path, *modules: str):
    """Import top-level modules of a flat reference repo without leaking its generic names.

    EarthFarseer's files are called ``model``, ``modules``, ``utils``...: any module of that name
    already imported is set aside during the import and restored afterwards.
    """
    names = {path.stem for path in root.glob("*.py")}
    saved = {name: sys.modules.pop(name) for name in list(sys.modules) if name.split(".")[0] in names}
    sys.path.insert(0, str(root))
    try:
        return [importlib.import_module(module) for module in modules]
    finally:
        sys.path.remove(str(root))
        for name in [name for name in sys.modules if name.split(".")[0] in names]:
            del sys.modules[name]
        sys.modules.update(saved)


@pytest.fixture(scope="module")
def reference():
    model_module, fotf_module = _import_reference(oracle_repo("EarthFarseer"), "model", "FoTF_module")
    return model_module, fotf_module


def _reference_model(reference, monkeypatch, shape_in, gf=None, **kwargs):
    """Official ``Earthfarseer_model``; ``gf=(width, depth)`` overrides the hard-coded GF_Block size."""
    model_module, fotf_module = reference
    if gf is not None:
        official_gf_block = fotf_module.GF_Block

        def gf_block(**gf_kwargs):
            assert (gf_kwargs["embed_dim"], gf_kwargs["depth"]) == (768, 12)  # the literals replaced
            return official_gf_block(**{**gf_kwargs, "embed_dim": gf[0], "depth": gf[1]})

        monkeypatch.setattr(fotf_module, "GF_Block", gf_block)
    try:
        return model_module.Earthfarseer_model(shape_in=shape_in, **kwargs)
    finally:
        monkeypatch.undo()


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
    for mode in ("eval", "train"):
        getattr(reference, mode)()
        getattr(port, mode)()
        with torch.no_grad():
            _assert_close(port(x), reference(x))


REDUCED = [
    # shape_in, reference kwargs, (GF width, depth)
    ((4, 2, 32, 32), dict(hid_S=16, hid_T=32, N_S=2, N_T=3), (32, 2)),
    ((3, 1, 32, 32), dict(hid_S=8, hid_T=16, N_S=4, N_T=2, incep_ker=[3, 5], groups=4), (32, 2)),
    ((2, 1, 64, 64), dict(hid_S=8, hid_T=16, N_S=6, N_T=2), (48, 1)),
]


@pytest.mark.parametrize("shape_in, kwargs, gf", REDUCED)
def test_earthfarseer_reduced_configs_match_reference(reference, monkeypatch, shape_in, kwargs, gf):
    torch.manual_seed(0)
    ref = _reference_model(reference, monkeypatch, shape_in, gf=gf, **kwargs)
    torch.manual_seed(0)
    port = Earthfarseer(shape_in=shape_in, gf_embed_dim=gf[0], gf_depth=gf[1], **kwargs)
    _assert_same_state(ref, port)
    assert _n_params(port) == _n_params(ref)
    assert (port.H1, port.W1) == (ref.H1, ref.W1)

    torch.manual_seed(1)
    x = torch.randn(2, *shape_in)
    _compare_forward(ref, port, x)


def test_earthfarseer_builder_and_official_state_dict(reference, monkeypatch):
    shape_in, kwargs, gf = REDUCED[1]
    torch.manual_seed(0)
    ref = _reference_model(reference, monkeypatch, shape_in, gf=gf, **kwargs)
    torch.manual_seed(5)  # a different initialisation, overwritten by the official weights
    port = build_model(
        "earthfarseer", task="forecasting", in_channels=shape_in[1], history=shape_in[0], img_size=shape_in[2],
        gf_embed_dim=gf[0], gf_depth=gf[1], **kwargs,
    )
    port.load_state_dict(ref.state_dict(), strict=True)
    x = torch.randn(1, *shape_in)
    _compare_forward(ref, port, x)

    seg = build_model(
        "earthfarseer", task="segmentation", in_channels=shape_in[1], history=shape_in[0], img_size=shape_in[2],
        gf_embed_dim=gf[0], gf_depth=gf[1], **kwargs,
    )
    result = seg.load_state_dict(ref.state_dict(), strict=False)
    assert result.unexpected_keys == [] and result.missing_keys == ["segmentation_head.weight", "segmentation_head.bias"]
    ref.eval()
    seg.eval()
    with torch.no_grad():
        _assert_close(seg(x), seg.segmentation_head(ref(x)[:, 0]))


def test_earthfarseer_reference_fails_where_the_port_deviates(reference, monkeypatch):
    # H divisible by 3: the reference's token-grid formula gives 13 for a 12x12 latent.
    ref = _reference_model(reference, monkeypatch, (2, 1, 48, 48), gf=(32, 1), hid_S=8, hid_T=16, N_T=2)
    assert ref.H1 == 13
    with pytest.raises(RuntimeError, match="invalid for input of size"):
        ref(torch.randn(1, 2, 1, 48, 48))
    # Odd N_S: int(H / 2 ** (N_S / 2)) is not the latent size either.
    ref = _reference_model(reference, monkeypatch, (2, 1, 64, 64), gf=(32, 1), hid_S=8, hid_T=16, N_S=3, N_T=2)
    assert ref.H1 == 22  # latent 32
    with pytest.raises(RuntimeError, match="invalid for input of size"):
        ref(torch.randn(1, 2, 1, 64, 64))
    # Non-square input: FoTF builds its transformer for H x H.
    ref = _reference_model(reference, monkeypatch, (2, 1, 32, 64), gf=(32, 1), hid_S=8, hid_T=16, N_T=2)
    with pytest.raises(AssertionError):
        ref(torch.randn(1, 2, 1, 32, 64))

    for shape_in, extra in [((2, 1, 48, 48), {}), ((2, 1, 64, 64), {"N_S": 3}), ((2, 1, 32, 64), {})]:
        port = Earthfarseer(shape_in=shape_in, hid_S=8, hid_T=16, N_T=2, gf_embed_dim=32, gf_depth=1, **extra).eval()
        with torch.no_grad():
            assert port(torch.randn(1, *shape_in)).shape == (1, *shape_in)


def test_earthfarseer_default_config_matches_reference(reference, monkeypatch):
    _full_size_only()
    torch.manual_seed(0)
    ref = _reference_model(reference, monkeypatch, (10, 1, 64, 64))
    torch.manual_seed(0)
    port = build_model("earthfarseer", task="forecasting")
    _assert_same_state(ref, port)
    assert _n_params(port) == 157_518_863

    torch.manual_seed(1)
    x = torch.randn(2, 10, 1, 64, 64)
    _compare_forward(ref, port, x)
