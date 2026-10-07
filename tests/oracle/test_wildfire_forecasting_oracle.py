"""Kondylatos et al. (2022) LSTM / ConvLSTM checked against the official wildfire_forecasting code.

Reference (pinned in repos.yaml): Orion-AI-Lab/wildfire_forecasting,
``wildfire_forecasting/models/modules/fire_modules.py`` and ``convlstm.py``. That module imports
fastai (``unet``) and torchvision (``resnet18``) without using either; both are replaced by empty
stand-in modules for the import when they are not installed. Lightning, Hydra and torchmetrics are
only imported by the LightningModule wrappers (``greece_fire_models.py``), which are not needed.
The paper configurations come from ``configs/experiment/{lstm_temporal_cls,clstm_spatiotemporal_cls}.yaml``.
"""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

import numpy as np
import pytest
import torch

from oracle_utils import import_from, oracle_repo
from pyhazards.models import build_model
from pyhazards.models.convlstm import ConvLSTM

DYNAMIC_FEATURES = [
    "1 km 16 days NDVI",
    "LST_Day_1km",
    "LST_Night_1km",
    "era5_max_d2m",
    "era5_max_t2m",
    "era5_max_sp",
    "era5_max_tp",
    "sminx",
    "era5_max_wind_speed",
    "era5_min_rh",
]
STATIC_FEATURES = ["dem_mean", "slope_mean", "roads_distance", "waterway_distance", "population_density"]
PAPER_HPARAMS = {
    "lstm": {"hidden_size": 64, "lstm_layers": 1, "dropout": 0.5},
    "convlstm": {"hidden_size": 32, "lstm_layers": 1, "dropout": 0.5},
}
EXPECTED_PARAMS = {"lstm": 29_652, "convlstm": 372_212}


def _hparams(variant: str, **overrides) -> dict:
    return {
        "dynamic_features": DYNAMIC_FEATURES,
        "static_features": STATIC_FEATURES,
        "clc": "vec",
        **PAPER_HPARAMS[variant],
        **overrides,
    }


def _stub(name: str, **attrs) -> ModuleType:
    module = ModuleType(name)
    module.__dict__.update(attrs)
    return module


@pytest.fixture(scope="module")
def reference():
    root = oracle_repo("wildfire_forecasting")
    stubs = {
        "fastai": _stub("fastai"),
        "fastai.vision": _stub("fastai.vision"),
        "fastai.vision.models": _stub("fastai.vision.models", unet=None),
    }
    if importlib.util.find_spec("torchvision") is None:
        stubs.update(
            {
                "torchvision": _stub("torchvision"),
                "torchvision.models": _stub("torchvision.models"),
                "torchvision.models.resnet": _stub("torchvision.models.resnet", resnet18=None),
            }
        )
    patch = pytest.MonkeyPatch()  # restores sys.modules after the import
    for name, module in stubs.items():
        patch.setitem(sys.modules, name, module)
    numpy_errors = np.geterr()  # fire_modules calls np.seterr(divide="ignore", invalid="ignore")
    try:
        fire_modules = import_from(root, "wildfire_forecasting.models.modules.fire_modules")
        convlstm = import_from(root, "wildfire_forecasting.models.modules.convlstm")
    finally:
        np.seterr(**numpy_errors)
        patch.undo()
    return fire_modules, convlstm


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _assert_same_state(reference: torch.nn.Module, port: torch.nn.Module) -> None:
    ref_state, port_state = reference.state_dict(), port.state_dict()
    assert list(ref_state) == list(port_state)
    for key, value in ref_state.items():
        assert torch.equal(value, port_state[key]), key


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def _seeded_pair(reference_cls, variant: str, seed: int = 0, **overrides):
    torch.manual_seed(seed)
    ref = reference_cls(_hparams(variant, **overrides))
    torch.manual_seed(seed)
    port_kwargs = {"hidden_size": overrides.get("hidden_size", PAPER_HPARAMS[variant]["hidden_size"])}
    if "lstm_layers" in overrides:
        port_kwargs["lstm_layers"] = overrides["lstm_layers"]
    port = build_model("wildfire_forecasting", task="classification", variant=variant, **port_kwargs)
    return ref, port


def _compare_forward(ref: torch.nn.Module, port: torch.nn.Module, x: torch.Tensor) -> None:
    ref.eval()
    port.eval()
    with torch.no_grad():
        expected = ref(x)
        _assert_close(port(x), expected)
    assert torch.allclose(expected.exp().sum(dim=1), torch.ones(x.size(0)))  # log-probabilities

    ref.train()
    port.train()
    torch.manual_seed(5)
    expected = ref(x)
    torch.manual_seed(5)
    _assert_close(port(x), expected)


def test_lstm_matches_reference(reference):
    fire_modules, _ = reference
    ref, port = _seeded_pair(fire_modules.SimpleLSTM, "lstm")
    assert _n_params(ref) == _n_params(port) == EXPECTED_PARAMS["lstm"]
    _assert_same_state(ref, port)
    port.load_state_dict(ref.state_dict(), strict=True)

    torch.manual_seed(1)
    _compare_forward(ref, port, torch.randn(8, 10, 25))


def test_convlstm_matches_reference(reference):
    fire_modules, _ = reference
    ref, port = _seeded_pair(fire_modules.SimpleConvLSTM, "convlstm")
    assert _n_params(ref) == _n_params(port) == EXPECTED_PARAMS["convlstm"]
    _assert_same_state(ref, port)
    port.load_state_dict(ref.state_dict(), strict=True)

    torch.manual_seed(1)
    _compare_forward(ref, port, torch.randn(4, 10, 25, 25, 25))


@pytest.mark.parametrize("variant", ["lstm", "convlstm"])
def test_other_hyperparameters_match_reference(reference, variant):
    fire_modules, _ = reference
    reference_cls = fire_modules.SimpleLSTM if variant == "lstm" else fire_modules.SimpleConvLSTM
    # The hidden size of the reference default model configs (configs/model/greecefire_*_model.yaml),
    # and a two-layer stack.
    ref, port = _seeded_pair(reference_cls, variant, seed=3, hidden_size=16, lstm_layers=2)
    _assert_same_state(ref, port)
    torch.manual_seed(4)
    shape = (3, 6, 25) if variant == "lstm" else (3, 6, 25, 25, 25)
    _compare_forward(ref, port, torch.randn(*shape))


def test_reference_convlstm_equals_pyhazards_convlstm(reference):
    _, convlstm = reference
    torch.manual_seed(0)
    ref = convlstm.ConvLSTM(25, 32, (3, 3), 2, True, True, False, dilation=1)
    torch.manual_seed(0)
    port = ConvLSTM(25, 32, kernel_size=(3, 3), num_layers=2)
    _assert_same_state(ref, port)

    torch.manual_seed(1)
    x = torch.randn(2, 5, 25, 25, 25)
    with torch.no_grad():
        ref_outputs, ref_states = ref(x)
        outputs, states = port(x)
    _assert_close(outputs, ref_outputs[-1])
    _assert_close(states[-1][0], ref_states[-1][0])
    _assert_close(states[-1][1], ref_states[-1][1])
