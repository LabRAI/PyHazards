"""SAF-Net checked against the official notebook network and checkpoint (xuguangning1218/TI_Prediction, pinned).

The official code has no license: PyHazards' SAF-Net is written from the paper, and the notebook's
``Net`` class is executed here, from the pinned ``SAF-Net.ipynb``, only as a test oracle. The released
checkpoint ``model_saver/SAF_Net.pkl`` is read from the same checkout.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from oracle_utils import oracle_repo
from pyhazards.models import build_model


def _official_net():
    repo = oracle_repo("TI_Prediction")
    notebook = json.loads((repo / "SAF-Net.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    source = next(cell for cell in cells if "class Net(nn.Module)" in cell)
    source = source[: source.index("net = Net()")]
    # Notebook globals the class uses: ahead_times = [0, 1, 2, 3] (t, t-6 h, t-12 h, t-18 h).
    scope = {"torch": torch, "nn": nn, "F": F, "ahead_times": [0, 1, 2, 3]}
    exec(compile(source, "SAF-Net.ipynb", "exec"), scope)
    return scope["Net"], repo


def _inputs(batch: int = 4, seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(batch, 96, generator=generator), torch.rand(batch, 2, 4, 31, 31, 4, generator=generator)


def _assert_same_state(reference: nn.Module, port: nn.Module) -> None:
    ref_state, port_state = reference.state_dict(), port.state_dict()
    assert list(ref_state) == list(port_state)
    for key, value in ref_state.items():
        assert torch.equal(value, port_state[key]), key


def test_parameters_and_seeded_initialisation_match_official():
    Net, _ = _official_net()
    torch.manual_seed(0)
    reference = Net()
    torch.manual_seed(0)
    port = build_model("saf_net", task="regression")
    assert sum(p.numel() for p in reference.parameters()) == sum(p.numel() for p in port.parameters()) == 880_233
    _assert_same_state(reference, port)


def test_outputs_match_official_in_eval_and_train_mode():
    Net, _ = _official_net()
    torch.manual_seed(0)
    reference = Net()
    port = build_model("saf_net", task="regression")
    port.load_state_dict(reference.state_dict(), strict=True)
    wide, deep = _inputs()
    reference.eval()
    port.eval()
    with torch.no_grad():
        torch.testing.assert_close(port(wide, deep), reference(wide, deep), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(port({"wide": wide, "deep": deep}), reference(wide, deep), rtol=1e-5, atol=1e-6)
    # Train mode: BatchNorm uses batch statistics and updates its running statistics in call order.
    reference.train()
    port.train()
    expected = reference(wide, deep)
    actual = port(wide, deep)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    _assert_same_state(reference, port)
    expected.sum().backward()
    actual.sum().backward()
    for (name, ref_param), port_param in zip(reference.named_parameters(), port.parameters()):
        torch.testing.assert_close(port_param.grad, ref_param.grad, rtol=1e-4, atol=1e-6, msg=name)


def test_released_checkpoint_loads_strictly_and_matches_official():
    Net, repo = _official_net()
    state = torch.load(repo / "model_saver" / "SAF_Net.pkl", map_location="cpu", weights_only=True)
    reference = Net()
    reference.load_state_dict(state, strict=True)
    port = build_model("saf_net", task="regression")
    port.load_state_dict(state, strict=True)
    reference.eval()
    port.eval()

    # Wide predictors of real 2015-2018 test cases, scaled as in the notebook (MinMaxScaler fitted on
    # the training and test rows together); the ERA-Interim wind cubes are not in the repository.
    import pandas as pd

    train = pd.read_csv(repo / "data" / "CMA_train_24h.csv", header=None)
    test = pd.read_csv(repo / "data" / "CMA_test_24h.csv", header=None)
    features = pd.concat([train, test]).iloc[:, 5:].to_numpy(np.float64)
    low, high = features.min(axis=0), features.max(axis=0)
    span = np.where(high > low, high - low, 1.0)
    wide = torch.as_tensor((test.iloc[:6, 5:].to_numpy(np.float64) - low) / span, dtype=torch.float32)
    _, deep = _inputs(batch=6, seed=2)
    with torch.no_grad():
        expected = reference(wide, deep)
        actual = port(wide, deep)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert torch.all(actual >= 0)  # final ReLU on the MinMax-scaled intensity


def test_input_validation():
    port = build_model("saf_net", task="regression")
    wide, deep = _inputs(batch=2)
    with pytest.raises(ValueError, match="deep inputs shaped"):
        port(wide, deep[..., :3])
    with pytest.raises(ValueError, match="wide inputs shaped"):
        port(wide[:, :90], deep)
