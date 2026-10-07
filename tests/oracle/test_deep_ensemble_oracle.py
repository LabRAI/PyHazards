"""deep_ensemble checked against the deep ensemble of Kondylatos et al. (2025).

Reference (pinned in repos.yaml): Orion-AI-Lab/uncertainty-wildfires (MIT), the code of
"Uncertainty-Aware Deep Learning for Wildfire Danger Forecasting" (arXiv 2509.25017):

- ``models/model.py`` ``SimpleLSTM``, the member (``configs/config_det.json``: ``output_lstm`` 128,
  dropout 0.5, 25 features). Only the class is executed (``load_definitions``): the module also
  imports the vendored Bayesian-layer library for its other models.
- ``utils/train_functions.py`` ``uncertainties`` (the ensemble aggregation used by ``test.py``) and
  ``enable_dropout``; executed the same way, because ``utils/__init__`` imports matplotlib.
- ``trained_models/ensembles/no-noisy/ensembles-<seed>-checkpoint.pth``: the paper's ten ensemble
  members (``configs_test/config_des.json``, ``num_models`` 10), each trained with
  ``torch.manual_seed(seed)`` before it was built. The checkpoints also pickle the repository's
  ``ConfigParser``; it is unpickled into an inert stand-in.

PyHazards builds the members as ``wildfire_forecasting`` LSTMs with ``hidden_size=128``: the same
layers and parameter names (that model returns ``log_softmax`` where the reference returns logits;
the ensemble applies a softmax to both).
"""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from oracle_utils import load_definitions, oracle_repo
from pyhazards.models import DeepEnsemble, build_model

SEEDS = (1, 12, 17, 123, 1234, 2222, 12345, 54321, 77777, 87877)
MEMBER_PARAMS = 104_308  # SimpleLSTM(output_lstm=128, len_features=25)
LAG = 45  # days per sample in the paper's next-day setting (dataset "lag": 45)


class _Inert:
    """Stand-in for non-torch classes pickled in the checkpoints; keeps their state, runs no code."""

    def __setstate__(self, state):
        self.state = state


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in {"torch", "collections", "pathlib", "numpy"}:
            return super().find_class(module, name)
        return _Inert


_PICKLE = SimpleNamespace(Unpickler=_CheckpointUnpickler, load=pickle.load, __name__="pickle")


@pytest.fixture(scope="module")
def reference():
    root = oracle_repo("uncertainty-wildfires")
    namespace = {"torch": torch, "nn": nn}
    lstm = load_definitions(root / "models" / "model.py", ["SimpleLSTM"], namespace)["SimpleLSTM"]
    functions = load_definitions(root / "utils" / "train_functions.py", ["uncertainties", "enable_dropout"], namespace)
    checkpoints = {}
    for seed in SEEDS:
        path = root / "trained_models" / "ensembles" / "no-noisy" / f"ensembles-{seed}-checkpoint.pth"
        checkpoints[seed] = torch.load(path, map_location="cpu", weights_only=False, pickle_module=_PICKLE)
    return SimpleNamespace(SimpleLSTM=lstm, checkpoints=checkpoints, **functions)


def _reference_member(reference, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return reference.SimpleLSTM(output_lstm=128, dropout=0.5, len_features=25, noisy=False)


def _ensemble(**kwargs) -> DeepEnsemble:
    return build_model(
        "deep_ensemble",
        task="classification",
        base_model="wildfire_forecasting",
        base_kwargs={"hidden_size": 128},
        num_members=len(SEEDS),
        seeds=SEEDS,
        **kwargs,
    )


def _official_pair(reference, **kwargs):
    references, ensemble = [], _ensemble(**kwargs)
    for seed, member in zip(SEEDS, ensemble.members):
        state = reference.checkpoints[seed]["state_dict"]
        ref = _reference_member(reference)
        ref.load_state_dict(state, strict=True)
        member.load_state_dict(state, strict=True)
        references.append(ref.eval())
    return references, ensemble.eval()


def _assert_close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_checkpoints_are_the_paper_ensemble(reference):
    for seed in SEEDS:
        config = reference.checkpoints[seed]["config"].state["_config"]
        assert config["seed"] == seed
        assert dict(config["arch"]["args"]) == {"output_lstm": 128, "dropout": 0.5}
        assert config["noisy"] is False
        assert len(config["features"]["dynamic"]) + len(config["features"]["static"]) == 25
        assert reference.checkpoints[seed]["epoch"] == 30


def test_members_match_reference_initialisation(reference):
    ensemble = _ensemble()
    assert ensemble.seeds == list(SEEDS)
    for seed, member in zip(SEEDS, ensemble.members):
        ref = _reference_member(reference, seed)
        assert sum(p.numel() for p in ref.parameters()) == sum(p.numel() for p in member.parameters()) == MEMBER_PARAMS
        ref_state, state = ref.state_dict(), member.state_dict()
        assert set(ref_state) == set(state)  # same names; the reference creates the LSTM before the LayerNorm
        for key, value in ref_state.items():
            assert torch.equal(value, state[key]), (seed, key)
    assert sum(p.numel() for p in ensemble.parameters()) == len(SEEDS) * MEMBER_PARAMS


def test_official_members_load_and_match(reference):
    references, ensemble = _official_pair(reference)
    torch.manual_seed(0)
    x = torch.randn(16, LAG, 25)
    with torch.no_grad():
        for ref, member in zip(references, ensemble.members):
            _assert_close(member(x), torch.log_softmax(ref(x), dim=1))


def test_ensemble_aggregation_matches_reference(reference):
    references, ensemble = _official_pair(reference)
    torch.manual_seed(1)
    x = torch.randn(32, LAG, 25)
    with torch.no_grad():
        outputs, mean, epistemic, aleatoric, mi, entropy = reference.uncertainties([ref(x) for ref in references], 0.000001)
        terms = ensemble.predict_uncertainty(x)
        log_probs = ensemble(x)
    _assert_close(terms["member_probs"], outputs.transpose(0, 1))
    _assert_close(terms["probs"], mean)
    _assert_close(terms["epistemic"], epistemic)
    _assert_close(terms["aleatoric"], aleatoric)
    _assert_close(terms["entropy"], entropy)
    _assert_close(terms["mutual_information"], mi)
    _assert_close(log_probs.exp(), mean)
    assert epistemic.max() > 1e-4  # the trained members disagree


def test_mc_dropout_passes_match_reference_loop(reference):
    """test.py with ``dropout: true`` and ``forward_passes: 3``: dropout on at test time, passes pooled."""
    references, ensemble = _official_pair(reference, mc_dropout_passes=3)
    x = torch.randn(8, LAG, 25)
    for ref in references:
        reference.enable_dropout(ref)
    with torch.no_grad():
        torch.manual_seed(7)
        outputs_list = [ref(x) for ref in references for _ in range(3)]
        _, mean, epistemic, aleatoric, mi, entropy = reference.uncertainties(outputs_list, 0.000001)
        torch.manual_seed(7)
        terms = ensemble.predict_uncertainty(x)
    assert terms["member_probs"].shape == (3 * len(SEEDS), 8, 2)
    assert not torch.equal(terms["member_probs"][0], terms["member_probs"][1])  # dropout was active
    _assert_close(terms["probs"], mean)
    _assert_close(terms["epistemic"], epistemic)
    _assert_close(terms["aleatoric"], aleatoric)
    _assert_close(terms["entropy"], entropy)
    _assert_close(terms["mutual_information"], mi)
    assert not any(m.training for m in ensemble.modules())  # dropout switched back off
