"""deep_ensemble: member seeding, mixture outputs per task, uncertainty terms, training helpers.

The comparison with the reference deep ensemble of Kondylatos et al. (2025), including its ten
released members, lives in tests/oracle/test_deep_ensemble_oracle.py. The Gaussian regression path and
the FGSM helpers follow Lakshminarayanan et al. (2017), which released no code; they are checked here
against the paper's formulas.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhazards.datasets import load_dataset
from pyhazards.engine import Trainer
from pyhazards.models import DeepEnsemble, build_model
from pyhazards.models.deep_ensemble import (
    classification_uncertainties,
    fgsm_example,
    gaussian_nll_loss,
    input_range_epsilon,
    split_gaussian_output,
)

EPS = 1e-6


def _lstm_ensemble(**kwargs) -> DeepEnsemble:
    return build_model("deep_ensemble", task="classification", base_model="wildfire_forecasting", **kwargs)


def _mlp_ensemble(task: str, out_dim: int, **kwargs) -> DeepEnsemble:
    return build_model("deep_ensemble", task=task, base_model="mlp", base_kwargs={"in_dim": 3, "out_dim": out_dim, "hidden_dim": 16}, **kwargs)


def _close(actual, expected, **kwargs):
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6, **kwargs)


def test_members_equal_single_models_built_with_their_seed():
    ensemble = _lstm_ensemble(num_members=3, seeds=[5, 6, 7])
    assert ensemble.num_members == 3 and ensemble.seeds == [5, 6, 7]
    for seed, member in zip(ensemble.seeds, ensemble.members):
        torch.manual_seed(seed)
        single = build_model("wildfire_forecasting", task="classification")
        for (key, value), (other_key, other) in zip(member.state_dict().items(), single.state_dict().items()):
            assert key == other_key and torch.equal(value, other)
    assert not torch.equal(ensemble.members[0].fc1.weight, ensemble.members[1].fc1.weight)
    default = _lstm_ensemble()
    assert default.num_members == 5 and default.seeds == [0, 1, 2, 3, 4]


def test_builder_leaves_the_global_random_state_alone():
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    _lstm_ensemble(num_members=2)
    torch.testing.assert_close(torch.rand(3), expected)


def test_classification_mixture_and_uncertainty_terms():
    ensemble = _lstm_ensemble(num_members=4).eval()
    x = torch.randn(5, 10, 25)
    with torch.no_grad():
        member_probs = torch.stack([member(x).exp() for member in ensemble.members])
        log_probs = ensemble(x)
        terms = ensemble.predict_uncertainty(x)
    mean = member_probs.mean(dim=0)
    assert log_probs.shape == (5, 2)
    _close(log_probs.exp(), mean)
    _close(terms["probs"], mean)
    _close(terms["member_probs"], member_probs)
    _close(terms["epistemic"], ((member_probs - mean) ** 2).mean(dim=0))
    _close(terms["aleatoric"], (member_probs * (1 - member_probs)).mean(dim=0))
    entropy = -(mean * torch.log(mean + EPS)).sum(dim=1)
    member_entropy = -(member_probs * torch.log(member_probs + EPS)).sum(dim=2).mean(dim=0)
    _close(terms["entropy"], entropy)
    _close(terms["mutual_information"], entropy - member_entropy)
    assert terms["entropy"].shape == terms["mutual_information"].shape == (5,)
    assert (terms["mutual_information"] > -1e-5).all()


def test_one_member_ensemble_is_the_member():
    ensemble = _lstm_ensemble(num_members=1, seeds=[3]).eval()
    torch.manual_seed(3)
    single = build_model("wildfire_forecasting", task="classification").eval()
    x = torch.randn(4, 10, 25)
    with torch.no_grad():
        _close(ensemble(x), single(x))
        terms = ensemble.predict_uncertainty(x)
    assert torch.count_nonzero(terms["epistemic"]) == 0
    _close(terms["mutual_information"], torch.zeros(4))


def test_binary_segmentation_members_use_a_sigmoid():
    ensemble = build_model(
        "deep_ensemble", task="segmentation", base_model="logistic_regression", base_kwargs={"in_channels": 4}, num_members=3
    )
    x = torch.randn(2, 4, 8, 8)
    with torch.no_grad():
        member_probs = torch.stack([torch.sigmoid(member(x)) for member in ensemble.members])
        logits = ensemble(x)
        terms = ensemble.predict_uncertainty(x)
    mean = member_probs.mean(dim=0)
    assert logits.shape == (2, 1, 8, 8)
    _close(torch.sigmoid(logits), mean)
    _close(terms["probs"], mean)
    _close(terms["epistemic"], member_probs.var(dim=0, unbiased=False))
    _close(terms["aleatoric"], (member_probs * (1 - member_probs)).mean(dim=0))
    entropy = -(mean * torch.log(mean + EPS) + (1 - mean) * torch.log(1 - mean + EPS)).squeeze(1)
    _close(terms["entropy"], entropy)
    assert terms["mutual_information"].shape == (2, 8, 8)


def test_member_arguments_are_passed_through():
    ensemble = build_model("deep_ensemble", task="segmentation", base_model="utae", base_kwargs={"in_channels": 3}, num_members=2).eval()
    x = torch.randn(1, 3, 3, 32, 32)
    positions = torch.tensor([[10.0, 11.0, 12.0]])
    with torch.no_grad():
        out = ensemble(x, batch_positions=positions)
        expected = torch.stack([torch.sigmoid(m(x, batch_positions=positions)) for m in ensemble.members]).mean(dim=0)
    assert out.shape == (1, 1, 32, 32)
    _close(torch.sigmoid(out), expected)


def test_gaussian_regression_is_the_moment_matched_mixture():
    ensemble = _mlp_ensemble("regression", out_dim=4, num_members=3)
    x = torch.randn(6, 3)
    with torch.no_grad():
        raw = torch.stack([member(x) for member in ensemble.members])
        mean, variance = ensemble(x)
        terms = ensemble.predict_uncertainty(x)
    mu, var = raw[:, :, :2], F.softplus(raw[:, :, 2:]) + 1e-6
    _close(mean, mu.mean(dim=0))
    _close(variance, (var + mu**2).mean(dim=0) - mu.mean(dim=0) ** 2)  # the paper's formula
    _close(terms["aleatoric"], var.mean(dim=0))
    _close(terms["epistemic"], mu.var(dim=0, unbiased=False))
    _close(terms["member_variances"], var)
    assert mean.shape == variance.shape == (6, 2)
    odd = _mlp_ensemble("regression", out_dim=3, num_members=2)
    with pytest.raises(ValueError, match="even number"):
        odd(x)


def test_point_regression_uses_the_member_spread():
    ensemble = _mlp_ensemble("regression", out_dim=1, num_members=3, regression_output="point")
    x = torch.randn(6, 3)
    with torch.no_grad():
        raw = torch.stack([member(x) for member in ensemble.members])
        mean, variance = ensemble(x)
        terms = ensemble.predict_uncertainty(x)
    _close(mean, raw.mean(dim=0))
    _close(variance, raw.var(dim=0, unbiased=False))
    assert "aleatoric" not in terms


def test_gaussian_nll_is_equation_1():
    output, target = torch.randn(8, 4), torch.randn(8, 2)
    mean, variance = split_gaussian_output(output)
    expected = (0.5 * torch.log(variance) + (target - mean) ** 2 / (2 * variance)).mean()
    _close(gaussian_nll_loss(output, target), expected)
    _close(gaussian_nll_loss(output, target), nn.GaussianNLLLoss()(mean, target, variance))
    with pytest.raises(ValueError, match="target shape"):
        gaussian_nll_loss(output, torch.randn(8, 3))


def test_input_range_epsilon_is_one_percent_of_each_dimension():
    x = torch.stack([torch.linspace(0, 255, 10), torch.linspace(-1, 1, 10)], dim=1)
    x[3, 1] = float("nan")
    epsilon = input_range_epsilon(x)
    assert epsilon.shape == (1, 2)
    _close(epsilon, torch.tensor([[2.55, 0.02]]))  # the paper's example: 2.55 for inputs in [0, 255]
    sequence = torch.randn(5, 4, 3)
    per_feature = input_range_epsilon(sequence, dims=(0, 1))
    assert per_feature.shape == (1, 1, 3)
    _close(per_feature.flatten(), 0.01 * (sequence.amax(dim=(0, 1)) - sequence.amin(dim=(0, 1))))


def test_fgsm_example_steps_along_the_gradient_sign():
    model = nn.Linear(3, 2)
    x, y = torch.randn(4, 3), torch.tensor([0, 1, 1, 0])
    loss_fn = nn.CrossEntropyLoss()
    epsilon = torch.tensor([0.1, 0.2, 0.3])
    adversarial = fgsm_example(model, x, y, loss_fn, epsilon)
    assert model.weight.grad is None and not adversarial.requires_grad  # only the input gradient
    probe = x.clone().requires_grad_(True)
    loss_fn(model(probe), y).backward()
    _close(adversarial, x + epsilon * probe.grad.sign())


def _toy_classification():
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(60, 3, generator=generator)
    return x, (x[:, 0] > 0).long()


def _fit(ensemble, x, y, **kwargs):
    return ensemble.fit(
        x,
        y,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.3),
        epochs=15,
        batch_size=10,
        **kwargs,
    )


def test_fit_trains_each_member_independently():
    x, y = _toy_classification()
    ensemble = _mlp_ensemble("classification", out_dim=2, num_members=2, seeds=[4, 9])
    first_member = copy.deepcopy(ensemble.members[0])
    _fit(ensemble, x, y)
    with torch.no_grad():
        accuracy = (ensemble(x).argmax(dim=1) == y).float().mean()
    assert accuracy > 0.9
    # Member 0 trained alone, with its own seed, ends up the same: members do not interact.
    alone = _fit(DeepEnsemble([first_member], task="classification", seeds=[4]), x, y)
    for value, other in zip(ensemble.members[0].state_dict().values(), alone.members[0].state_dict().values()):
        assert torch.equal(value, other)


def test_fit_with_adversarial_examples():
    x, y = _toy_classification()
    plain = _fit(_mlp_ensemble("classification", out_dim=2, num_members=2), x, y)
    adversarial = _fit(_mlp_ensemble("classification", out_dim=2, num_members=2), x, y, adversarial_epsilon=input_range_epsilon(x))
    plain_state, adversarial_state = plain.members[0].state_dict(), adversarial.members[0].state_dict()
    assert any(not torch.equal(plain_state[key], adversarial_state[key]) for key in plain_state)
    with torch.no_grad():
        assert (adversarial(x).argmax(dim=1) == y).float().mean() > 0.9
    with pytest.raises(ValueError, match="length"):
        _fit(_mlp_ensemble("classification", out_dim=2, num_members=1), x, y[:-1])


def test_mc_dropout_passes_pool_stochastic_samples():
    ensemble = _lstm_ensemble(num_members=2, mc_dropout_passes=4).eval()
    x = torch.randn(3, 10, 25)
    with torch.no_grad():
        terms = ensemble.predict_uncertainty(x)
    assert terms["member_probs"].shape == (8, 3, 2)
    assert not torch.equal(terms["member_probs"][0], terms["member_probs"][1])
    assert not any(module.training for module in ensemble.modules())
    deterministic = _lstm_ensemble(num_members=2).eval()
    with torch.no_grad():
        torch.testing.assert_close(deterministic(x), deterministic(x), rtol=0, atol=0)


def test_trainer_refuses_fit_and_evaluates_the_ensemble():
    bundle = load_dataset("wildfire_danger_synthetic", micro=True).load()
    ensemble = _lstm_ensemble(num_members=2)
    with pytest.raises(TypeError, match="independently"):
        Trainer(ensemble, device="cpu").fit(
            bundle, optimizer=torch.optim.Adam(ensemble.parameters()), loss_fn=nn.NLLLoss()
        )
    assert Trainer(ensemble, device="cpu").evaluate(bundle, split="test")


def test_invalid_arguments():
    with pytest.raises(ValueError, match="base_model"):
        build_model("deep_ensemble", task="classification")
    with pytest.raises(ValueError, match="itself"):
        build_model("deep_ensemble", task="classification", base_model="deep_ensemble")
    with pytest.raises(KeyError):
        build_model("deep_ensemble", task="classification", base_model="no_such_model")
    with pytest.raises(ValueError, match="distinct"):
        _lstm_ensemble(num_members=2, seeds=[1, 1])
    with pytest.raises(ValueError, match="distinct"):
        _lstm_ensemble(num_members=3, seeds=[1, 2])
    with pytest.raises(ValueError, match="num_members"):
        _lstm_ensemble(num_members=0)
    with pytest.raises(TypeError, match="base_kwargs"):
        _lstm_ensemble(hidden_size=128)
    with pytest.raises(ValueError, match="regression_output"):
        _lstm_ensemble(regression_output="quantile")
    with pytest.raises(ValueError, match="mc_dropout_passes"):
        _lstm_ensemble(mc_dropout_passes=-1)
    with pytest.raises(ValueError, match="task"):
        DeepEnsemble([nn.Linear(2, 2)], task="detection")


def test_members_must_return_one_tensor_with_channels():
    class Pair(nn.Module):
        def forward(self, x):
            return x, x

    with pytest.raises(ValueError, match="one tensor"):
        DeepEnsemble([Pair()])(torch.randn(2, 3))
    with pytest.raises(ValueError, match="shape"):
        DeepEnsemble([nn.Identity()])(torch.randn(4))
    with pytest.raises(ValueError, match="shape"):
        classification_uncertainties(torch.rand(3, 4))
