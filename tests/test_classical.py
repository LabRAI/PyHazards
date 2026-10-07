"""random_forest / xgboost: instance features, the estimator-module contract, Trainer guard, persistence.

Equality with the official Kondylatos et al. (2022) random-forest notebook (and with xgboost's
XGBClassifier defaults) lives in tests/oracle/test_classical_oracle.py. Tests that need xgboost skip
when it is not installed (it is an optional extra).
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import NotFittedError

from pyhazards.datasets import load_dataset
from pyhazards.engine import Trainer
from pyhazards.models import EstimatorModule, build_model, kondylatos_instance_features
from pyhazards.models.classical import KONDYLATOS_DAILY_LAYOUT, KONDYLATOS_RF_PARAMS

needs_xgboost = pytest.mark.skipif(importlib.util.find_spec("xgboost") is None, reason="xgboost is not installed")


@pytest.fixture(scope="module")
def danger():
    return load_dataset("wildfire_danger_synthetic", samples=150).load()


@pytest.fixture(scope="module")
def fitted_forest(danger):
    train = danger.get_split("train")
    return build_model("random_forest", task="classification").fit(train.inputs, train.targets)


def test_instance_features_follow_the_notebook_order():
    daily = torch.arange(2 * 3 * 25, dtype=torch.float32).reshape(2, 3, 25)
    daily[0, 0, 1] = float("nan")
    features = kondylatos_instance_features(daily)
    values = daily.double().numpy()
    assert features.shape == (2, 35) and features.dtype == np.float64
    np.testing.assert_array_equal(features[1, :10], values[1, :, :10].mean(axis=0))  # ten-day means
    assert features[0, 1] == values[0, 1:, 1].mean()  # the missing day is skipped (np.nanmean)
    np.testing.assert_array_equal(features[:, 10:], values[:, -1, :])  # t-1 dynamic, static, land cover
    assert kondylatos_instance_features(torch.randn(4, 7, 8), n_dynamic=3, n_static=2, n_land_cover=3).shape == (4, 11)


def test_instance_features_validate_shape():
    with pytest.raises(ValueError, match="shape"):
        kondylatos_instance_features(torch.randn(2, 10, 24))
    with pytest.raises(ValueError, match="shape"):
        kondylatos_instance_features(torch.randn(2, 25))
    with pytest.raises(ValueError, match="daily_layout"):
        kondylatos_instance_features(torch.randn(2, 10, 25), n_dynamic=0)


def test_random_forest_defaults_are_the_notebook_configuration():
    model = build_model("random_forest", task="classification")
    assert isinstance(model, EstimatorModule) and isinstance(model.estimator, RandomForestClassifier)
    params = model.estimator.get_params()
    assert {key: params[key] for key in KONDYLATOS_RF_PARAMS} == KONDYLATOS_RF_PARAMS
    assert model.daily_layout == KONDYLATOS_DAILY_LAYOUT
    assert list(model.parameters()) == [] and model.state_dict() == {}
    assert not model.is_fitted


def test_unfitted_model_raises():
    model = build_model("random_forest", task="classification")
    for call in (model, model.predict_proba, model.predict):
        with pytest.raises(NotFittedError, match="fit"):
            call(torch.randn(2, 10, 25))


def test_fit_and_forward_on_daily_tensors(danger, fitted_forest):
    test = danger.get_split("test")
    assert fitted_forest.is_fitted
    log_probs = fitted_forest(test.inputs)
    assert log_probs.shape == (len(test.targets), 2)
    assert log_probs.dtype == torch.float32 and not log_probs.requires_grad
    probabilities = fitted_forest.predict_proba(test.inputs)
    torch.testing.assert_close(log_probs.exp(), torch.from_numpy(probabilities).float())
    np.testing.assert_array_equal(fitted_forest.predict(test.inputs), fitted_forest.estimator.classes_[probabilities.argmax(1)])
    assert fitted_forest(test.inputs.double()).dtype == torch.float64
    assert fitted_forest(test.inputs.numpy()).dtype == torch.get_default_dtype()
    accuracy = (log_probs.argmax(dim=1) == test.targets).float().mean()
    assert accuracy > 0.6  # the synthetic danger data carry a learnable signal


def test_two_dimensional_features_are_accepted(danger, fitted_forest):
    test = danger.get_split("test")
    features = kondylatos_instance_features(test.inputs)
    np.testing.assert_array_equal(fitted_forest.predict_proba(features), fitted_forest.predict_proba(test.inputs))
    with pytest.raises(ValueError, match="35 features"):
        fitted_forest(torch.randn(3, 30))
    with pytest.raises(ValueError, match="shape"):
        fitted_forest(torch.randn(3, 10, 25, 1))
    no_layout = build_model("random_forest", task="classification", daily_layout=None)
    with pytest.raises(ValueError, match="shape"):
        no_layout.fit(torch.randn(6, 10, 25), torch.tensor([0, 1] * 3))


def test_target_validation():
    model = build_model("random_forest", task="classification")
    with pytest.raises(ValueError, match="shape"):
        model.fit(torch.randn(6, 10, 25), torch.zeros(6, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="rows"):
        model.fit(torch.randn(6, 10, 25), torch.zeros(5, dtype=torch.long))
    model.fit(torch.randn(6, 10, 25), torch.tensor([[0], [1]] * 3))  # (n, 1) labels are flattened
    assert model(torch.randn(2, 10, 25)).shape == (2, 2)


def test_regression_forest():
    model = build_model("random_forest", task="regression")
    assert isinstance(model.estimator, RandomForestRegressor)
    assert model.estimator.max_depth == 10
    x = torch.randn(30, 10, 25)
    model.fit(x, torch.randn(30))
    assert model(x[:4]).shape == (4, 1)
    model.fit(x, torch.randn(30, 2))
    assert model(x[:4]).shape == (4, 2)
    with pytest.raises(ValueError, match="predict_proba"):
        model.predict_proba(x[:4])


def test_builder_arguments():
    with pytest.raises(ValueError, match="task"):
        build_model("random_forest", task="segmentation")
    model = build_model("random_forest", task="classification", n_estimators=7, n_jobs=2, class_weight="balanced")
    assert (model.estimator.n_estimators, model.estimator.n_jobs, model.estimator.class_weight) == (7, 2, "balanced")
    with pytest.raises(TypeError):
        build_model("random_forest", task="classification", not_an_option=1)


def test_save_and_load_round_trip(tmp_path, danger, fitted_forest):
    path = tmp_path / "forest.joblib"
    fitted_forest.save(path)
    loaded = EstimatorModule.load(path)
    assert (loaded.name, loaded.task, loaded.daily_layout) == ("random_forest", "classification", KONDYLATOS_DAILY_LAYOUT)
    test = danger.get_split("test")
    np.testing.assert_array_equal(loaded.predict_proba(test.inputs), fitted_forest.predict_proba(test.inputs))

    import joblib

    joblib.dump({"something": "else"}, tmp_path / "other.joblib")
    with pytest.raises(ValueError, match="EstimatorModule.save"):
        EstimatorModule.load(tmp_path / "other.joblib")


def test_trainer_refuses_fit_and_evaluates_fitted_model(danger, fitted_forest):
    model = build_model("random_forest", task="classification")
    with pytest.raises(TypeError, match=r"model\.fit\(inputs, targets\)"):
        Trainer(model, device="cpu").fit(danger, optimizer=None, loss_fn=torch.nn.NLLLoss())
    trainer = Trainer(fitted_forest, device="cpu")
    metrics = trainer.evaluate(danger, split="test")
    assert metrics and all(np.isfinite(list(metrics.values())))
    predictions = trainer.predict(danger, split="test")
    assert torch.cat(predictions).shape == (len(danger.get_split("test").targets), 2)


def test_estimator_module_validation():
    with pytest.raises(ValueError, match="task"):
        EstimatorModule(RandomForestClassifier(), task="segmentation")
    with pytest.raises(ValueError, match="predict_proba"):
        EstimatorModule(RandomForestRegressor(), task="classification")


def test_missing_xgboost_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "xgboost", None)  # makes "import xgboost" fail
    with pytest.raises(ImportError, match=r"pyhazards\[xgboost\]"):
        build_model("xgboost", task="classification")


@needs_xgboost
def test_xgboost_uses_library_defaults(danger):
    import xgboost

    model = build_model("xgboost", task="classification", n_jobs=2)
    assert isinstance(model.estimator, xgboost.XGBClassifier)
    assert repr(build_model("xgboost", task="classification").estimator.get_params()) == repr(
        xgboost.XGBClassifier().get_params()
    )
    train, test = danger.get_split("train"), danger.get_split("test")
    model.fit(train.inputs, train.targets)
    log_probs = model(test.inputs)
    assert log_probs.shape == (len(test.targets), 2)
    torch.testing.assert_close(log_probs.exp().sum(dim=1), torch.ones(len(test.targets)))
    with pytest.raises(TypeError, match="model.fit"):
        Trainer(model, device="cpu").fit(danger)


@needs_xgboost
def test_xgboost_regression():
    import xgboost

    model = build_model("xgboost", task="regression", n_estimators=5, n_jobs=1)
    assert isinstance(model.estimator, xgboost.XGBRegressor)
    x = torch.randn(20, 10, 25)
    assert model.fit(x, torch.randn(20))(x[:3]).shape == (3, 1)
