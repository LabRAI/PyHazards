"""random_forest / xgboost checked against the Kondylatos et al. (2022) random-forest notebook.

Reference (pinned in repos.yaml): Orion-AI-Lab/wildfire_forecasting, ``notebooks/RF.ipynb``. The test
executes the notebook's own code cells (feature construction, hyperparameters, classifier, prediction)
on a synthetic dataset served the way the notebook reads ``FireDataset_npy(access_mode='temporal')``
(``batch_size=1`` tuples of dynamic ``(1, 10, 10)``, static ``(1, 5)``, land cover ``(1, 10)`` and
label). Cell 0 (imports of the Greek-datacube dataset class) and cells 1-2 (feature names, data
paths) are replaced by that loader. PyHazards gets the same samples as the LSTM's daily tensor
``(batch, 10, 25)``, built by the reference's ``combine_dynamic_static_inputs``
(``wildfire_forecasting/models/greece_fire_models.py``; only that function is executed, since the
module imports Lightning).

XGBoost has no reference code (the paper's hyperparameters are in its Supporting Information), so
the ``xgboost`` model is compared with ``xgboost.XGBClassifier()`` at its library defaults, at the
version pinned in requirements.txt, on the notebook's features.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from oracle_utils import load_definitions, oracle_package, oracle_repo
from pyhazards.models import build_model, kondylatos_instance_features

XGBOOST_VERSION = "3.2.0"
SPLIT_SIZES = {"train": 300, "val": 60, "test": 150}


def _notebook_cells(root) -> dict:
    notebook = json.loads((root / "notebooks" / "RF.ipynb").read_text(encoding="utf-8"))
    sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]

    def find(marker: str) -> str:
        matches = [source for source in sources if marker in source]
        assert len(matches) == 1, f"RF.ipynb: expected one cell containing {marker!r}"
        return matches[0]

    return {
        "features": find("X_train = []"),
        "hyperparameters": find("n_est = 100"),
        "classifier": find("RandomForestClassifier(n_estimators=n_est"),
        "prediction": find("probs_pred = clf.predict_proba"),
    }


def _synthetic_split(rng: np.random.Generator, n: int):
    """Daily covariates with a fire signal; values are float32-representable, stored as float64."""
    dynamic = rng.normal(size=(n, 10, 10)).astype(np.float32).astype(np.float64)
    static = rng.uniform(size=(n, 5)).astype(np.float32).astype(np.float64)
    land_cover = rng.dirichlet(np.ones(10), size=n).astype(np.float32).astype(np.float64)
    score = (
        dynamic[:, -3:, 4].mean(axis=1)  # e.g. maximum temperature over the last days
        - dynamic[:, -3:, 9].mean(axis=1)  # e.g. minimum relative humidity
        + static[:, 0]
        + rng.normal(scale=0.7, size=n)
    )
    labels = (score > np.quantile(score, 2 / 3)).astype(np.int64)  # two negatives per positive
    # Missing values on earlier days, so that the notebook's np.nanmean matters.
    earlier = dynamic[:, :-1]
    earlier[rng.uniform(size=earlier.shape) < 0.05] = np.nan
    return dynamic, static, land_cover, labels


@pytest.fixture(scope="module")
def notebook_run():
    root = oracle_repo("wildfire_forecasting")
    cells = _notebook_cells(root)
    combine = load_definitions(
        root / "wildfire_forecasting" / "models" / "greece_fire_models.py",
        ["combine_dynamic_static_inputs"],
        {"torch": torch},
    )["combine_dynamic_static_inputs"]

    rng = np.random.default_rng(2022)
    splits = {name: _synthetic_split(rng, size) for name, size in SPLIT_SIZES.items()}
    # The notebook's training loader shuffles; a fixed order makes the forest comparable.
    dataloaders = {
        name: DataLoader(TensorDataset(*(torch.from_numpy(a) for a in arrays)), batch_size=1, shuffle=False)
        for name, arrays in splits.items()
    }
    namespace = {
        "torch": torch,
        "np": np,
        "warnings": warnings,
        "RandomForestClassifier": RandomForestClassifier,
        "classification_report": classification_report,
        "roc_auc_score": roc_auc_score,
        "average_precision_score": average_precision_score,
        "dataloaders": dataloaders,
    }
    for key in ("features", "hyperparameters", "classifier", "prediction"):
        exec(compile(cells[key], f"RF.ipynb[{key}]", "exec"), namespace)

    daily = {}
    for name, (dynamic, static, land_cover, _) in splits.items():
        daily[name] = combine(torch.from_numpy(dynamic), torch.from_numpy(static), torch.from_numpy(land_cover), "temporal")
    labels = {name: torch.from_numpy(arrays[3]) for name, arrays in splits.items()}
    return namespace, daily, labels


def test_daily_tensor_is_the_lstm_input(notebook_run):
    _, daily, _ = notebook_run
    assert daily["train"].shape == (SPLIT_SIZES["train"], 10, 25)
    assert daily["train"].dtype == torch.float32  # combine_dynamic_static_inputs ends with .float()
    lstm = build_model("wildfire_forecasting", task="classification")
    with torch.no_grad():
        assert lstm(torch.nan_to_num(daily["test"][:4])).shape == (4, 2)


def test_instance_features_match_notebook(notebook_run):
    namespace, daily, _ = notebook_run
    for split, notebook_features in (("train", namespace["X_train"]), ("val", namespace["X_val"]), ("test", namespace["X_test"])):
        features = kondylatos_instance_features(daily[split])
        assert features.shape == notebook_features.shape == (SPLIT_SIZES[split], 35)
        np.testing.assert_array_equal(features, notebook_features)  # NaN positions included
    assert torch.isnan(daily["train"]).any()  # the means must skip missing days, as np.nanmean does
    assert not np.isnan(namespace["X_train"]).any()  # the last day is always observed


def test_random_forest_matches_notebook(notebook_run):
    namespace, daily, labels = notebook_run
    clf = namespace["clf"]
    model = build_model("random_forest", task="classification")
    assert model.estimator.get_params() == clf.get_params()

    model.fit(daily["train"], labels["train"])
    probabilities = model.predict_proba(daily["test"])
    np.testing.assert_array_equal(probabilities, clf.predict_proba(namespace["X_test"]))
    np.testing.assert_array_equal(probabilities[:, 1], namespace["probs_pred"])
    np.testing.assert_array_equal(model.predict(daily["test"]), namespace["y_pred"])
    assert probabilities[:, 1].std() > 0.1  # a non-trivial forest
    # Another seed gives another forest, so the equality above is not vacuous.
    other = build_model("random_forest", task="classification", random_state=0).fit(daily["train"], labels["train"])
    assert not np.array_equal(other.predict_proba(daily["test"]), probabilities)
    # Ready-made 35-feature rows give the same result.
    np.testing.assert_array_equal(model.predict_proba(namespace["X_test"]), probabilities)

    log_probs = model(daily["test"].double())
    assert log_probs.shape == (SPLIT_SIZES["test"], 2) and log_probs.dtype == torch.float64
    torch.testing.assert_close(log_probs.exp(), torch.from_numpy(probabilities), rtol=1e-12, atol=1e-15)
    torch.testing.assert_close(model(daily["test"]).exp(), torch.from_numpy(probabilities).float())


def test_xgboost_matches_library_defaults(notebook_run):
    xgboost = oracle_package("xgboost", XGBOOST_VERSION)
    namespace, daily, labels = notebook_run
    assert repr(build_model("xgboost", task="classification").estimator.get_params()) == repr(
        xgboost.XGBClassifier().get_params()
    )

    reference = xgboost.XGBClassifier(n_jobs=4).fit(namespace["X_train"], namespace["y_train"].ravel())
    model = build_model("xgboost", task="classification", n_jobs=4).fit(daily["train"], labels["train"])
    assert model.estimator.get_booster().num_boosted_rounds() == 100
    np.testing.assert_array_equal(model.predict_proba(daily["test"]), reference.predict_proba(namespace["X_test"]))
