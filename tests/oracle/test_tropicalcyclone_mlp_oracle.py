"""tropicalcyclone_mlp checked against the official Keras model (wenweixu/tropicalcyclone_MLP, pinned).

The official ``models.mlp`` is executed from the pinned ``models.py`` with Keras 3 on the PyTorch
backend (the original Keras 2.2.4 / TensorFlow 1.12 stack no longer installs); only its ``Adam(lr=...)``
argument is mapped to Keras 3's ``learning_rate``. Keras ``Dense`` computes ``x @ kernel + bias``
in every version, so the official weights are copied into the port and the outputs compared.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from oracle_utils import load_definitions, oracle_package, oracle_repo
from pyhazards.models import build_model
from pyhazards.models.tropicalcyclone_mlp import SHIPS_PREDICTORS, load_keras_weights

os.environ.setdefault("KERAS_BACKEND", "torch")

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "ships_xu2021" / "train_global_fill_REA_na_wo_img_scaled.fixture.csv"


def _official():
    keras = oracle_package("keras", "3.12.4", "requirements-tc.txt")
    if keras.backend.backend() != "torch":
        pytest.skip("set KERAS_BACKEND=torch before keras is imported")
    from keras.layers import Dense, Input
    from keras.models import Model
    from keras.optimizers import Adam

    def adam(lr=None, **kwargs):  # Keras 2 spelling used by the official code
        return Adam(learning_rate=lr, **kwargs)

    repo = oracle_repo("tropicalcyclone_MLP")
    scope = {"Dense": Dense, "Input": Input, "Model": Model, "Adam": adam}
    official = load_definitions(repo / "models.py", ["mlp"], scope)["mlp"]
    return keras, official, repo


def _hand_features(repo: Path):
    """The official predictor list ``utils.hand_features`` (utils.py imports h5py, scipy, ...)."""
    source = (repo / "utils.py").read_text(encoding="utf-8")
    start = source.index("hand_features = [")
    end = source.index("]", start)
    scope: dict = {}
    exec(source[start : end + 1], scope)
    return scope["hand_features"]


def test_architecture_and_parameter_count_match_official():
    keras, official, repo = _official()
    reference = official(input_shape=(121,))
    port = build_model("tropicalcyclone_mlp", task="regression")
    n_port = sum(p.numel() for p in port.parameters())
    assert reference.count_params() == n_port == 4_448_257
    dense = [layer for layer in reference.layers if layer.weights]
    assert [layer.name for layer in dense] == port.layer_names == ["dense", "dense_1", "dense_2"]
    assert [layer.activation.__name__ for layer in dense] == ["sigmoid", "relu", "linear"]
    assert list(port.activation_names) == ["sigmoid", "relu"]
    # Official training configuration (models.mlp compile): MAE loss, Adam with learning rate 1e-4.
    assert reference.loss == "mae"
    assert float(reference.optimizer.learning_rate) == pytest.approx(1e-4)
    # The predictor list is the official one, in order.
    assert tuple(_hand_features(repo)) == SHIPS_PREDICTORS


def test_official_weights_give_identical_outputs():
    keras, official, _ = _official()
    keras.utils.set_random_seed(0)
    reference = official(input_shape=(121,))
    port = build_model("tropicalcyclone_mlp", task="regression").eval()
    load_keras_weights(port, reference.get_weights())

    torch.manual_seed(1)
    x = torch.randn(32, 121)
    import pandas as pd

    real = torch.as_tensor(pd.read_csv(FIXTURE, usecols=list(SHIPS_PREDICTORS))[list(SHIPS_PREDICTORS)].to_numpy(np.float32))
    for inputs in (x, real):
        expected = np.asarray(reference.predict(inputs.numpy(), verbose=0))
        with torch.no_grad():
            actual = port(inputs).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_initialisation_follows_keras_defaults():
    """Glorot-uniform kernels and zero biases, like Keras Dense; draws differ between frameworks."""
    keras, official, _ = _official()
    keras.utils.set_random_seed(0)
    reference = official(input_shape=(121,))
    torch.manual_seed(0)
    port = build_model("tropicalcyclone_mlp", task="regression")
    for name, (kernel, bias) in zip(port.layer_names, zip(reference.get_weights()[::2], reference.get_weights()[1::2])):
        layer = getattr(port, name)
        fan_in, fan_out = kernel.shape
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        for weights in (np.asarray(kernel), layer.weight.detach().numpy()):
            assert np.abs(weights).max() <= limit + 1e-7
            assert weights.std() == pytest.approx(limit / np.sqrt(3.0), rel=0.05)
        assert not np.any(bias) and not torch.any(layer.bias)
