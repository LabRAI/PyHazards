"""EQTransformer checked against the official Keras models and SeisBench's port.

References (pinned in repos.yaml / requirements-earthquake.txt):

- smousavi05/EQTransformer at a589da6 (MIT): the released models ``EqT_original_model.h5`` (paper
  model) and ``EqT_model_conservative.h5``, loaded with ``tf.keras.models.load_model`` (TensorFlow
  2.11) and the official custom layers of ``EQTransformer/core/EqT_utils.py``; the official ``picker``
  of the same file; the 100 STEAD sample traces ``ModelsAndSampleData/100samples.hdf5``.
- SeisBench 0.12.6 (GPL-3.0, oracle only) ``EQTransformer`` with its conversions of the two models.

The released graphs call SpatialDropout1D with ``training=True``, so the official models are random at
inference. For deterministic comparisons the Keras models are rebuilt from their stored configuration
with the SpatialDropout1D rate set to 0 (dropout has no weights; the graph is otherwise unchanged).

Tolerances: both implementations run in float32. In eval mode the outputs agree to ~2e-6 on the same
normalised input (the port's own float32-vs-float64 difference is 1e-6), and to ~1e-5 when ``annotate``
normalises the raw traces in float32 instead of the reference's float64; rtol 1e-4 / atol 1e-5 is used.
In train mode (BatchNorm on batch statistics over 6000-sample sequences) float32 round-off alone moves
the port's outputs by ~2e-4, so rtol / atol 1e-3 is used there.
"""

from __future__ import annotations

import importlib.util
import json
import math
import warnings

import numpy as np
import pytest
import torch

from oracle_utils import oracle_asset, oracle_package, oracle_repo
from pyhazards.models import build_model
from pyhazards.models.eqtransformer import (
    EQTransformer,
    keras_weight_map,
    load_eqtransformer_keras_weights,
    read_keras_h5_weights,
)

REQUIREMENTS = "requirements-earthquake.txt"
MODELS = {"original": "EqT_original_model.h5", "conservative": "EqT_model_conservative.h5"}
SEISBENCH = {
    "original": ("seisbench_eqtransformer_original_nonconservative", "original_nonconservative.pt.v1", 2, "non-conservative"),
    "conservative": ("seisbench_eqtransformer_original", "original.pt.v3", 3, "conservative"),
}
PARAMETERS = {"original": 371_639, "conservative": 376_423}
EVAL_TOL = dict(rtol=1e-4, atol=1e-5)
TRAIN_TOL = dict(rtol=1e-3, atol=1e-3)

_CACHE: dict = {}


def _tf():
    tf = oracle_package("tensorflow", "2.11.0", REQUIREMENTS)
    tf.config.threading.set_inter_op_parallelism_threads(8)
    return tf


def _official_utils():
    """``EqT_utils.py`` loaded as a standalone module (its package ``__init__`` imports the whole toolbox)."""
    if "utils" not in _CACHE:
        _tf()
        oracle_package("obspy", "1.5.1", REQUIREMENTS)  # EqT_utils imports obspy's trigger_onset
        path = oracle_repo("EQTransformer") / "EQTransformer" / "core" / "EqT_utils.py"
        spec = importlib.util.spec_from_file_location("eqt_utils_official", path)
        module = importlib.util.module_from_spec(spec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec.loader.exec_module(module)
        _CACHE["utils"] = module
    return _CACHE["utils"]


def _h5(variant: str):
    return oracle_repo("EQTransformer") / "ModelsAndSampleData" / MODELS[variant]


def _custom_objects():
    utils = _official_utils()
    return {
        "SeqSelfAttention": utils.SeqSelfAttention,
        "FeedForward": utils.FeedForward,
        "LayerNormalization": utils.LayerNormalization,
        "f1": utils.f1,
    }


def _keras(variant: str, all_dropout_off: bool = False):
    """The official model with SpatialDropout1D rate 0 (and optionally every other dropout off)."""
    key = ("keras", variant, all_dropout_off)
    if key not in _CACHE:
        tf = _tf()
        loaded = tf.keras.models.load_model(str(_h5(variant)), custom_objects=_custom_objects(), compile=False)
        config = loaded.get_config()
        for layer in config["layers"]:
            cfg = layer["config"]
            if layer["class_name"] == "SpatialDropout1D":
                cfg["rate"] = 0.0
            if all_dropout_off:
                if layer["class_name"] == "LSTM":
                    cfg["dropout"] = cfg["recurrent_dropout"] = 0.0
                if layer["class_name"] == "Bidirectional":
                    cfg["layer"]["config"]["dropout"] = cfg["layer"]["config"]["recurrent_dropout"] = 0.0
                if layer["class_name"] == "FeedForward":
                    cfg["dropout_rate"] = 0.0
        model = tf.keras.Model.from_config(config, custom_objects=_custom_objects())
        model.set_weights(loaded.get_weights())
        _CACHE[key] = (model, loaded)
    return _CACHE[key][0]


def _keras_outputs(model, x_wc: np.ndarray, training: bool = False) -> np.ndarray:
    """Keras ``(batch, 6000, 3)`` input -> ``(batch, 3, 6000)`` (detection, P, S)."""
    if training:
        outputs = [o.numpy() for o in model(x_wc, training=True)]
    else:
        outputs = model.predict(x_wc, batch_size=16, verbose=0)
    return np.concatenate([o[..., 0][:, None, :] for o in outputs], axis=1)


def _port(variant: str, **kwargs) -> EQTransformer:
    model = EQTransformer(variant=variant, **kwargs)
    load_eqtransformer_keras_weights(model, _h5(variant))
    return model.eval()


def _stead_samples(count: int = 100):
    """The 100 STEAD traces shipped with the official repository, normalised as the official tester does."""
    if "stead" not in _CACHE:
        import h5py

        utils = _official_utils()
        path = oracle_repo("EQTransformer") / "ModelsAndSampleData" / "100samples.hdf5"
        raw, normalised, arrivals = [], [], []
        with h5py.File(path, "r") as handle:
            for name in sorted(handle["data"]):
                dataset = handle["data"][name]
                data = np.array(dataset, dtype=np.float32)  # (6000, 3), components E, N, Z
                raw.append(data.copy())
                normalised.append(utils.normalize(data.astype(np.float64), mode="std").astype(np.float32))
                arrivals.append((int(dataset.attrs["p_arrival_sample"]), int(dataset.attrs["s_arrival_sample"])))
        _CACHE["stead"] = (np.stack(raw), np.stack(normalised), arrivals)
    raw, normalised, arrivals = _CACHE["stead"]
    return raw[:count], normalised[:count], arrivals[:count]


@pytest.mark.parametrize("variant", sorted(MODELS))
def test_parameter_counts_and_strict_loading(variant):
    keras_model = _keras(variant)
    keras_trainable = sum(int(np.prod(w.shape)) for w in keras_model.trainable_weights)
    port = build_model("eqtransformer", task="picking", variant=variant)
    assert sum(p.numel() for p in port.parameters() if p.requires_grad) == keras_trainable == PARAMETERS[variant]
    # Every Keras weight maps to exactly one port tensor (checked inside the loader) and the
    # mapping covers every parameter and buffer except BatchNorm's num_batches_tracked.
    weights = read_keras_h5_weights(_h5(variant))
    mapping = keras_weight_map(port.lstm_blocks)
    assert sorted(set(mapping.values())) == sorted(weights)
    load_eqtransformer_keras_weights(port, _h5(variant))
    # The Keras topology agrees with the port: the stored graph's layer names are the ones the map uses.
    import h5py

    with h5py.File(_h5(variant), "r") as handle:
        layers = {layer["name"]: layer for layer in json.loads(handle.attrs["model_config"])["config"]["layers"]}
    inbound = lambda name: layers[name]["inbound_nodes"][0][0][0]  # noqa: E731
    lstm_p, lstm_s = mapping["lstm_p.kernel"][0], mapping["lstm_s.kernel"][0]
    assert inbound("attentionP") == lstm_p and inbound("attentionS") == lstm_s
    decoders = {
        branch: [mapping[f"{branch}.convs.{i}.weight"][0] for i in range(7)]
        for branch in ("decoder_d", "decoder_p", "decoder_s")
    }
    assert inbound("detector") == decoders["decoder_d"][-1]
    assert inbound("picker_P") == decoders["decoder_p"][-1]
    assert inbound("picker_S") == decoders["decoder_s"][-1]
    for i in range(len(port.res_cnn)):
        conv1 = mapping[f"res_cnn.{i}.conv1.weight"][0]
        bn2 = mapping[f"res_cnn.{i}.bn2.weight"][0]
        assert inbound(bn2) == conv1


@pytest.mark.parametrize("variant", sorted(MODELS))
def test_eval_outputs_match_official_keras(variant):
    keras_model = _keras(variant)
    port = _port(variant)
    rng = np.random.default_rng(0)
    x_random = rng.standard_normal((2, 6000, 3)).astype(np.float32)
    raw, normalised, _ = _stead_samples(24)
    x = np.concatenate([x_random, normalised])
    expected = _keras_outputs(keras_model, x)
    with torch.no_grad():
        actual = port(torch.from_numpy(x).transpose(1, 2)).numpy()
        annotated = port.annotate(torch.from_numpy(raw).transpose(1, 2)).numpy()
    np.testing.assert_allclose(actual, expected, **EVAL_TOL)
    # annotate() applies the official normalisation (EqT_utils.normalize, mode "std") itself.
    np.testing.assert_allclose(annotated, expected[2:], **EVAL_TOL)


@pytest.mark.parametrize("variant", sorted(MODELS))
def test_train_mode_matches_official_keras_without_dropout(variant):
    # Dropout masks come from different random generators, so every dropout is switched off on both
    # sides; what remains is BatchNorm with batch statistics.
    keras_model = _keras(variant, all_dropout_off=True)
    port = EQTransformer(variant=variant, drop_rate=0.0)
    load_eqtransformer_keras_weights(port, _h5(variant))
    port.train()
    x = np.random.default_rng(1).standard_normal((3, 6000, 3)).astype(np.float32)
    expected = _keras_outputs(keras_model, x, training=True)
    actual = port(torch.from_numpy(x).transpose(1, 2)).detach().numpy()
    np.testing.assert_allclose(actual, expected, **TRAIN_TOL)


@pytest.mark.parametrize("variant", sorted(MODELS))
def test_official_graph_applies_spatial_dropout_at_inference(variant):
    """The released graphs keep SpatialDropout1D active in predict(); the port does with mc_dropout=True."""
    tf = _tf()
    loaded = tf.keras.models.load_model(str(_h5(variant)), custom_objects=_custom_objects(), compile=False)
    config = loaded.get_config()
    calls = [
        node[0][3]
        for layer in config["layers"]
        if layer["class_name"] == "SpatialDropout1D"
        for node in layer["inbound_nodes"]
    ]
    assert calls and all(kwargs == {"training": True} for kwargs in calls)
    x = np.random.default_rng(2).standard_normal((1, 6000, 3)).astype(np.float32)
    first, second = _keras_outputs(loaded, x), _keras_outputs(loaded, x)
    assert np.abs(first - second).max() > 1e-4
    port = _port(variant)
    xt = torch.from_numpy(x).transpose(1, 2)
    with torch.no_grad():
        assert torch.equal(port(xt), port(xt))
        port.mc_dropout = True
        assert (port(xt) - port(xt)).abs().max() > 1e-4


@pytest.mark.parametrize("variant", sorted(MODELS))
def test_outputs_match_seisbench(variant):
    sbm = oracle_package("seisbench", "0.12.6", REQUIREMENTS)
    import seisbench.models as sb_models

    asset, filename, lstm_blocks, compatible = SEISBENCH[variant]
    reference = sb_models.EQTransformer(lstm_blocks=lstm_blocks, original_compatible=compatible)
    state = torch.load(oracle_asset(asset) / filename, map_location="cpu", weights_only=True)
    reference.load_state_dict(state, strict=True)
    reference.eval()
    assert sbm.__version__ == "0.12.6"
    port = _port(variant)
    _, normalised, _ = _stead_samples(8)
    x = torch.from_numpy(np.concatenate([normalised, np.random.default_rng(3).standard_normal((2, 6000, 3)).astype(np.float32)])).transpose(1, 2)
    with torch.no_grad():
        # SeisBench's converted weights take components in the order Z, N, E.
        expected = torch.stack(reference(x[:, [2, 1, 0]]), dim=1)
        actual = port(x)
    torch.testing.assert_close(actual, expected, **EVAL_TOL)


def _official_matches(utils, probabilities: np.ndarray, thresholds):
    args = {
        "detection_threshold": thresholds[0],
        "P_threshold": thresholds[1],
        "S_threshold": thresholds[2],
        "estimate_uncertainty": False,
    }
    zeros = np.zeros(probabilities.shape[-1], dtype=probabilities.dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matches, _, _ = utils.picker(args, probabilities[0], probabilities[1], probabilities[2], zeros, zeros, zeros)
    return matches


@pytest.mark.parametrize("thresholds", [(0.2, 0.1, 0.1), (0.5, 0.3, 0.3), (0.05, 0.02, 0.02)])
def test_picks_match_official_picker(thresholds):
    """``extract_picks`` / ``extract_detections`` reproduce ``EqT_utils.picker`` on the 100 STEAD traces."""
    utils = _official_utils()
    keras_model = _keras("original")
    _, normalised, _ = _stead_samples(100)
    probabilities = _keras_outputs(keras_model, normalised)
    # Synthetic traces with several events and noise exercise the association rules further.
    rng = np.random.default_rng(4)
    synthetic = np.clip(rng.random((20, 3, 6000)).astype(np.float32) ** 6 + 0.0, 0.0, 1.0)
    for trace in synthetic:
        for start in rng.integers(0, 5800, size=3):
            trace[0, start : start + rng.integers(5, 400)] = rng.uniform(0.0, 1.0)
    probabilities = np.concatenate([probabilities, synthetic])
    port = EQTransformer()
    tensor = torch.from_numpy(probabilities)
    picks_all = port.extract_picks(tensor, *thresholds, all_events=True)
    picks_first = port.extract_picks(tensor, *thresholds)
    detections = port.extract_detections(tensor, *thresholds)
    n_events = 0
    for trace, all_picks, first_picks, events in zip(probabilities, picks_all, picks_first, detections):
        matches = _official_matches(utils, trace, thresholds)
        n_events += len(matches)
        assert [(on, off) for on, off, _ in events] == [(int(on), int(match[0])) for on, match in matches.items()]
        for (on, off, probability), match in zip(events, matches.values()):
            assert probability == pytest.approx(float(match[1]), abs=1e-3)
        expected_p = [float(m[3]) for m in matches.values() if m[3] is not None]
        expected_s = [float(m[6]) for m in matches.values() if m[6] is not None]
        assert [p for p, _ in all_picks["P"]] == expected_p
        assert [s for s, _ in all_picks["S"]] == expected_s
        assert [prob for _, prob in all_picks["P"]] == pytest.approx([float(m[4]) for m in matches.values() if m[3] is not None])
        first = list(matches.values())[:1]
        assert [p for p, _ in first_picks["P"]] == [float(m[3]) for m in first if m[3] is not None]
        assert [s for s, _ in first_picks["S"]] == [float(m[6]) for m in first if m[6] is not None]
    assert n_events > 50


def test_port_picks_on_stead_samples_equal_official_pipeline():
    """End to end: raw STEAD traces -> annotate -> extract_picks equals Keras outputs -> official picker."""
    utils = _official_utils()
    keras_model = _keras("original")
    raw, normalised, arrivals = _stead_samples(100)
    expected = _keras_outputs(keras_model, normalised)
    port = _port("original")
    with torch.no_grad():
        annotations = port.annotate(torch.from_numpy(raw).transpose(1, 2))
    picks = port.extract_picks(annotations)
    residuals = []
    for trace_picks, trace, (p_true, s_true) in zip(picks, expected, arrivals):
        matches = list(_official_matches(utils, trace, (0.2, 0.1, 0.1)).values())[:1]
        assert [p for p, _ in trace_picks["P"]] == [float(m[3]) for m in matches if m[3] is not None]
        assert [s for s, _ in trace_picks["S"]] == [float(m[6]) for m in matches if m[6] is not None]
        if trace_picks["P"]:
            residuals.append(trace_picks["P"][0][0] - p_true)
    # The paper model picks these (training-set) traces well: most P picks within 0.1 s.
    assert np.mean(np.abs(residuals) <= 10) > 0.9


def _keras_initialised(variant: str):
    tf = _tf()
    tf.random.set_seed(0)
    loaded = _CACHE.get(("keras", variant, False), (None, None))[1]
    if loaded is None:
        _keras(variant)
        loaded = _CACHE[("keras", variant, False)][1]
    return tf.keras.Model.from_config(loaded.get_config(), custom_objects=_custom_objects())


@pytest.mark.parametrize("variant", sorted(MODELS))
def test_initialisation_matches_keras_distributions(variant):
    """Same initialisers as Keras (random generators differ, so distributions are compared)."""
    fresh = _keras_initialised(variant)
    keras_weights = {}
    for layer in fresh.layers:
        for weight in layer.weights:
            name = weight.name.split(":")[0]
            short = name[len(layer.name) + 1 :] if name.startswith(layer.name + "/") else name
            keras_weights[(layer.name, short)] = weight.numpy()
    torch.manual_seed(0)
    port = EQTransformer(variant=variant)
    state = port.state_dict()
    for key, (layer, name) in keras_weight_map(port.lstm_blocks).items():
        if (layer, name) not in keras_weights:
            # TF 2.11 names LSTM cell weights ".../lstm_cell_N/kernel"; match by suffix.
            candidates = [k for k in keras_weights if k[0] == layer and k[1].split("/")[-1] == name.split("/")[-1]
                          and (("forward" in k[1]) == ("forward" in name)) and (("backward" in k[1]) == ("backward" in name))]
            assert len(candidates) == 1, (key, layer, name)
            expected = keras_weights[candidates[0]]
        else:
            expected = keras_weights[(layer, name)]
        actual = state[key].double().numpy()
        if expected.ndim == 3:
            expected = expected.transpose(2, 1, 0)
        assert actual.shape == expected.shape, key
        if np.unique(expected).size <= 2:  # constants: zeros / ones, LSTM bias with unit forget bias
            np.testing.assert_array_equal(actual, expected, err_msg=key)
            continue
        if name.endswith("recurrent_kernel"):  # orthogonal: orthonormal rows on both sides
            np.testing.assert_allclose(actual @ actual.T, np.eye(actual.shape[0]), atol=1e-5, err_msg=key)
            np.testing.assert_allclose(expected @ expected.T, np.eye(expected.shape[0]), atol=1e-5, err_msg=key)
            continue
        # Same support (glorot_uniform bound, or glorot_normal truncated at two standard deviations)
        # on both sides, and the same spread.
        keras_shape = state[key].shape if actual.ndim != 3 else expected.transpose(2, 1, 0).shape
        if len(keras_shape) == 3:
            fan_in, fan_out = keras_shape[0] * keras_shape[1], keras_shape[0] * keras_shape[2]
        else:
            fan_in, fan_out = keras_shape
        if key.rsplit(".", 1)[-1] in ("Wt", "Wx", "Wa", "W1", "W2"):
            bound = 2.0 * math.sqrt(2.0 / (fan_in + fan_out)) / 0.87962566103423978
        else:
            bound = math.sqrt(6.0 / (fan_in + fan_out))
        assert np.abs(actual).max() <= bound + 1e-6 and np.abs(expected).max() <= bound + 1e-6, key
        assert np.abs(actual).max() > 0.5 * bound, key
        if expected.size >= 100:
            assert actual.std() == pytest.approx(expected.std(), rel=0.2), key
    assert math.isclose(port.transformer_d0.norm1.epsilon, 1e-14, rel_tol=1e-12)
