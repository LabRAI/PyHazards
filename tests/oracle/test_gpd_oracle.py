"""GPD checked against the official Keras model and ``gpd_predict.py`` (interseismic/generalized-phase-detection).

Reference: interseismic/generalized-phase-detection at ea81ef1 (pinned in repos.yaml, MIT): the network in
``model_pol.json`` with the released weights ``model_pol_best.hdf5``, run on TensorFlow 2.11 / tf.keras
(requirements-earthquake.txt). ``model_pol.json`` wraps the network in Keras' multi-GPU ``Lambda`` layers,
whose functions are stored as Python 3.6 bytecode and cannot be deserialised on Python 3.10; the oracle is
therefore built from the JSON's inner ``Sequential`` config (the network itself, unchanged) and the
weights are copied in by layer name from ``model_pol_best.hdf5``. SeisBench 0.12.6 (GPL-3.0, oracle only)
with its "original" conversion of the same weights is a second reference.

Checks: parameter count (1,741,003 trainable); Keras-style initialisation bounds; the weight file loads
strictly; eval-mode and train-mode outputs equal Keras on random windows and on real windows of the shipped
Anza 2016 data; BatchNorm moving statistics follow Keras 2.2.2's update; outputs equal SeisBench's GPD; and the ``gpd_predict.py`` pipeline (ObsPy filtering, sliding
windows, ``trigger_onset`` picks) gives identical picks with the official Keras model and with the port on
18 minutes of the Anza aftershock sequence, which are compared with the shipped ``anza2016.out``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from oracle_utils import load_definitions, oracle_asset, oracle_package, oracle_repo
from pyhazards.models import build_model
from pyhazards.models.gpd import GPD, load_gpd_keras_weights

ATOL = 1e-6  # float32 softmax probabilities; the remaining differences are summation-order rounding


def _repo():
    return oracle_repo("generalized-phase-detection")


def _keras_model(load_weights: bool = True):
    tf = oracle_package("tensorflow", "2.11.0", "requirements-earthquake.txt")
    import h5py

    root = _repo()
    config = json.loads((root / "model_pol.json").read_text())
    assert config["keras_version"] == "2.2.2"
    wrapper_layers = [layer["class_name"] for layer in config["config"]["layers"]]
    assert wrapper_layers == ["InputLayer", "Lambda", "Lambda", "Lambda", "Sequential", "Concatenate"]
    inner = next(layer for layer in config["config"]["layers"] if layer["class_name"] == "Sequential")["config"]
    model = tf.keras.Sequential.from_config(inner)
    if load_weights:
        with h5py.File(root / "model_pol_best.hdf5", "r") as handle:
            group = handle["model_weights/sequential_1"]
            stored = {name.decode() for name in group.attrs["weight_names"]}
            used = set()
            for layer in model.layers:
                values = []
                for weight in layer.weights:
                    key = f"{layer.name}_1/{weight.name.split('/')[-1]}"
                    values.append(np.asarray(group[key]))
                    used.add(key)
                if values:
                    layer.set_weights(values)
            assert used == stored
    return model


def _port(pretrained: bool = True) -> GPD:
    model = GPD()
    if pretrained:
        load_gpd_keras_weights(model, _repo() / "model_pol_best.hdf5")
    return model.eval()


def _max_normalised(windows: np.ndarray) -> np.ndarray:
    """``gpd_predict.py``: windows (n, 400, 3) divided by their max |amplitude|."""
    return windows / np.max(np.abs(windows), axis=(1, 2))[:, None, None]


def _anza_stream(start_offset_s: float, duration_s: float, filtered: bool = True):
    """The official pre-processing of ``gpd_predict.py`` on a slice of the Anza day (N, E, Z files)."""
    obspy = oracle_package("obspy", "1.5.1", "requirements-earthquake.txt")
    root = _repo()
    files = (root / "anza2016.in").read_text().split()
    assert [name.split(".")[-2] for name in files] == ["HHN", "HHE", "HHZ"]
    stream = obspy.Stream()
    for name in files:
        stream += obspy.read(str(root / name))
    latest_start = np.max([trace.stats.starttime for trace in stream])
    earliest_stop = np.min([trace.stats.endtime for trace in stream])
    stream.trim(latest_start, earliest_stop)
    start = latest_start + start_offset_s
    stream.trim(start, start + duration_s - stream[0].stats.delta)
    stream.detrend(type="linear")
    if filtered:
        stream.filter(type="bandpass", freqmin=3.0, freqmax=20.0)
    return stream


def _anza_windows(count: int = 64) -> np.ndarray:
    stream = _anza_stream(8 * 3600 + 4 * 60, 120.0)
    data = np.stack([trace.data for trace in stream], axis=-1)  # (samples, 3) in N, E, Z order
    starts = np.linspace(0, data.shape[0] - 400, count).astype(int)
    return _max_normalised(np.stack([data[s : s + 400] for s in starts])).astype(np.float32)


def test_parameter_count_and_initialisation_match_keras():
    keras_model = _keras_model(load_weights=False)
    trainable = sum(int(np.prod(weight.shape)) for weight in keras_model.trainable_weights)
    port = build_model("gpd", task="classification")
    assert sum(p.numel() for p in port.parameters()) == trainable == 1_741_003
    port_layers = dict(port.named_modules())
    for layer in keras_model.layers:
        if not layer.weights:
            continue
        module = port_layers[layer.name]
        for weight in layer.weights:
            kind = weight.name.split("/")[-1].split(":")[0]
            reference = weight.numpy()
            if kind == "kernel":
                ours = module.weight.detach().numpy()
                fan_in, fan_out = (np.prod(reference.shape[:-1]), reference.shape[-1])
                limit = np.sqrt(6.0 / (fan_in + fan_out))  # glorot_uniform (VarianceScaling fan_avg uniform)
                assert np.abs(reference).max() <= limit and np.abs(ours).max() <= limit
                assert abs(ours.std() - reference.std()) < 0.1 * limit
            else:
                target = {"bias": "bias", "gamma": "weight", "beta": "bias", "moving_mean": "running_mean", "moving_variance": "running_var"}[kind]
                np.testing.assert_array_equal(getattr(module, target).detach().numpy(), reference)


def test_released_weights_match_keras_in_eval_and_train_mode():
    rng = np.random.default_rng(0)
    random_windows = _max_normalised(rng.standard_normal((32, 400, 3))).astype(np.float32)
    for windows in (random_windows, _anza_windows()):
        # Fresh models for every comparison: a train-mode call updates the moving statistics.
        keras_model, port = _keras_model(), _port()
        expected = keras_model(windows, training=False).numpy()
        with torch.no_grad():
            actual = port(torch.from_numpy(windows).permute(0, 2, 1)).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=ATOL)

        # Train mode: BatchNorm normalises with batch statistics (GPD has no dropout).
        expected_train = keras_model(windows, training=True).numpy()
        port.train()
        with torch.no_grad():
            actual_train = port(torch.from_numpy(windows).permute(0, 2, 1)).numpy()
        # Batch statistics are float32 reductions over thousands of values; TensorFlow's multithreaded
        # reduction order varies between runs (observed differences up to 3e-6), hence atol 1e-5.
        np.testing.assert_allclose(actual_train, expected_train, rtol=1e-5, atol=1e-5)


def test_batchnorm_statistics_follow_keras_2_2_2():
    """Moving statistics of a training step, as Keras 2.2.2 (the version that trained GPD) updates them.

    ``keras/layers/normalization.py`` of Keras 2.2.2 does, after ``tf.nn.moments``::

        variance *= sample_size / (sample_size - (1.0 + self.epsilon))
        K.moving_average_update(self.moving_variance, variance, self.momentum)

    and its TensorFlow backend's ``moving_average_update`` is ``assign_moving_average(x, value, momentum,
    zero_debias=True)``. TensorFlow 2.11's tf.keras no longer does either, so the reference here is that
    update built from the same TensorFlow functions.
    """
    tf = oracle_package("tensorflow", "2.11.0", "requirements-earthquake.txt")
    from tensorflow.python.training import moving_averages

    port = _port()
    port.train()
    rng = np.random.default_rng(1)
    batches = [_max_normalised(rng.standard_normal((8, 400, 3))).astype(np.float32) for _ in range(3)]
    # Inputs of batch_normalization_1 (after conv1d_1, 3-D) and batch_normalization_5 (after dense_1, 2-D).
    captured = {name: [] for name in ("batch_normalization_1", "batch_normalization_5")}
    hooks = [
        getattr(port, name).register_forward_pre_hook(lambda module, args, name=name: captured[name].append(args[0].detach().numpy()))
        for name in captured
    ]
    initial = {name: (getattr(port, name).running_mean.clone(), getattr(port, name).running_var.clone()) for name in captured}
    for batch in batches:
        port(torch.from_numpy(batch).permute(0, 2, 1))
    for hook in hooks:
        hook.remove()

    graph = tf.Graph()
    with graph.as_default():
        for name, inputs in captured.items():
            mean0, var0 = initial[name]
            moving_mean = tf.compat.v1.Variable(mean0.numpy(), name=f"{name}_mean")
            moving_var = tf.compat.v1.Variable(var0.numpy(), name=f"{name}_var")
            feed = tf.compat.v1.placeholder(tf.float32, inputs[0].shape)
            axes = [0] if len(inputs[0].shape) == 2 else [0, 2]  # PyTorch layout (batch, channels[, samples])
            mean, variance = tf.nn.moments(feed, axes)
            sample_size = tf.cast(tf.reduce_prod([tf.shape(feed)[axis] for axis in axes]), tf.float32)
            variance *= sample_size / (sample_size - (1.0 + 1e-3))
            updates = [
                moving_averages.assign_moving_average(moving_mean, mean, 0.99, zero_debias=True),
                moving_averages.assign_moving_average(moving_var, variance, 0.99, zero_debias=True),
            ]
            with tf.compat.v1.Session(graph=graph) as session:
                session.run(tf.compat.v1.global_variables_initializer())
                for value in inputs:
                    expected_mean, expected_var = session.run(updates, {feed: value})
            module = getattr(port, name)
            np.testing.assert_allclose(module.running_mean.numpy(), expected_mean, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(module.running_var.numpy(), expected_var, rtol=1e-5, atol=1e-6)
            assert int(module.num_batches_tracked) == len(inputs)


def test_matches_seisbench_original_gpd():
    oracle_package("seisbench", "0.12.6", "requirements-earthquake.txt")
    import seisbench.models as sbm

    reference = sbm.GPD(in_channels=3, classes=3, phases="PSN", original_compatible=True)
    state = torch.load(oracle_asset("seisbench_gpd_original") / "original.pt", map_location="cpu", weights_only=True)
    reference.load_state_dict(state, strict=True)
    reference.eval()
    port = _port()
    windows = torch.from_numpy(_anza_windows()).permute(0, 2, 1)  # N, E, Z
    with torch.no_grad():
        expected = reference(windows[:, [2, 0, 1]])  # SeisBench's conversion takes Z, N, E
        actual = port(windows)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=ATOL)
    # SeisBench's conversion permuted the first convolution's input channels from N, E, Z to Z, N, E.
    torch.testing.assert_close(state["conv1.weight"], port.conv1d_1.weight[:, [2, 0, 1]], rtol=0, atol=0)


def _official_picks(keras_model, stream, trigger_onset):
    """``gpd_predict.py`` from the sliding window to the picks (its hyper-parameters)."""
    names = load_definitions(_repo() / "gpd_predict.py", ["sliding_window"], {"np": np})
    sliding_window = names["sliding_window"]
    n_shift, n_win, n_feat, min_proba = 10, 200, 400, 0.95
    dt = stream[0].stats.delta
    tt = (np.arange(0, stream[0].data.size, n_shift) + n_win) * dt
    windows = np.zeros((sliding_window(stream[0].data, n_feat, stepsize=n_shift).shape[0], n_feat, 3))
    for channel in range(3):
        windows[:, :, channel] = sliding_window(stream[channel].data, n_feat, stepsize=n_shift)
    windows = windows / np.max(np.abs(windows), axis=(1, 2))[:, None, None]
    tt = tt[: windows.shape[0]]
    ts = keras_model.predict(windows, verbose=False, batch_size=3000)
    picks = {}
    for phase, column in (("P", 0), ("S", 1)):
        picks[phase] = []
        for on, off in trigger_onset(ts[:, column], min_proba, 0.1):
            if off == on:
                continue
            pick = np.argmax(ts[on:off, column]) + on
            picks[phase].append(round(float(tt[pick]) / dt))
    return picks, ts


def test_gpd_predict_pipeline_picks_are_identical_on_anza_data():
    # 08:02-08:20 UTC on 2016-06-10, the first minutes of the Mw 5.2 Anza aftershock sequence. The slice
    # starts a multiple of 10 samples after the day start, so window times fall on the official grid.
    start_offset, duration = 8 * 3600 + 2 * 60, 18 * 60
    stream = _anza_stream(start_offset, duration)
    from obspy.signal.trigger import trigger_onset  # the function gpd_predict.py imports

    keras_model = _keras_model()
    expected, keras_probabilities = _official_picks(keras_model, stream, trigger_onset)

    port = _port()
    waveforms = torch.from_numpy(np.stack([trace.data for trace in stream]))[None]  # float64, N, E, Z
    with torch.no_grad():
        annotations = port.annotate(waveforms, stride=10)
    np.testing.assert_allclose(annotations[0].numpy().T, keras_probabilities, rtol=1e-5, atol=ATOL)
    picks = port.extract_picks(annotations)[0]
    actual = {phase: [int(sample) for sample, _ in picks[phase]] for phase in ("P", "S")}
    assert actual == expected
    assert len(expected["P"]) > 30 and len(expected["S"]) > 90

    # The shipped anza2016.out (the authors' whole-day run with Keras 2.2.2): inside the slice, away from
    # the filter edges, the picks are the same to the millisecond (33 P and 96 S picks).
    obspy = oracle_package("obspy", "1.5.1", "requirements-earthquake.txt")
    t0 = stream[0].stats.starttime
    lo, hi = 60.0, duration - 60.0
    shipped = {"P": [], "S": []}
    for line in (_repo() / "anza2016.out").read_text().splitlines():
        _, _, phase, stamp = line.split()
        offset = obspy.UTCDateTime(stamp) - t0
        if lo <= offset <= hi:
            shipped[phase].append(offset)
    for phase in ("P", "S"):
        ours = np.array([sample * 0.01 for sample in actual[phase] if lo <= sample * 0.01 <= hi])
        theirs = np.array(shipped[phase])
        exact = sum(np.any(np.abs(ours - t) < 1e-3) for t in theirs)
        assert len(ours) == len(theirs), (phase, len(ours), len(theirs))
        assert exact == len(theirs), (phase, exact, len(theirs))


def test_zero_window_is_left_at_zero():
    port = _port()
    waveforms = torch.zeros(1, 3, 600)
    with torch.no_grad():
        annotations = port.annotate(waveforms)
    assert torch.isfinite(annotations).all()
    with pytest.raises(ValueError):
        port.annotate(torch.zeros(1, 3, 399))
