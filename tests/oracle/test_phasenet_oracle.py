"""PhaseNet checked against the official TensorFlow code and its released checkpoint.

Reference (pinned in repos.yaml): AI4EPS/PhaseNet at 62005c6 (MIT). The official network is built by
``UNet(config, mode="pred")`` from ``phasenet/model.py`` (only ``ModelConfig``, ``crop_and_concat``
and ``UNet`` are executed, inside a private ``tf.Graph``, so the module's global
``disable_eager_execution()`` call never runs) and restored from ``model/190703-214543/model_95.ckpt``
with ``tf.compat.v1.train.Saver``, exactly as ``phasenet/predict.py`` does. TensorFlow 2.11 is the
release in the repository's ``env.yaml``.

Checks: the pure-Python checkpoint reader returns TensorFlow's values; variable names, shapes and the
268,443 trainable parameters; the initialisation scheme; outputs in eval mode (moving statistics) and
in train mode (batch statistics) and the moving-average update, on random input of several lengths and
on 30-s windows of real STEAD traces; the official peak picking on the same probabilities; and the
SeisBench 0.12.6 port (GPL-3.0, second oracle) with its conversion of the same checkpoint.
"""

from __future__ import annotations

import logging
import warnings

import h5py
import numpy as np
import pytest
import torch

from oracle_utils import load_definitions, oracle_asset, oracle_package, oracle_repo
from pyhazards.models import build_model
from pyhazards.models._tf_checkpoint import read_tf_checkpoint
from pyhazards.models.phasenet import PhaseNet, load_phasenet_tf_checkpoint, state_dict_from_tf_checkpoint

CHECKPOINT = "model/190703-214543/model_95.ckpt"
# float32 convolutions in TensorFlow and PyTorch sum in different orders; differences stay ~1e-6.
TOLERANCE = dict(rtol=1e-5, atol=1e-5)


@pytest.fixture(scope="module")
def tf():
    module = oracle_package("tensorflow", "2.11.0", "requirements-earthquake.txt")
    module.get_logger().setLevel(logging.ERROR)
    return module


@pytest.fixture(scope="module")
def checkpoint():
    return oracle_repo("PhaseNet") / CHECKPOINT


class OfficialPhaseNet:
    """The official TF1 graph in its own ``tf.Graph`` / session."""

    def __init__(self, tf, checkpoint=None):
        root = oracle_repo("PhaseNet")
        names = ["ModelConfig", "crop_and_concat", "UNet"]
        defs = load_definitions(root / "phasenet" / "model.py", names, {"tf": tf, "np": np, "logging": logging})
        self.tf = tf
        self.graph = tf.Graph()
        with self.graph.as_default():
            # ModelConfig defaults equal model/190703-214543/config.log for every graph-building field.
            self.model = defs["UNet"](config=defs["ModelConfig"](), mode="pred")
            self.update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            self.session = tf.compat.v1.Session(graph=self.graph)
            if checkpoint is None:
                self.session.run(tf.compat.v1.global_variables_initializer())
            else:
                tf.compat.v1.train.Saver(tf.compat.v1.global_variables()).restore(self.session, str(checkpoint))

    def variables(self):
        with self.graph.as_default():
            variables = self.tf.compat.v1.global_variables()
        return {v.name[:-2]: value for v, value in zip(variables, self.session.run(variables))}

    def trainable_shapes(self):
        with self.graph.as_default():
            variables = self.tf.compat.v1.trainable_variables()
        # global_step is created with get_variable and so is "trainable"; it is not a network weight.
        return [(v.name[:-2], tuple(v.shape.as_list())) for v in variables if v.name != "global_step:0"]

    def __call__(self, x: np.ndarray, training: bool = False, logits: bool = False, update: bool = False):
        feed = {
            self.model.X: x.transpose(0, 2, 1)[:, :, None, :],  # (batch, samples, 1, channels)
            self.model.is_training: training,
            self.model.drop_rate: 0.0,
        }
        fetch = [self.model.logits if logits else self.model.preds] + (self.update_ops if update else [])
        out = self.session.run(fetch, feed_dict=feed)[0]
        return out[:, :, 0, :].transpose(0, 2, 1)  # (batch, classes, samples)


def _port(checkpoint) -> PhaseNet:
    return load_phasenet_tf_checkpoint(PhaseNet(), checkpoint).eval()


def _normalised(batch: int, length: int, seed: int) -> np.ndarray:
    x = np.random.default_rng(seed).standard_normal((batch, 3, length)).astype("float32")
    x -= x.mean(axis=-1, keepdims=True)
    return x / x.std(axis=-1, keepdims=True)


def _stead_windows(samples: int = 3000) -> np.ndarray:
    path = oracle_repo("EQTransformer") / "ModelsAndSampleData" / "100samples.hdf5"
    with h5py.File(path, "r") as handle:
        data = np.stack([handle["data"][name][()] for name in sorted(handle["data"])])
    # STEAD traces are (6000, 3) in E, N, Z order (PhaseNet's order); 30-s windows around the P arrivals.
    return np.ascontiguousarray(data[:, 300 : 300 + samples].transpose(0, 2, 1))


def test_checkpoint_reader_matches_tensorflow(tf, checkpoint):
    reader = tf.train.load_checkpoint(str(checkpoint))
    names = reader.get_variable_to_shape_map()
    ours = read_tf_checkpoint(checkpoint)
    assert set(ours) == set(names)
    for name in names:
        np.testing.assert_array_equal(ours[name], reader.get_tensor(name), err_msg=name)


def test_names_shapes_and_parameter_count(tf):
    official = OfficialPhaseNet(tf)
    shapes = official.trainable_shapes()
    assert sum(int(np.prod(shape)) for _, shape in shapes) == 268_443
    port = PhaseNet()
    assert sum(p.numel() for p in port.parameters() if p.requires_grad) == 268_443

    converted = state_dict_from_tf_checkpoint(official.variables())
    assert list(converted) == list(port.state_dict())  # same names, same (graph creation) order
    trainable = [name for name, _ in port.named_parameters()]
    renames = {"kernel": "weight", "gamma": "weight", "beta": "bias", "bias": "bias"}
    expected = [f"{n.split('/')[0]}.{n.split('/')[1]}.{renames[n.split('/')[2]]}" for n, _ in shapes]
    assert trainable == expected
    port.load_state_dict(converted, strict=True)


def test_initialisation_matches_reference_scheme(tf):
    official = state_dict_from_tf_checkpoint(OfficialPhaseNet(tf).variables())
    torch.manual_seed(0)
    port = PhaseNet().state_dict()
    for name, value in port.items():
        reference = official[name].float()
        if name.endswith("num_batches_tracked"):
            continue
        if not name.endswith(".weight") or "_bn" in name:
            torch.testing.assert_close(value, reference)  # zero biases, BN gamma 1 / beta 0, moving 0 / 1
            continue
        # Glorot uniform: both inside the same bound, with matching spread.
        fans = reference.shape[1] * reference.shape[2] + reference.shape[0] * reference.shape[2]
        limit = float(np.sqrt(6.0 / fans))
        assert value.abs().max() <= limit and reference.abs().max() <= limit, name
        if value.numel() >= 1000:
            assert abs(float(value.std()) / float(reference.std()) - 1) < 0.1, name
            assert abs(float(value.std()) - limit / np.sqrt(3)) < 0.1 * limit, name


@pytest.mark.parametrize("length", [3000, 3001, 1234])
def test_pretrained_outputs_match_official_in_eval_mode(tf, checkpoint, length):
    official = OfficialPhaseNet(tf, checkpoint)
    port = _port(checkpoint)
    x = _normalised(4, length, seed=length)
    with torch.no_grad():
        for logits in (True, False):
            expected = torch.from_numpy(official(x, logits=logits))
            torch.testing.assert_close(port(torch.from_numpy(x), logits=logits), expected, **TOLERANCE)


def test_train_mode_batch_statistics_and_moving_averages(tf, checkpoint):
    official = OfficialPhaseNet(tf, checkpoint)
    port = _port(checkpoint).train()
    x = _normalised(6, 3000, seed=1)
    expected = torch.from_numpy(official(x, training=True, logits=True, update=True))
    with torch.no_grad():
        actual = port(torch.from_numpy(x), logits=True)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)  # batch statistics in float32
    # One moving-average step (momentum 0.99) from the checkpoint statistics.
    updated = state_dict_from_tf_checkpoint(official.variables())
    for name, value in port.state_dict().items():
        if name.endswith("running_mean") or name.endswith("running_var"):
            torch.testing.assert_close(value, updated[name], rtol=1e-4, atol=1e-6, msg=name)


def test_real_stead_windows_and_official_peak_picking(tf, checkpoint):
    root = oracle_repo("PhaseNet")
    detect_peaks = load_definitions(root / "phasenet" / "detect_peaks.py", ["detect_peaks"], {"np": np, "warnings": warnings})[
        "detect_peaks"
    ]
    official = OfficialPhaseNet(tf, checkpoint)
    port = _port(checkpoint)
    raw = _stead_windows()
    normalised = raw - raw.mean(axis=-1, keepdims=True)  # phasenet/data_reader.py normalize()
    std = raw.std(axis=-1, keepdims=True)
    normalised /= np.where(std == 0, 1, std)
    expected = official(normalised.astype("float32"))
    with torch.no_grad():
        probabilities = port.annotate(torch.from_numpy(raw))
    torch.testing.assert_close(probabilities, torch.from_numpy(expected), **TOLERANCE)

    picks = port.extract_picks(probabilities)
    n_picks = 0
    for trace, trace_picks in zip(expected, picks):
        for phase, channel in (("P", 1), ("S", 2)):
            # phasenet/util.py detect_peaks_thread: mph=0.5, mpd=0.5/dt
            index, prob = detect_peaks(trace[channel], mph=0.5, mpd=0.5 / 0.01, show=False)
            assert [int(i) for i, _ in trace_picks[phase]] == [int(i) for i in index]
            np.testing.assert_allclose([p for _, p in trace_picks[phase]], prob, rtol=1e-4, atol=1e-5)
            n_picks += len(index)
    assert n_picks > 100  # the pretrained model picks most of the 100 events


def test_seisbench_port_agrees(checkpoint):
    seisbench_models = oracle_package("seisbench", "0.12.6", "requirements-earthquake.txt")
    import seisbench.models as sbm

    _ = seisbench_models
    reference = sbm.PhaseNet(in_channels=3, classes=3, phases="NPS", sampling_rate=100, norm="std")
    weights = oracle_asset("seisbench_phasenet_original") / "original.pt.v2"
    reference.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    reference.eval()
    port = _port(checkpoint)
    x = torch.from_numpy(_normalised(3, 3001, seed=7))  # SeisBench PhaseNet takes 3001 samples
    with torch.no_grad():
        torch.testing.assert_close(port(x), reference(x), **TOLERANCE)


def test_builder_loads_a_local_checkpoint(checkpoint):
    model = build_model("phasenet", task="picking", pretrained=str(checkpoint)).eval()
    torch.testing.assert_close(model.state_dict()["Output.output_conv.bias"], _port(checkpoint).state_dict()["Output.output_conv.bias"])
