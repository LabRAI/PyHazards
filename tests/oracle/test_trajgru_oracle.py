"""TrajGRU (and the HKO-7 ConvGRU baseline) checked against the official MXNet implementation.

Reference (pinned in repos.yaml): sxjscience/HKO-7 (MIT), the MXNet code of Shi et al. (NeurIPS
2017), and its released checkpoints for HKO-7 and MovingMNIST++ (``hko7_hko_weights``,
``hko7_movingmnist_weights``). The authors tested the code on MXNet 0.12; it runs unchanged on
MXNet 1.9.1, the last release of the retired project, which needs numpy < 1.24
(requirements-trajgru.txt, a suite of its own in the Oracle workflow).

The reference networks are built exactly as the official training scripts build them:
``nowcasting.config`` merged with an official configuration file, a ``MovingMNISTFactory`` (its
loss is the only part that depends on the dataset, and the frame size is taken from
``cfg.MOVINGMNIST.IMG_SIZE``), ``encoder_forecaster_build_networks`` (which also applies the
official ``MSRAPrelu`` initialisation), ``EncoderForecasterStates`` for the zero initial states, and
the encoder/forecaster forward passes of ``mnist_get_prediction``.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from oracle_utils import oracle_asset, oracle_package, oracle_repo
from pyhazards.models import build_model
from pyhazards.models.trajgru import TRAJGRU_CONFIGS, TrajGRU, load_hko7_params, warp

HKO_DIR = "experiments/hko/configurations"
MNIST_DIR = "experiments/movingmnist/configurations"
HKO_TRAJGRU = f"{HKO_DIR}/trajgru_55_55_33_1_64_1_192_1_192_13_13_9_b4.yml"
HKO_CONVGRU = f"{HKO_DIR}/convgru_55_55_33_1_64_1_192_1_192_b4.yml"
MNIST = {name: f"{MNIST_DIR}/{name}.yml" for name in (
    "trajgru_1_64_1_96_1_96_L5", "trajgru_1_64_1_96_1_96_L9", "trajgru_1_64_1_96_1_96_L13",
    "trajgru_1_64_1_96_1_96_L17", "convgru_1_64_1_96_1_96_K3D2", "convgru_1_64_1_96_1_96_K5",
    "convgru_1_64_1_96_1_96_K7",
)}
# Released checkpoints: (asset, directory, iteration) for each configuration file.
CHECKPOINTS = {
    HKO_TRAJGRU: ("hko7_hko_weights", "TrajGRU", 79999),
    HKO_CONVGRU: ("hko7_hko_weights", "ConvGRU", 49999),
    MNIST["trajgru_1_64_1_96_1_96_L5"]: ("hko7_movingmnist_weights", "TrajGRU-L5", 199999),
    MNIST["trajgru_1_64_1_96_1_96_L9"]: ("hko7_movingmnist_weights", "TrajGRU-L9", 199999),
    MNIST["trajgru_1_64_1_96_1_96_L13"]: ("hko7_movingmnist_weights", "TrajGRU-L13", 199999),
    MNIST["trajgru_1_64_1_96_1_96_L17"]: ("hko7_movingmnist_weights", "TrajGRU-L17", 199999),
    MNIST["convgru_1_64_1_96_1_96_K3D2"]: ("hko7_movingmnist_weights", "ConvGRU-K3D2", 199999),
    MNIST["convgru_1_64_1_96_1_96_K5"]: ("hko7_movingmnist_weights", "ConvGRU-K5", 199999),
    MNIST["convgru_1_64_1_96_1_96_K7"]: ("hko7_movingmnist_weights", "ConvGRU-K7", 199999),
}
# Parameter counts of the reference code for each configuration file.
PARAMETER_COUNTS = {
    HKO_TRAJGRU: 12_150_053,
    HKO_CONVGRU: 13_649_081,
    MNIST["trajgru_1_64_1_96_1_96_L5"]: 2_842_157,
    MNIST["trajgru_1_64_1_96_1_96_L9"]: 3_421_277,
    MNIST["trajgru_1_64_1_96_1_96_L13"]: 4_000_397,
    MNIST["trajgru_1_64_1_96_1_96_L17"]: 4_579_517,
    MNIST["convgru_1_64_1_96_1_96_K3D2"]: 2_604_817,
    MNIST["convgru_1_64_1_96_1_96_K5"]: 4_767_505,
    MNIST["convgru_1_64_1_96_1_96_K7"]: 8_011_537,
}


def _is_hko(config_file: str) -> bool:
    return config_file.startswith(HKO_DIR)


@contextlib.contextmanager
def hko7(config_file: str, batch: int, in_len: int, out_len: int, rnn_overrides=None, build: bool = True):
    """Official encoder/forecaster networks for one configuration file (fresh ``nowcasting`` import)."""
    repo = oracle_repo("HKO-7")
    mx = oracle_package("mxnet", "1.9.1", "requirements-trajgru.txt")
    # nowcasting/config.py refuses to import unless the HKO-7 radar folders exist; the network
    # code never reads them.
    for folder in ("radarPNG", "radarPNG_mask"):
        (repo / "hko_data" / folder).mkdir(parents=True, exist_ok=True)
    paths = [str(repo), str(repo / "experiments" / "movingmnist")]
    before = set(sys.modules)
    sys.path[:0] = paths
    try:
        config = importlib.import_module("nowcasting.config")
        edict = importlib.import_module("nowcasting.helpers.ordered_easydict").OrderedEasyDict
        user = edict(yaml.safe_load((repo / config_file).read_text()))
        # hko_main.py merges into cfg.MODEL, mnist_rnn_main.py into cfg (cfg_from_file calls yaml.load
        # without a Loader, which PyYAML 6 rejects, so its merge step is called directly).
        config._merge_two_config(user, config.cfg.MODEL if _is_hko(config_file) else config.cfg)
        cfg = config.cfg
        if _is_hko(config_file):
            cfg.MOVINGMNIST.IMG_SIZE = 480  # HKO-7 radar maps are 480 x 480
            cfg.MODEL.ENCODER_FORECASTER.HAS_MASK = False  # only shapes the loss network
        for key, value in (rnn_overrides or {}).items():
            cfg.MODEL.ENCODER_FORECASTER.RNN_BLOCKS[key] = value
        factory_module = importlib.import_module("mnist_rnn_factory")
        ef = importlib.import_module("nowcasting.encoder_forecaster")
        encoder = forecaster = None
        with contextlib.redirect_stdout(io.StringIO()):
            factory = factory_module.MovingMNISTFactory(batch_size=batch, in_seq_len=in_len, out_seq_len=out_len)
            if build:
                encoder, forecaster, _ = ef.encoder_forecaster_build_networks(factory, [mx.cpu()])
        yield SimpleNamespace(mx=mx, cfg=cfg, ef=ef, factory=factory, encoder=encoder, forecaster=forecaster)
    finally:
        for path in paths:
            sys.path.remove(path)
        for name in set(sys.modules) - before:
            if name.split(".")[0] in {"nowcasting", "mnist_rnn_factory"}:
                del sys.modules[name]


def _port_kwargs(cfg) -> dict:
    """PyHazards builder arguments for the architecture fields of an HKO-7 configuration."""
    ef, rnn = cfg.MODEL.ENCODER_FORECASTER, cfg.MODEL.ENCODER_FORECASTER.RNN_BLOCKS
    tup = lambda rows: tuple(tuple(int(v) for v in row) for row in rows)  # noqa: E731
    return dict(
        num_filter=tuple(int(v) for v in rnn.NUM_FILTER),
        layer_type=tuple(rnn.LAYER_TYPE),
        L=tuple(int(v) for v in rnn.L),
        h2h_kernel=tup(rnn.H2H_KERNEL),
        h2h_dilate=tup(rnn.H2H_DILATE),
        i2h_kernel=tup(rnn.I2H_KERNEL),
        i2h_pad=tup(rnn.I2H_PAD),
        stack_num=tuple(int(v) for v in rnn.STACK_NUM),
        first_conv=tuple(int(v) for v in ef.FIRST_CONV),
        last_deconv=tuple(int(v) for v in ef.LAST_DECONV),
        downsample=tup(ef.DOWNSAMPLE),
        upsample=tup(ef.UPSAMPLE),
        rnn_act_type=cfg.MODEL.RNN_ACT_TYPE,
        cnn_act_type=cfg.MODEL.CNN_ACT_TYPE,
        residual_connection=bool(rnn.RES_CONNECTION),
        init_grid=bool(cfg.MODEL.TRAJRNN.INIT_GRID),
    )


def _port(cfg, num_output_frames: int) -> TrajGRU:
    assert cfg.MODEL.OUT_TYPE == "direct" and cfg.MODEL.FRAME_STACK == 1
    return TrajGRU(num_output_frames=num_output_frames, **_port_kwargs(cfg))


def _reference_params(ref) -> dict:
    params = {}
    for net in (ref.encoder, ref.forecaster):
        params.update(net.get_params()[0])
    return params


def _predict(ref, data: np.ndarray, states=None):
    """``mnist_get_prediction``: data (T, B, 1, H, W) -> prediction (T_out, B, 1, H, W), states."""
    mx = ref.mx
    if states is None:
        states = ref.ef.EncoderForecasterStates(factory=ref.factory, ctx=mx.cpu())
        states.reset_all()
    ref.encoder.forward(is_train=False, data_batch=mx.io.DataBatch(data=[mx.nd.array(data)] + states.get_encoder_states()))
    states.update(ref.encoder.get_outputs())
    ref.forecaster.forward(is_train=False, data_batch=mx.io.DataBatch(data=states.get_forecaster_state()))
    return ref.forecaster.get_outputs()[0].asnumpy(), states


def _port_predict(port: TrajGRU, data: np.ndarray, **kwargs):
    with torch.no_grad():
        out = port(torch.from_numpy(data).transpose(0, 1), **kwargs)
    if isinstance(out, tuple):
        return out[0].transpose(0, 1).numpy(), out[1]
    return out.transpose(0, 1).numpy()


def _assert_close(actual, expected) -> None:
    torch.testing.assert_close(torch.as_tensor(actual), torch.as_tensor(expected), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("config_file", list(PARAMETER_COUNTS))
def test_parameter_names_shapes_and_counts_match_reference(config_file):
    with hko7(config_file, batch=1, in_len=2, out_len=2) as ref:
        params = _reference_params(ref)
        port = _port(ref.cfg, 2)
    state = {key.replace(".", "_"): tuple(value.shape) for key, value in port.state_dict().items()}
    assert state == {name: tuple(value.shape) for name, value in params.items()}
    count = sum(p.numel() for p in port.parameters())
    assert count == sum(int(np.prod(value.shape)) for value in params.values()) == PARAMETER_COUNTS[config_file]


def test_presets_and_paper_table_1():
    with hko7(HKO_TRAJGRU, batch=1, in_len=1, out_len=1) as ref:
        hko = _port_kwargs(ref.cfg)
    with hko7(MNIST["trajgru_1_64_1_96_1_96_L13"], batch=1, in_len=1, out_len=1) as ref:
        mnist = _port_kwargs(ref.cfg)
    for preset, expected in ((TRAJGRU_CONFIGS["hko7"], hko), (TRAJGRU_CONFIGS["movingmnist"], mnist)):
        assert expected.pop("layer_type") == ("TrajGRU",) * 3
        for key in ("rnn_act_type", "cnn_act_type"):
            assert expected.pop(key) == "leaky"
        assert expected.pop("residual_connection") and expected.pop("init_grid")
        assert {key: preset[key] for key in expected} == expected
    assert TRAJGRU_CONFIGS["hko7"]["num_output_frames"] == 20 and TRAJGRU_CONFIGS["movingmnist"]["num_output_frames"] == 10

    # Paper Table 1 (MovingMNIST++, millions of parameters). Traj-L9, Traj-L13, Conv-K5-D1 and
    # Conv-K7-D1 agree with the code; the code gives Conv-K3-D2 2.60M and Traj-L5 2.84M, the
    # paper prints these two values swapped, and Traj-L17 has 4.58M, not the 4.77M of the table.
    table = {"trajgru_1_64_1_96_1_96_L9": 3.42, "trajgru_1_64_1_96_1_96_L13": 4.00,
             "convgru_1_64_1_96_1_96_K5": 4.77, "convgru_1_64_1_96_1_96_K7": 8.01}
    for name, millions in table.items():
        assert round(PARAMETER_COUNTS[MNIST[name]] / 1e6, 2) == millions
    assert round(PARAMETER_COUNTS[MNIST["convgru_1_64_1_96_1_96_K3D2"]] / 1e6, 2) == 2.60
    assert round(PARAMETER_COUNTS[MNIST["trajgru_1_64_1_96_1_96_L5"]] / 1e6, 2) == 2.84
    # The builder presets are the official configurations.
    assert sum(p.numel() for p in build_model("trajgru", task="forecasting").parameters()) == 12_150_053
    assert sum(p.numel() for p in build_model("trajgru", task="forecasting", config="movingmnist").parameters()) == 4_000_397


@pytest.mark.parametrize("config_file", list(CHECKPOINTS))
def test_official_weights_match_reference(config_file):
    asset, folder, iteration = CHECKPOINTS[config_file]
    directory = oracle_asset(asset) / folder
    hko = _is_hko(config_file)
    batch, in_len, out_len, size = (1, 5, 20, 480) if hko else (2, 10, 10, 64)
    with hko7(config_file, batch=batch, in_len=in_len, out_len=out_len) as ref:
        with contextlib.redirect_stdout(io.StringIO()):
            ref.ef.load_encoder_forecaster_params(str(directory), iteration, ref.encoder, ref.forecaster)
        port = _port(ref.cfg, out_len)
        load_hko7_params(port, _reference_params(ref))
        port.eval()
        data = np.random.RandomState(0).rand(in_len, batch, 1, size, size).astype(np.float32)
        expected, _ = _predict(ref, data)
    assert expected.shape == (out_len, batch, 1, size, size)
    _assert_close(_port_predict(port, data), expected)


def _randomise_flow_layers(ref, nets, seed: int) -> None:
    """Non-zero flow layers (the official initialisation zeroes them, so every warp starts as identity)."""
    params = {}
    for net in nets:
        params.update(net.get_params()[0])
    rng = np.random.RandomState(seed)
    for name in params:
        if "_f_out_" in name:
            scale = 0.05 if name.endswith("weight") else 2.0
            params[name] = ref.mx.nd.array(rng.normal(0.0, scale, params[name].shape))
    for net in nets:
        net.set_params(params, {}, allow_extra=True, force_init=True)


def test_random_flows_and_odd_batch_match_reference():
    with hko7(MNIST["trajgru_1_64_1_96_1_96_L5"], batch=3, in_len=4, out_len=3) as ref:
        _randomise_flow_layers(ref, (ref.encoder, ref.forecaster), seed=1)
        port = _port(ref.cfg, 3)
        load_hko7_params(port, _reference_params(ref))
        data = np.random.RandomState(2).rand(4, 3, 1, 64, 64).astype(np.float32)
        expected, _ = _predict(ref, data)
    _assert_close(_port_predict(port, data), expected)


def test_stacked_encoder_blocks_match_reference():
    # Several cells per block (residual connections between them) appear in no released
    # configuration. With STACK_NUM > 1 the official forecaster network fails to bind on MXNet
    # 1.9.1, so only the encoder (whose final states feed the forecaster) is compared here.
    overrides = {"STACK_NUM": [2, 3, 1]}
    with hko7(MNIST["trajgru_1_64_1_96_1_96_L5"], batch=2, in_len=3, out_len=1, rnn_overrides=overrides, build=False) as ref:
        mx, factory = ref.mx, ref.factory
        my_module = importlib.import_module("nowcasting.my_module")
        with contextlib.redirect_stdout(io.StringIO()):
            encoder = my_module.MyModule(
                factory.encoder_sym(), data_names=[d.name for d in factory.encoder_data_desc()], label_names=[],
                context=[mx.cpu()], name="encoder_net",
            )
            encoder.bind(data_shapes=factory.encoder_data_desc(), label_shapes=None, inputs_need_grad=True)
            encoder.init_params(mx.init.MSRAPrelu(slope=0.2))  # as encoder_forecaster_build_networks
        _randomise_flow_layers(ref, (encoder,), seed=3)
        port = _port(ref.cfg, 1)
        assert [len(getattr(port, f"ebrnn{i}")) for i in (1, 2, 3)] == [2, 3, 1]
        params = encoder.get_params()[0]
        encoder_state = {k: v for k, v in port.state_dict().items() if not k.startswith(("fbrnn", "fup", "fdeconv", "conv_final", "out."))}
        port.load_state_dict(
            {**port.state_dict(), **{k: torch.from_numpy(params[k.replace(".", "_")].asnumpy()) for k in encoder_state}}
        )
        data = np.random.RandomState(4).rand(3, 2, 1, 64, 64).astype(np.float32)
        states = ref.ef.EncoderForecasterStates(factory=factory, ctx=mx.cpu())
        states.reset_all()
        encoder.forward(is_train=False, data_batch=mx.io.DataBatch(data=[mx.nd.array(data)] + states.get_encoder_states()))
        expected = [out.asnumpy() for out in encoder.get_outputs()]  # per block, stacked states concatenated
    with torch.no_grad():
        actual = port.encode(torch.from_numpy(data).transpose(0, 1))
    assert len(actual) == len(expected) == 3
    for block_states, reference in zip(actual, expected):
        _assert_close(torch.cat(block_states, dim=1), reference)


def test_carried_over_encoder_states_match_reference():
    # mnist_rnn_main.py resets the encoder states once and then starts every batch from the
    # previous batch's final states; ``initial_states`` / ``return_states`` reproduce that.
    asset, folder, iteration = CHECKPOINTS[MNIST["trajgru_1_64_1_96_1_96_L13"]]
    with hko7(MNIST["trajgru_1_64_1_96_1_96_L13"], batch=2, in_len=10, out_len=10) as ref:
        with contextlib.redirect_stdout(io.StringIO()):
            ref.ef.load_encoder_forecaster_params(str(oracle_asset(asset) / folder), iteration, ref.encoder, ref.forecaster)
        port = _port(ref.cfg, 10)
        load_hko7_params(port, _reference_params(ref))
        rng = np.random.RandomState(3)
        first, second = (rng.rand(10, 2, 1, 64, 64).astype(np.float32) for _ in range(2))
        _, states = _predict(ref, first)
        expected, _ = _predict(ref, second, states)
    _, port_states = _port_predict(port, first, return_states=True)
    _assert_close(_port_predict(port, second, initial_states=port_states), expected)


def test_initialisation_follows_reference_scheme():
    # MXNet and PyTorch draw different random numbers, so the scheme is compared, not the values:
    # zero tensors must be zero in both, and every other tensor must have the MSRAPrelu(0.2) std.
    with hko7(HKO_TRAJGRU, batch=1, in_len=1, out_len=1) as ref:
        params = {name: value.asnumpy() for name, value in _reference_params(ref).items()}
        torch.manual_seed(0)
        port = _port(ref.cfg, 1)
    for key, tensor in port.state_dict().items():
        name = key.replace(".", "_")
        reference = params[name]
        if not reference.any():
            assert not tensor.any(), name
            continue
        receptive = int(np.prod(reference.shape[2:]))
        expected_std = math.sqrt(2.0 / (1.0 + 0.2 ** 2) / ((reference.shape[0] + reference.shape[1]) * receptive / 2.0))
        if reference.size >= 5000:
            assert abs(reference.std() / expected_std - 1) < 0.1, name
            assert abs(tensor.std().item() / expected_std - 1) < 0.1, name
    assert not port.ebrnn1[0].f_out.weight.any() and not port.fbrnn3[0].f_out.bias.any()


def test_warp_matches_mxnet_grid_generator_and_bilinear_sampler():
    mx = oracle_package("mxnet", "1.9.1", "requirements-trajgru.txt")
    rng = np.random.RandomState(4)
    data = rng.normal(size=(2, 3, 9, 11)).astype(np.float32)
    flows = rng.normal(0.0, 3.0, size=(2, 2 * 4, 9, 11)).astype(np.float32)  # 4 links, many out of range
    expected = []
    for link in range(4):
        grid = mx.nd.GridGenerator(data=-mx.nd.array(flows[:, 2 * link : 2 * link + 2]), transform_type="warp")
        expected.append(mx.nd.BilinearSampler(data=mx.nd.array(data), grid=grid).asnumpy())
    actual = warp(torch.from_numpy(data), torch.from_numpy(flows))
    _assert_close(actual, np.concatenate(expected, axis=1))

