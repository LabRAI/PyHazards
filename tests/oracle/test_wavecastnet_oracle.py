"""WaveCastNet checked against the official PyTorch code and its released checkpoints.

Reference (pinned in repos.yaml): dwlyu/WaveCastNet at c859e04 (MIT). ``AEConvLEM_dense`` and
``AEConvLEM_sparse`` are imported from ``src/models_earthquake`` with the README / notebook arguments
(``dt=1, num_channels=3, num_kernels=144, kernel_size=(3, 3), padding=(1, 1), activation="tanh",
frame_size=(43, 28)``; sparse: ``mask_mode=1, mask_ratio=1 - 101/564``). ``Encoder1d`` reads the station
list from a hard-coded NERSC path, so ``numpy.load`` is redirected to the repository's
``filtered_coord.npy`` (or to a list of the checkpoint's size) while the official modules run.

Checks: parameter counts (10,093,242 dense, 16,535,430 sparse), parameter names and shapes, identical
seeded initialisation, identical outputs in evaluation and training mode (same global seed for the
decoder noise and the station mask, and the port's generator hook), batch-norm statistics, gradients,
the rolling validation forecast of ``earthquake_train.py``, the released ``best_lem_dense_.pt`` (strict
load, same outputs) and ``best_lem_irr_mask_shakealert_.pt`` (strict load with 558 stations, same
outputs), the training ``Huber`` loss and the ``Validation_pixel`` metrics.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch
import torch.nn as nn

from oracle_utils import import_from, load_definitions, oracle_asset, oracle_repo
from pyhazards.metrics.wavefield import wavefield_acc, wavefield_metrics, wavefield_rfne, wavefield_rmse
from pyhazards.models import build_model
from pyhazards.models.wavecastnet import (
    DEFAULT_MASK_RATIO,
    WAVECASTNET_STATIONS,
    WAVECASTNET_WEIGHTS,
    ConvLEMCell,
    WaveCastNet,
    WaveCastNetLoss,
    WaveCastNetSparse,
    load_wavecastnet_checkpoint,
)

OFFICIAL = dict(
    dt=1, num_channels=3, num_kernels=144, kernel_size=(3, 3), padding=(1, 1), activation="tanh", frame_size=(43, 28)
)
EXACT = dict(rtol=0, atol=0)
# The checkpoint comparisons run the same float32 operations in the same order; allow summation noise.
TOLERANCE = dict(rtol=1e-5, atol=1e-6)
NERSC = "/global/homes/d/dwlyu/earthquake/"


@pytest.fixture(scope="module")
def source():
    return oracle_repo("WaveCastNet") / "src" / "models_earthquake"


@pytest.fixture(scope="module")
def official(source):
    """The official model classes (``ConvLEMCell`` module objects included)."""
    modules = {name: import_from(source, name) for name in ("ConvLEMCell", "AEConvLEM_dense", "AEConvLEM_sparse")}
    return modules


@pytest.fixture
def stations(source, monkeypatch):
    """Redirect the official ``np.load`` of the NERSC station list to ``coords`` (default: the repo file)."""
    original = np.load
    state = {"coords": original(source / "filtered_coord.npy")}

    def load(path, *args, **kwargs):
        if isinstance(path, str) and path.startswith(NERSC):
            return state["coords"]
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", load)
    return state


def _count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _assert_same_state(reference: nn.Module, port: nn.Module) -> None:
    ref, ours = reference.state_dict(), port.state_dict()
    assert list(ref) == list(ours)
    for key in ref:
        assert ref[key].shape == ours[key].shape, key
        torch.testing.assert_close(ours[key], ref[key], **EXACT, msg=key)


def _load_official_state(path, model: nn.Module) -> None:
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=True)


def test_parameter_counts_names_and_seeded_initialisation(official, stations):
    torch.manual_seed(0)
    dense_ref = official["AEConvLEM_dense"].AEConvLEM_dense(**OFFICIAL)
    torch.manual_seed(0)
    dense = WaveCastNet()
    assert _count(dense_ref) == _count(dense) == 10_093_242
    _assert_same_state(dense_ref, dense)
    assert dense.encoder_1_convlem.W_z4.shape == (144, 43, 28)

    torch.manual_seed(3)
    sparse_ref = official["AEConvLEM_sparse"].AEConvLEM_sparse(mask_mode=1, mask_ratio=DEFAULT_MASK_RATIO, **OFFICIAL)
    torch.manual_seed(3)
    sparse = WaveCastNetSparse(station_coords=stations["coords"])
    assert _count(sparse_ref) == _count(sparse) == 16_535_430
    _assert_same_state(sparse_ref, sparse)

    # Both cell variants of ConvLEMCell.py (ConvLEMCell: no reset gate, xavier-normal).
    for name, reset_gate in (("ConvLEMCell", False), ("ConvLEMCell_1", True)):
        torch.manual_seed(5)
        cell_ref = getattr(official["ConvLEMCell"], name)(
            dt=0.5, in_channels=6, out_channels=8, kernel_size=(3, 3), padding=(1, 1), activation="relu", frame_size=(5, 4)
        )
        torch.manual_seed(5)
        cell = ConvLEMCell(6, 8, (5, 4), kernel_size=3, padding=1, dt=0.5, activation="relu", reset_gate=reset_gate)
        _assert_same_state(cell_ref, cell)
        x, h, c = torch.randn(2, 6, 5, 4), torch.randn(2, 8, 5, 4), torch.randn(2, 8, 5, 4)
        for ours, ref in zip(cell(x, h, c), cell_ref(x, h, c)):
            torch.testing.assert_close(ours, ref, **EXACT)


@pytest.mark.parametrize("future_seq", [1, 3])
def test_dense_outputs_match_in_eval_and_train_mode(official, future_seq):
    torch.manual_seed(11)
    reference = official["AEConvLEM_dense"].AEConvLEM_dense(**OFFICIAL)
    port = WaveCastNet()
    port.load_state_dict(reference.state_dict())
    x = torch.randn(2, 3, 3, 344, 224)

    reference.eval(), port.eval()
    with torch.no_grad():
        torch.manual_seed(1)
        expected = reference(x, future_seq=future_seq)
        torch.manual_seed(1)
        torch.testing.assert_close(port(x, future_seq), expected, **EXACT)
        # Generator hook: the same draws as the official global generator.
        torch.manual_seed(99)
        torch.testing.assert_close(port(x, future_seq, generator=torch.Generator().manual_seed(1)), expected, **EXACT)
        # decoder_noise hook: the official uniform noise drawn explicitly.
        torch.manual_seed(1)
        noise = torch.rand(2, 144, 43, 28)
        torch.testing.assert_close(port(x, future_seq, decoder_noise=noise), expected, **EXACT)
        # The decoder starts from noise also in eval mode, so outputs depend on the seed.
        torch.manual_seed(2)
        assert not torch.equal(reference(x, future_seq=future_seq), expected)

    reference.train(), port.train()
    torch.manual_seed(4)
    expected = reference(x, future_seq=future_seq)
    torch.manual_seed(4)
    actual = port(x, future_seq)
    torch.testing.assert_close(actual, expected, **EXACT)
    _assert_same_state(reference, port)  # batch-norm running statistics
    grad_out = torch.randn_like(expected)
    expected.backward(grad_out)
    actual.backward(grad_out)
    for (name, p_ref), p_port in zip(reference.named_parameters(), port.parameters()):
        torch.testing.assert_close(p_port.grad, p_ref.grad, rtol=1e-5, atol=1e-7, msg=name)


def test_sparse_outputs_match(official, stations):
    torch.manual_seed(12)
    reference = official["AEConvLEM_sparse"].AEConvLEM_sparse(mask_mode=1, mask_ratio=DEFAULT_MASK_RATIO, **OFFICIAL)
    port = WaveCastNetSparse(station_coords=stations["coords"])
    port.load_state_dict(reference.state_dict())
    x = torch.randn(2, 3, 3, 344, 224)

    reference.eval(), port.eval()
    with torch.no_grad():
        torch.manual_seed(6)
        expected = reference(x, future_seq=2)
        torch.manual_seed(6)
        torch.testing.assert_close(port(x, 2), expected, **EXACT)
        torch.testing.assert_close(port(x, 2, generator=torch.Generator().manual_seed(6)), expected, **EXACT)
        # Official draw order: the station mask (one uniform per sample and station, masked if < ratio),
        # then the decoder noise; the station_mask and decoder_noise hooks take them explicitly.
        torch.manual_seed(6)
        draw, noise = torch.rand(2, 564), torch.rand(2, 144, 43, 28)
        kept = port(x, 2, station_mask=draw >= DEFAULT_MASK_RATIO, decoder_noise=noise)
        torch.testing.assert_close(kept, expected, **EXACT)

    reference.train(), port.train()
    torch.manual_seed(7)
    expected = reference(x, future_seq=2)
    torch.manual_seed(7)
    torch.testing.assert_close(port(x, 2), expected, **EXACT)
    _assert_same_state(reference, port)  # batch-norm running statistics


def test_rolling_validation_forecast_matches_earthquake_train(official):
    """``earthquake_train.py`` validation: six calls of ``step`` frames and one of ``step // 2``, each on the
    previous output; here with step 2 (13 frames) instead of 30 (195 frames) to keep the test fast."""
    torch.manual_seed(13)
    reference = official["AEConvLEM_dense"].AEConvLEM_dense(**OFFICIAL).eval()
    port = WaveCastNet(future_seq=2).eval()
    port.load_state_dict(reference.state_dict())
    x, step = torch.randn(1, 3, 2, 344, 224), 2
    with torch.no_grad():
        torch.manual_seed(8)
        output, current = [], x
        for i in range(7):
            current = reference(current, future_seq=step if i < 6 else step // 2).detach()
            output.append(current)
        expected = torch.concat(output, dim=2)
        torch.manual_seed(8)
        actual = port.rollout(x, 13)
    assert actual.shape == (1, 3, 13, 344, 224)
    torch.testing.assert_close(actual, expected, **EXACT)


def test_released_dense_checkpoint(official):
    path = oracle_asset("wavecastnet_dense_checkpoint") / WAVECASTNET_WEIGHTS["dense"]["filename"]
    reference = official["AEConvLEM_dense"].AEConvLEM_dense(**OFFICIAL)
    _load_official_state(path, reference)
    port = load_wavecastnet_checkpoint(WaveCastNet(), path)
    built = build_model("wavecastnet", task="forecasting", pretrained=str(path))
    _assert_same_state(reference, port)
    _assert_same_state(reference, built)
    reference.eval(), port.eval()
    x = torch.randn(1, 3, 4, 344, 224)
    with torch.no_grad():
        torch.manual_seed(21)
        expected = reference(x, future_seq=3)
        torch.manual_seed(21)
        actual = port(x, 3)
    torch.testing.assert_close(actual, expected, **TOLERANCE)
    assert torch.isfinite(actual).all() and actual.abs().max() > 0


def test_released_sparse_checkpoint(official, stations, source):
    path = oracle_asset("wavecastnet_sparse_checkpoint") / "best_lem_irr_mask_shakealert_.pt"
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    assert state["module.encoder.FC1.0.weight"].shape == (1204, 558)
    # The released station list has 564 entries: the checkpoint's 558-station list was not published.
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_wavecastnet_checkpoint(WaveCastNetSparse(station_coords=stations["coords"]), path)
    coords = stations["coords"][:558]  # any 558 grid points: names, shapes and arithmetic are what is checked
    stations["coords"] = coords
    reference = official["AEConvLEM_sparse"].AEConvLEM_sparse(mask_mode=1, mask_ratio=DEFAULT_MASK_RATIO, **OFFICIAL)
    _load_official_state(path, reference)
    port = load_wavecastnet_checkpoint(WaveCastNetSparse(station_coords=coords), path)
    _assert_same_state(reference, port)
    reference.eval(), port.eval()
    x = torch.randn(1, 3, 3, 344, 224)
    with torch.no_grad():
        torch.manual_seed(22)
        expected = reference(x, future_seq=2)
        torch.manual_seed(22)
        actual = port(x, 2)
    torch.testing.assert_close(actual, expected, **TOLERANCE)


def test_station_files_match_the_pinned_hashes(source):
    for spec in WAVECASTNET_STATIONS.values():
        assert hashlib.sha256((source / spec["filename"]).read_bytes()).hexdigest() == spec["sha256"]
    assert len(np.load(source / "filtered_coord.npy")) == 564
    assert len(np.load(source / "shakealert_coords.npy")) == 101


def test_loss_and_metrics_match_the_official_functions(source):
    train_script = source.parent.parent / "earthquake_train.py"
    huber = load_definitions(train_script, ["Huber"], {"torch": torch, "nn": nn})["Huber"](reduction="mean")
    validation = import_from(source, "Validation_pixel")
    torch.manual_seed(0)
    pred, target = torch.randn(3, 3, 5, 16, 8), torch.randn(3, 3, 5, 16, 8) * 0.3
    loss = WaveCastNetLoss()
    torch.testing.assert_close(loss(pred, target), huber(pred.flatten(), target.flatten()), **EXACT)
    torch.testing.assert_close(
        WaveCastNetLoss(delta=0.7)(pred, target), huber(pred.flatten(), target.flatten(), delta=0.7), **EXACT
    )
    expected = {
        "acc": validation.validation_acc(pred, target),
        "rfne": validation.validation_rfne(pred, target),
        "rmse": validation.validation_rmse(pred, target),
    }
    metrics = wavefield_metrics(pred, target)
    for key, value in expected.items():
        assert metrics[key] == pytest.approx(value, rel=1e-6)
    for fn, official_fn in ((wavefield_acc, validation.validation_acc), (wavefield_rfne, validation.validation_rfne),
                            (wavefield_rmse, validation.validation_rmse)):
        for channel in range(3):
            assert float(fn(pred, target)[:, channel].mean()) == pytest.approx(
                official_fn(pred[:, channel : channel + 1], target[:, channel : channel + 1]), rel=1e-6
            )
