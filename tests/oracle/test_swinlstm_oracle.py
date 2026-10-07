"""SwinLSTM checked against the official implementation and its released weights.

Reference (pinned in repos.yaml): SongTang-x/SwinLSTM (MIT), whose repository also holds the
Moving-MNIST SwinLSTM-D weights ``Pretrained/trained_model_state_dict``. It needs timm 0.4.12 and
einops, so this file runs in the ``swin`` suite of the Oracle workflow (requirements-swin.txt).

The rollout is the official one: ``functions.model_forward_multi_layer`` (SwinLSTM-D) and
``functions.model_forward_single_layer`` (SwinLSTM-B) are called on the official models.
``functions.py`` imports ``utils.py``, which needs matplotlib and scikit-image only for metrics and
plots; a stub ``utils`` module stands in for it.
"""

from __future__ import annotations

import sys
import types
from importlib import metadata
from pathlib import Path

import pytest
import torch

from oracle_utils import import_from, missing, oracle_repo
from pyhazards.models import build_model

REQUIREMENTS = Path(__file__).parent / "requirements-swin.txt"
CHECKPOINT = "Pretrained/trained_model_state_dict"
MNIST_D = dict(
    img_size=64, patch_size=2, in_chans=1, embed_dim=128, depths_downsample=[2, 6], depths_upsample=[6, 2],
    num_heads=[4, 8], window_size=4,
)
# configs.py defaults with --model SwinLSTM-B (depths [12], heads [4, 8], drop_path_rate 0.1).
MNIST_B = dict(
    img_size=64, patch_size=2, in_chans=1, embed_dim=128, depths=[12], num_heads=[4, 8], window_size=4,
    drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
)


@pytest.fixture(autouse=True)
def _reference_stack():
    for line in REQUIREMENTS.read_text().splitlines():
        if "==" not in line or line.lstrip().startswith("#"):
            continue
        name, version = line.split("#")[0].strip().split("==")
        if name not in {"timm", "einops"}:
            continue
        try:
            found = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing(f"{name} is not installed; pip install -r tests/oracle/requirements-swin.txt")
        if found != version:
            missing(f"needs {name}=={version}, found {found}")


@pytest.fixture()
def reference(monkeypatch):
    repo = oracle_repo("SwinLSTM")
    # functions.py needs utils.compute_metrics / utils.visualize only in its test loop.
    monkeypatch.setitem(sys.modules, "utils", types.SimpleNamespace(compute_metrics=None, visualize=None))
    return types.SimpleNamespace(
        repo=repo,
        D=import_from(repo, "SwinLSTM_D"),
        B=import_from(repo, "SwinLSTM_B"),
        functions=import_from(repo, "functions"),
    )


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _assert_same_state(reference: torch.nn.Module, port: torch.nn.Module) -> None:
    ref_state, port_state = reference.state_dict(), port.state_dict()
    assert list(ref_state) == list(port_state)
    for key, value in ref_state.items():
        assert torch.equal(value, port_state[key]), key


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def _rollout_d(functions, model, inputs, targets_len):
    return torch.stack(functions.model_forward_multi_layer(model, inputs, targets_len, [2, 6]), dim=1)


def _rollout_b(functions, model, inputs, targets_len, depths):
    return torch.stack(functions.model_forward_single_layer(model, inputs, targets_len, depths), dim=1)


def test_swinlstm_d_official_weights_match_reference(reference):
    state = torch.load(reference.repo / CHECKPOINT, map_location="cpu", weights_only=True)
    official = reference.D.SwinLSTM(**MNIST_D)
    official.load_state_dict(state, strict=True)
    port = build_model("swinlstm", task="forecasting")
    port.load_state_dict(state, strict=True)
    assert _n_params(port) == _n_params(official) == 20_191_969
    assert list(port.state_dict()) == list(state)

    torch.manual_seed(0)
    frames = torch.rand(2, 10, 1, 64, 64)
    official.eval()
    port.eval()
    with torch.no_grad():
        expected = _rollout_d(reference.functions, official, frames, 10)  # 9 warm-up + 10 predicted frames
        _assert_close(port(frames, include_warmup=True), expected)
        _assert_close(port(frames), expected[:, 9:])  # the official test loop keeps outputs[:, T_in - 1:]
        # One reference step (SwinLSTM.forward) with explicit states.
        out, down, up = port.step(frames[:, 0])
        ref_out, ref_down, ref_up = official(frames[:, 0], [None, None], [None, None])
        _assert_close(out, ref_out)
        _assert_close(port.step(frames[:, 1], down, up)[0], official(frames[:, 1], ref_down, ref_up)[0])


def test_swinlstm_d_initialisation_and_train_mode_match_reference(reference):
    torch.manual_seed(0)
    official = reference.D.SwinLSTM(**MNIST_D)
    torch.manual_seed(0)
    port = build_model("swinlstm", task="forecasting")
    _assert_same_state(official, port)

    torch.manual_seed(1)
    frames = torch.rand(2, 4, 1, 64, 64)
    official.train()
    port.train()
    torch.manual_seed(2)  # stochastic depth draws the same random numbers
    expected = _rollout_d(reference.functions, official, frames, 3)
    torch.manual_seed(2)
    _assert_close(port(frames, num_output_frames=3, include_warmup=True), expected)


def test_swinlstm_b_matches_reference(reference):
    torch.manual_seed(0)
    official = reference.B.SwinLSTM(**MNIST_B)
    torch.manual_seed(0)
    port = build_model("swinlstm", task="forecasting", variant="b", depths=[12])
    _assert_same_state(official, port)
    assert _n_params(port) == 2_778_417

    torch.manual_seed(1)
    frames = torch.rand(2, 4, 1, 64, 64)
    official.eval()
    port.eval()
    with torch.no_grad():
        _assert_close(port(frames, num_output_frames=4, include_warmup=True), _rollout_b(reference.functions, official, frames, 4, [12]))
    official.train()
    port.train()
    torch.manual_seed(3)
    expected = _rollout_b(reference.functions, official, frames, 2, [12])
    torch.manual_seed(3)
    _assert_close(port(frames, num_output_frames=2, include_warmup=True), expected)


def test_swinlstm_b_human36m_and_two_cell_configurations(reference):
    # Paper Table 1, Human3.6m: SwinLSTM-B with 12 Swin blocks, patch 2, 128x128x3 frames.
    torch.manual_seed(0)
    official = reference.B.SwinLSTM(**{**MNIST_B, "img_size": 128, "in_chans": 3})
    torch.manual_seed(0)
    port = build_model("swinlstm", task="forecasting", variant="b", depths=[12], img_size=128, in_channels=3)
    _assert_same_state(official, port)
    assert _n_params(port) == _n_params(official) == 2_781_747

    # Two SwinLSTM-B cells with different head counts, and a small two-level SwinLSTM-D.
    small_b = dict(MNIST_B, img_size=32, embed_dim=32, depths=[2, 4], num_heads=[2, 4], drop_path_rate=0.2)
    torch.manual_seed(4)
    official = reference.B.SwinLSTM(**small_b)
    torch.manual_seed(4)
    port = build_model(
        "swinlstm", task="forecasting", variant="b", img_size=32, embed_dim=32, depths=[2, 4], num_heads=[2, 4],
        drop_path_rate=0.2,
    )
    _assert_same_state(official, port)
    frames = torch.rand(3, 3, 1, 32, 32)
    official.train()
    port.train()
    torch.manual_seed(5)
    expected = _rollout_b(reference.functions, official, frames, 2, [2, 4])
    torch.manual_seed(5)
    _assert_close(port(frames, num_output_frames=2, include_warmup=True), expected)

    small_d = dict(MNIST_D, img_size=32, in_chans=2, embed_dim=32, depths_downsample=[2, 2], depths_upsample=[4, 2])
    torch.manual_seed(6)
    official = reference.D.SwinLSTM(**small_d)
    torch.manual_seed(6)
    port = build_model(
        "swinlstm", task="forecasting", img_size=32, in_channels=2, embed_dim=32, depths_downsample=[2, 2],
        depths_upsample=[4, 2],
    )
    _assert_same_state(official, port)
    frames = torch.rand(2, 3, 2, 32, 32)
    official.train()
    port.train()
    torch.manual_seed(7)
    expected = _rollout_d(reference.functions, official, frames, 2)
    torch.manual_seed(7)
    _assert_close(port(frames, num_output_frames=2, include_warmup=True), expected)


def test_swinlstm_segmentation_adaptation_is_the_reference_rollout_without_sigmoid(reference):
    # With one input and one output channel the segmentation head is the official model: its
    # logits are the official one-frame prediction before the final sigmoid.
    state = torch.load(reference.repo / CHECKPOINT, map_location="cpu", weights_only=True)
    official = reference.D.SwinLSTM(**MNIST_D)
    official.load_state_dict(state, strict=True)
    segmenter = build_model("swinlstm", task="segmentation")
    segmenter.load_state_dict(state, strict=True)
    official.eval()
    segmenter.eval()
    torch.manual_seed(0)
    frames = torch.rand(2, 3, 1, 64, 64)
    with torch.no_grad():
        expected = _rollout_d(reference.functions, official, frames, 1)[:, -1]
        logits = segmenter(frames)
    assert logits.shape == (2, 1, 64, 64)
    _assert_close(torch.sigmoid(logits), expected)
