"""TCN checked against the official locuslab/TCN code (pinned in repos.yaml).

The official modules use the deprecated hook-based ``torch.nn.utils.weight_norm`` (``weight_g`` /
``weight_v``); the port uses ``torch.nn.utils.parametrizations.weight_norm``. Parameter names are
compared after :func:`to_official_state_dict`, and official state dicts are loaded into the port
unchanged. Each check builds both models from the same seed and requires identical initial values
and outputs; configurations are the official example scripts' defaults.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from oracle_utils import import_from, oracle_repo
from pyhazards.models import build_model
from pyhazards.models.tcn import TCN, TemporalConvNet, to_official_state_dict


def _official(module: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # torch.nn.utils.weight_norm is deprecated
        return import_from(oracle_repo("TCN"), f"TCN.{module}")


def _build(factory, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return factory(*args, **kwargs)


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _assert_same_state(reference: torch.nn.Module, port: torch.nn.Module) -> None:
    ref_state, port_state = reference.state_dict(), to_official_state_dict(port.state_dict())
    assert list(ref_state) == list(port_state)
    for key, value in ref_state.items():
        assert torch.equal(value, port_state[key]), key


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_adding_problem_model_matches_official():
    official = _official("adding_problem.model")
    # add_test.py defaults: 2 inputs, 1 output, 8 levels x 30 channels, kernel 7, dropout 0.
    torch.manual_seed(0)
    reference = _build(official.TCN, 2, 1, [30] * 8, kernel_size=7, dropout=0.0)
    torch.manual_seed(0)
    port = build_model("tcn", task="regression")
    _assert_same_state(reference, port)
    assert _n_params(port) == _n_params(reference) == 96_001

    torch.manual_seed(1)
    x = torch.randn(4, 2, 400)  # official layout (batch, channels, length), seq_len 400
    reference.eval()
    port.eval()
    with torch.no_grad():
        _assert_close(port(x.transpose(1, 2)), reference(x))

    # Official state dicts load unchanged, and port state dicts load into the official code.
    restored = build_model("tcn", task="regression")
    restored.load_state_dict(reference.state_dict(), strict=True)
    exported = _build(official.TCN, 2, 1, [30] * 8, kernel_size=7, dropout=0.0)
    exported.load_state_dict(to_official_state_dict(port.state_dict()), strict=True)
    restored.eval()
    exported.eval()
    with torch.no_grad():
        _assert_close(restored(x.transpose(1, 2)), reference(x))
        _assert_close(exported(x), port(x.transpose(1, 2)))


def test_official_normal_init_does_not_reach_weight_normed_convs():
    """``init_weights`` writes N(0, 0.01) into the computed weight, which weight norm recomputes."""
    tcn = _official("tcn")
    torch.manual_seed(0)
    reference = _build(tcn.TemporalConvNet, 3, [16, 16], kernel_size=3, dropout=0.0)
    torch.manual_seed(0)
    port = TemporalConvNet(3, [16, 16], kernel_size=3, dropout=0.0)
    _assert_same_state(reference, port)
    block = reference.network[0]
    assert block.conv1.weight_v.std() > 0.1  # PyTorch's default Conv1d init, not N(0, 0.01)
    assert block.downsample.weight.std() < 0.02  # the plain 1x1 conv does get N(0, 0.01)
    with torch.no_grad():
        reference(torch.randn(1, 3, 8))  # the forward pre-hook recomputes conv1.weight from g, v
    effective = block.conv1.weight_g * block.conv1.weight_v / block.conv1.weight_v.norm(dim=(1, 2), keepdim=True)
    _assert_close(block.conv1.weight, effective)
    _assert_close(port.network[0].conv1.weight, effective)


def test_sequential_mnist_model_matches_official_in_train_mode():
    official = _official("mnist_pixel.model")
    # pmnist_test.py defaults: 1 input, 10 classes, 8 levels x 25 channels, kernel 7, dropout 0.05;
    # the head keeps PyTorch's initialisation and the official forward ends with log_softmax.
    torch.manual_seed(0)
    reference = _build(official.TCN, 1, 10, [25] * 8, kernel_size=7, dropout=0.05)
    torch.manual_seed(0)
    port = build_model(
        "tcn",
        task="classification",
        input_dim=1,
        out_dim=10,
        hidden_dim=25,
        num_levels=8,
        kernel_size=7,
        dropout=0.05,
        head_init="default",
    )
    _assert_same_state(reference, port)
    assert _n_params(port) == _n_params(reference) == 66_910

    torch.manual_seed(1)
    x = torch.randn(3, 1, 196)
    reference.eval()
    port.eval()
    with torch.no_grad():
        _assert_close(torch.log_softmax(port(x.transpose(1, 2)), dim=1), reference(x))
    reference.train()
    port.train()
    torch.manual_seed(2)
    expected = reference(x)
    torch.manual_seed(2)
    _assert_close(torch.log_softmax(port(x.transpose(1, 2)), dim=1), expected)  # same dropout masks


@pytest.mark.parametrize("task_dir", ["copy_memory", "poly_music"])
def test_sequence_heads_match_official(task_dir):
    official = _official(f"{task_dir}.model")
    if task_dir == "copy_memory":
        # copymem_test.py: 1 input, 10 classes, 8 levels x 10 channels, kernel 8; input (N, C, L).
        args, kwargs, head_init = (1, 10, [10] * 8), dict(kernel_size=8, dropout=0.0), "normal"
        x = torch.randn(2, 1, 30)
    else:
        # music_test.py: 88 keys in and out, 4 levels x 150 channels, kernel 5, dropout 0.25;
        # input (N, L, C), output sigmoid in float64.
        args, kwargs, head_init = (88, 88, [150] * 4), dict(kernel_size=5, dropout=0.25), "default"
        x = torch.randn(2, 40, 88)
    torch.manual_seed(0)
    reference = _build(official.TCN, *args, **kwargs)
    torch.manual_seed(0)
    port = TCN(*args, **kwargs, readout="sequence", head_init=head_init)
    _assert_same_state(reference, port)

    reference.eval()
    port.eval()
    with torch.no_grad():
        if task_dir == "copy_memory":
            _assert_close(port(x.transpose(1, 2)), reference(x))
        else:
            _assert_close(torch.sigmoid(port(x).double()), reference(x))


def test_official_and_port_are_causal():
    tcn = _official("tcn")
    torch.manual_seed(0)
    reference = _build(tcn.TemporalConvNet, 2, [8, 8, 8], kernel_size=3, dropout=0.0).eval()
    port = TemporalConvNet(2, [8, 8, 8], kernel_size=3, dropout=0.0).eval()
    port.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(1, 2, 40)
    t = 25
    perturbed = x.clone()
    perturbed[:, :, t + 1 :] += torch.randn_like(perturbed[:, :, t + 1 :])
    with torch.no_grad():
        for model in (reference, port):
            base, moved = model(x), model(perturbed)
            torch.testing.assert_close(base[:, :, : t + 1], moved[:, :, : t + 1], rtol=0, atol=0)
            assert not torch.allclose(base[:, :, t + 1 :], moved[:, :, t + 1 :])
        _assert_close(port(x), reference(x))
