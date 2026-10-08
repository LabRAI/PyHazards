"""GPD (Ross et al. 2018): shapes, parameter count, picking logic and the Keras weight loader (offline)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from pyhazards.models import build_model
from pyhazards.models.gpd import GPD, KerasBatchNorm1d, load_gpd_keras_weights


def _keras_file(model: GPD, path) -> None:
    """Write ``model``'s weights in the layout of the released ``model_pol_best.hdf5``."""
    with h5py.File(path, "w") as handle:
        group = handle.create_group("model_weights").create_group("sequential_1")
        for name, module in model.named_children():
            layer = group.create_group(f"{name}_1")
            if isinstance(module, torch.nn.Conv1d):
                layer["kernel:0"] = module.weight.detach().permute(2, 1, 0).numpy()
                layer["bias:0"] = module.bias.detach().numpy()
            elif isinstance(module, torch.nn.Linear):
                layer["kernel:0"] = module.weight.detach().t().numpy()
                layer["bias:0"] = module.bias.detach().numpy()
            else:
                layer["gamma:0"] = module.weight.detach().numpy()
                layer["beta:0"] = module.bias.detach().numpy()
                layer["moving_mean:0"] = module.running_mean.numpy()
                layer["moving_variance:0"] = module.running_var.numpy()


def test_parameter_count_and_output_shape():
    model = build_model("gpd", task="classification").eval()
    assert sum(p.numel() for p in model.parameters()) == 1_741_003
    probabilities = model(torch.randn(2, 3, 400))
    assert probabilities.shape == (2, 3)
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2))
    logits = model(torch.randn(2, 3, 400), logits=True)
    assert logits.shape == (2, 3)
    assert model.output_names == ("P", "S", "noise")
    assert model.component_order == "NEZ"


def test_shape_validation_and_builder_arguments():
    model = GPD().eval()
    for shape in [(2, 3, 300), (2, 2, 400), (3, 400)]:
        with pytest.raises(ValueError, match="shape"):
            model(torch.randn(*shape))
    with pytest.raises(ValueError):
        build_model("gpd", task="regression")
    with pytest.raises(TypeError):
        build_model("gpd", task="classification", hidden_dim=32)
    with pytest.raises(ValueError):
        build_model("gpd", task="picking", in_channels=2, pretrained="original")
    assert isinstance(build_model("gpd", task="picking"), GPD)


def test_keras_initialisation_bounds():
    torch.manual_seed(0)
    model = GPD()
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
            weight = module.weight
            receptive = weight[0, 0].numel() if weight.ndim == 3 else 1
            limit = (6.0 / ((weight.shape[0] + weight.shape[1]) * receptive)) ** 0.5
            assert weight.abs().max() <= limit
            assert torch.count_nonzero(module.bias) == 0


def test_annotate_windows_and_zero_windows():
    model = GPD().eval()
    waveforms = torch.randn(2, 3, 1000)
    waveforms[1, :, :500] = 0.0  # the first windows of trace 1 are all zero
    with torch.no_grad():
        annotations = model.annotate(waveforms, stride=10)
    assert annotations.shape == (2, 3, 61)  # (1000 - 400) / 10 + 1 windows
    assert torch.isfinite(annotations).all()
    windows = model.sliding_windows(waveforms, stride=10)
    assert windows.shape == (2, 61, 3, 400)
    torch.testing.assert_close(windows[0, 3], waveforms[0, :, 30:430] / waveforms[0, :, 30:430].abs().max())
    with pytest.raises(ValueError):
        model.annotate(torch.randn(1, 3, 399))


def test_extract_picks_follows_gpd_predict():
    model = GPD()
    n = 50
    annotations = np.zeros((1, 3, n), dtype=np.float32)
    annotations[0, 2] = 1.0
    # P: a trigger over windows 10..13 (peak at 12) and a one-window trigger at 30 (skipped).
    annotations[0, 0, 10:14] = [0.96, 0.97, 0.99, 0.5]
    annotations[0, 0, 14] = 0.05
    annotations[0, 0, 30] = 0.99
    # S: the peak is the last window of the run, which the official slice prob[on:off] excludes.
    annotations[0, 1, 20:23] = [0.96, 0.97, 0.99]
    picks = model.extract_picks(torch.from_numpy(annotations), stride=10)
    assert picks == [{"P": [(12 * 10 + 200.0, pytest.approx(0.99))], "S": [(21 * 10 + 200.0, pytest.approx(0.97))]}]
    # A stricter threshold (the paper's 0.98) keeps only the higher peaks.
    strict = model.extract_picks(torch.from_numpy(annotations), min_proba=0.98, stride=10)
    assert [s for s, _ in strict[0]["P"]] == [12 * 10 + 200.0]
    assert strict[0]["S"] == []


def test_keras_weight_file_round_trip(tmp_path):
    torch.manual_seed(0)
    source = GPD()
    for module in source.modules():
        if isinstance(module, KerasBatchNorm1d):
            module.running_mean.uniform_(-1, 1)
            module.running_var.uniform_(0.5, 2)
            module.weight.data.uniform_(0.5, 2)
            module.bias.data.uniform_(-1, 1)
    path = tmp_path / "gpd.hdf5"
    _keras_file(source, path)
    target = build_model("gpd", task="classification", pretrained=str(path)).eval()
    source.eval()
    x = torch.randn(4, 3, 400)
    with torch.no_grad():
        torch.testing.assert_close(target(x), source(x), rtol=0, atol=0)

    with h5py.File(path, "a") as handle:
        del handle["model_weights/sequential_1/dense_3_1/bias:0"]
    with pytest.raises(ValueError, match="lacks"):
        load_gpd_keras_weights(GPD(), path)


def test_keras_batchnorm_updates_are_zero_debiased():
    norm = KerasBatchNorm1d(4, zero_debias=True)
    norm.train()
    x1, x2 = torch.randn(16, 4) * 2 + 1, torch.randn(16, 4) - 3
    norm(x1)
    factor = 16 / (16 - 1.001)
    # The first update replaces the statistics by the batch's (zero-debiased moving average).
    torch.testing.assert_close(norm.running_mean, x1.mean(0))
    torch.testing.assert_close(norm.running_var, x1.var(0, unbiased=False) * factor)
    norm(x2)
    biased = 0.99 * 0.01 * x1.mean(0) + 0.01 * x2.mean(0)
    torch.testing.assert_close(norm.running_mean, biased / (1 - 0.99**2))
    assert int(norm.num_batches_tracked) == 2
