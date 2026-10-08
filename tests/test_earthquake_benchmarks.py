import math

import pytest
import torch
import torch.nn as nn

from pyhazards.benchmarks import run_benchmark
from pyhazards.configs import load_experiment_config
from pyhazards.datasets import load_dataset
from pyhazards.datasets.base import DataBundle, DataSplit, FeatureSpec, LabelSpec
from pyhazards.engine.runner import BenchmarkRunner
from pyhazards.models import build_model


def test_earthquake_vertical_slice(tmp_path):
    config = load_experiment_config("pyhazards/configs/earthquake/phasenet_smoke.yaml")
    summary = BenchmarkRunner().run(config, output_dir=str(tmp_path))

    assert summary.benchmark_name == "earthquake"
    assert summary.hazard_task == "earthquake.picking"
    for key in ("p_precision", "p_recall", "p_f1", "s_f1", "p_pick_mae", "p_residual_mean", "s_residual_std"):
        assert key in summary.metrics
    assert "json" in summary.report_paths and "pick_counts" in summary.report_paths
    assert summary.metadata["protocol"] == "phasenet"


def test_earthquake_forecasting_reports_the_wavecastnet_metrics(tmp_path):
    config = load_experiment_config("pyhazards/configs/earthquake/wavecastnet_benchmark_smoke.yaml")
    summary = BenchmarkRunner().run(config, output_dir=str(tmp_path))

    assert summary.hazard_task == "earthquake.forecasting"
    expected = {"acc", "rfne", "rmse", "mae", "mse"} | {f"{m}_{c}" for m in ("acc", "rfne") for c in "xyz"}
    assert set(summary.metrics) == expected
    assert "pycsep" not in summary.report_paths  # the mislabelled "pyCSEP-style" export was removed
    # WaveCastNet's decoder starts from noise; the benchmark draws it from a generator seeded by `seed`.
    model = build_model(config.model.name, task=config.model.task, **config.model.params)
    first = BenchmarkRunner().run(config, model=model, output_dir=str(tmp_path))
    assert BenchmarkRunner().run(config, model=model, output_dir=str(tmp_path)).metrics == first.metrics
    config.benchmark.params = {**config.benchmark.params, "seed": 1}
    assert BenchmarkRunner().run(config, model=model, output_dir=str(tmp_path)).metrics != first.metrics


class _Persistence(nn.Module):
    """Repeats the last input frame (a forecast without ``rollout``)."""

    def __init__(self, steps):
        super().__init__()
        self.steps = steps

    def forward(self, x):
        return x[:, :, -1:].expand(-1, -1, self.steps, -1, -1)


def _wavefield_bundle(targets, inputs=None):
    inputs = targets[:, :, :1] if inputs is None else inputs
    split = DataSplit(inputs, targets)
    return DataBundle(
        splits={"train": split, "val": split, "test": split},
        feature_spec=FeatureSpec(channels=targets.shape[1]),
        label_spec=LabelSpec(num_targets=1, task_type="regression"),
        metadata={"dataset": "fixture", "channel_names": ["e", "n"]},
    )


def test_forecasting_metrics_follow_the_official_definitions(tmp_path):
    config = load_experiment_config("pyhazards/configs/earthquake/wavecastnet_benchmark_smoke.yaml")
    targets = torch.randn(3, 2, 4, 4, 4)
    perfect = _wavefield_bundle(targets, inputs=targets[:, :, -1:].clone())
    constant = targets[:, :, -1:].expand(-1, -1, 4, -1, -1).clone()
    summary = run_benchmark("earthquake", _Persistence(4), _wavefield_bundle(constant, constant[:, :, :1]), config, output_dir=str(tmp_path))
    assert summary.metrics["acc"] == pytest.approx(1.0) and summary.metrics["rfne"] == pytest.approx(0.0, abs=1e-6)
    assert {"acc_e", "rfne_n"} <= set(summary.metrics)
    flipped = _wavefield_bundle(-constant, constant[:, :, :1])
    summary = run_benchmark("earthquake", _Persistence(4), flipped, config, output_dir=str(tmp_path))
    assert summary.metrics["acc"] == pytest.approx(-1.0) and summary.metrics["rfne"] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="forecast has shape"):
        run_benchmark("earthquake", _Persistence(2), perfect, config, output_dir=str(tmp_path))


class _OraclePicker(nn.Module):
    """Returns Gaussian probability peaks at the true arrivals shifted by ``shift`` samples."""

    sampling_rate = 100.0
    component_order = "ENZ"
    phase_channels = {"P": 1, "S": 2}

    def __init__(self, targets, shift=0.0, with_detection=False):
        super().__init__()
        self.targets = targets
        self.shift = shift
        self.seen = []
        if with_detection:
            self.extract_detections = self._detections

    def forward(self, x):
        self.seen.append(x)
        n, _, length = x.shape
        index = torch.arange(length, dtype=torch.float32)
        rows = self.targets[len(torch.cat(self.seen)) - n : len(torch.cat(self.seen))]
        out = torch.zeros(n, 3, length)
        for i, (p, s) in enumerate(rows.tolist()):
            for channel, arrival in ((1, p), (2, s)):
                if not math.isnan(arrival):
                    out[i, channel] = 0.9 * torch.exp(-0.5 * ((index - arrival - self.shift) / 5.0) ** 2)
        return out

    def _detections(self, annotations, **kwargs):
        return [[(0, 10, 0.9)] if bool(trace[1:].max() > 0.5) else [] for trace in annotations]


def _bundle(targets, length=3000, order="ZNE", rate=100.0):
    x = torch.arange(3.0).view(1, 3, 1).expand(len(targets), 3, length).clone()
    split = DataSplit(x, targets)
    return DataBundle(
        splits={"train": split, "val": split, "test": split},
        feature_spec=FeatureSpec(channels=3),
        label_spec=LabelSpec(num_targets=2, task_type="picking"),
        metadata={"dataset": "fixture", "sampling_rate": rate, "component_order": order},
    )


def _run(model, data, tmp_path, **params):
    config = load_experiment_config("pyhazards/configs/earthquake/phasenet_smoke.yaml")
    config.benchmark.params = params
    return run_benchmark("earthquake", model, data, config, output_dir=str(tmp_path))


TARGETS = torch.tensor([[500.0, 1500.0], [800.0, 1200.0], [float("nan"), float("nan")], [700.0, float("nan")]])


def test_perfect_picks_score_one_and_channels_are_reordered(tmp_path):
    model = _OraclePicker(TARGETS)
    summary = _run(model, _bundle(TARGETS), tmp_path, batch_size=3)
    assert summary.metrics["p_precision"] == summary.metrics["p_recall"] == summary.metrics["p_f1"] == 1.0
    assert summary.metrics["s_f1"] == 1.0
    assert summary.metrics["p_pick_mae"] == 0.0 and summary.metrics["p_residual_std"] == 0.0
    assert summary.metadata["pick_counts"]["S"] == {"manual": 2, "predicted": 2, "true_positive": 2}
    seen = torch.cat(model.seen)
    assert seen[0, :, 0].tolist() == [2.0, 1.0, 0.0]  # dataset Z, N, E -> model E, N, Z
    assert "detection_f1" not in summary.metrics


def test_tolerance_windows_follow_the_protocols(tmp_path):
    late = _OraclePicker(TARGETS, shift=20.0)  # 0.2 s late
    strict = _run(late, _bundle(TARGETS), tmp_path, protocol="phasenet")
    assert strict.metrics["p_precision"] == 0.0 and strict.metrics["p_recall"] == 0.0
    assert strict.metrics["p_residual_mean"] == pytest.approx(0.2)  # residuals within 0.5 s are kept
    late.seen.clear()
    loose = _run(late, _bundle(TARGETS), tmp_path, protocol="eqtransformer")
    assert loose.metrics["p_f1"] == 1.0 and loose.metrics["p_pick_mae"] == pytest.approx(0.2)
    late.seen.clear()
    custom = _run(late, _bundle(TARGETS), tmp_path, tolerance_s=0.25, residual_window_s=0.1)
    assert custom.metrics["p_f1"] == 1.0 and math.isnan(custom.metrics["p_pick_mae"])


def test_detection_metrics_and_regression_outputs(tmp_path):
    model = _OraclePicker(TARGETS, with_detection=True)
    summary = _run(model, _bundle(TARGETS), tmp_path)
    assert summary.metrics["detection_precision"] == summary.metrics["detection_recall"] == 1.0

    class Regressor(nn.Module):
        def forward(self, x):
            return TARGETS[: len(x)].nan_to_num(0.0) + 3.0

    summary = _run(Regressor(), _bundle(TARGETS), tmp_path)
    assert summary.metadata["pick_counts"]["P"] == {"manual": 3, "predicted": 4, "true_positive": 3}
    assert summary.metrics["p_precision"] == pytest.approx(0.75) and summary.metrics["p_recall"] == 1.0


def test_picking_input_checks(tmp_path):
    with pytest.raises(ValueError, match="Hz"):
        _run(_OraclePicker(TARGETS), _bundle(TARGETS, rate=50.0), tmp_path)
    with pytest.raises(ValueError, match="protocol"):
        _run(_OraclePicker(TARGETS), _bundle(TARGETS), tmp_path, protocol="unknown")

    class Bad(nn.Module):
        def forward(self, x):
            return x.mean(dim=(1, 2))

    with pytest.raises(ValueError, match="probabilities"):
        _run(Bad(), _bundle(TARGETS), tmp_path)


def test_synthetic_dataset_runs_through_the_picking_benchmark(tmp_path):
    data = load_dataset("earthquake_waveforms_synthetic", micro=True).load()
    test = data.get_split("test")
    model = _OraclePicker(test.targets)
    model.component_order = "ZNE"
    summary = _run(model, data, tmp_path)
    assert summary.metrics["p_f1"] == 1.0
