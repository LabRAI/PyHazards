import torch
from torch import nn

from pyhazards.benchmarks import register_benchmark
from pyhazards.benchmarks.base import Benchmark
from pyhazards.benchmarks.schemas import BenchmarkResult
from pyhazards.configs import (
    BenchmarkConfig,
    DatasetRef,
    ExperimentConfig,
    ModelRef,
    ReportConfig,
    dump_experiment_config,
    load_experiment_config,
)
from pyhazards.datasets import DataBundle, DataSplit, FeatureSpec, LabelSpec
from pyhazards.engine.runner import BenchmarkRunner


class DummyRegressionBenchmark(Benchmark):
    name = "dummy_regression"
    hazard_task = "flood.streamflow"

    def evaluate(self, model: nn.Module, data: DataBundle, config: ExperimentConfig) -> BenchmarkResult:
        split = data.get_split(config.benchmark.eval_split)
        preds = model(split.inputs)
        mae = float(torch.mean(torch.abs(preds - split.targets)).item())
        return BenchmarkResult(
            benchmark_name=self.name,
            hazard_task=config.benchmark.hazard_task,
            metrics={"mae": mae},
            metadata={"split": config.benchmark.eval_split},
        )


def _dummy_bundle() -> DataBundle:
    x = torch.randn(16, 4)
    y = torch.randn(16, 1)
    splits = {
        "train": DataSplit(x[:8], y[:8]),
        "val": DataSplit(x[8:12], y[8:12]),
        "test": DataSplit(x[12:], y[12:]),
    }
    return DataBundle(
        splits=splits,
        feature_spec=FeatureSpec(input_dim=4),
        label_spec=LabelSpec(num_targets=1, task_type="regression"),
    )


def test_benchmark_runner_executes_dummy_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr("pyhazards.benchmarks.registry._BENCHMARK_REGISTRY", {})
    register_benchmark("dummy_regression", DummyRegressionBenchmark)

    experiment = ExperimentConfig(
        benchmark=BenchmarkConfig(
            name="dummy_regression",
            hazard_task="flood.streamflow",
            metrics=["mae"],
            eval_split="test",
        ),
        dataset=DatasetRef(name="unused"),
        model=ModelRef(name="mlp", task="regression", params={"in_dim": 4, "out_dim": 1}),
        report=ReportConfig(output_dir=str(tmp_path), formats=["json", "md"]),
    )

    summary = BenchmarkRunner().run(
        experiment,
        data=_dummy_bundle(),
        output_dir=str(tmp_path),
    )

    assert summary.benchmark_name == "dummy_regression"
    assert summary.hazard_task == "flood.streamflow"
    assert "mae" in summary.metrics
    assert summary.report_paths["json"].endswith("dummy_regression.json")
    assert (tmp_path / "dummy_regression.md").exists()


def test_experiment_config_roundtrip(tmp_path):
    path = tmp_path / "experiment.yaml"
    experiment = ExperimentConfig(
        benchmark=BenchmarkConfig(name="dummy_regression", hazard_task="flood.streamflow"),
        dataset=DatasetRef(name="fpa_fod_tabular", params={"micro": True, "task": "cause"}),
        model=ModelRef(name="wildfire_fpa", task="classification", params={"in_dim": 8, "out_dim": 5}),
        report=ReportConfig(output_dir="reports", formats=["json", "csv"]),
        seed=7,
        metadata={"owner": "wave1"},
    )

    dump_experiment_config(experiment, path)
    loaded = load_experiment_config(path)

    assert loaded.benchmark.hazard_task == "flood.streamflow"
    assert loaded.dataset.params["micro"] is True
    assert loaded.model.params["out_dim"] == 5
    assert loaded.report.formats == ["json", "csv"]
    assert loaded.seed == 7
