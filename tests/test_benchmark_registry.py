import pytest
import torch.nn as nn

from pyhazards.benchmarks import available_benchmarks, build_benchmark, register_benchmark
from pyhazards.benchmarks.base import Benchmark
from pyhazards.benchmarks.registry import _BENCHMARK_REGISTRY
from pyhazards.benchmarks.schemas import BenchmarkResult


class DummyBenchmark(Benchmark):
    name = "dummy_benchmark"
    hazard_task = "wildfire.danger"

    def evaluate(self, model: nn.Module, data, config):
        return BenchmarkResult(
            benchmark_name=self.name,
            hazard_task=self.hazard_task,
            metrics={"score": 1.0},
        )


def test_register_and_build_benchmark(monkeypatch):
    monkeypatch.setattr("pyhazards.benchmarks.registry._BENCHMARK_REGISTRY", {})
    register_benchmark("dummy_benchmark", DummyBenchmark)

    assert available_benchmarks() == ["dummy_benchmark"]
    benchmark = build_benchmark("dummy_benchmark")
    assert isinstance(benchmark, DummyBenchmark)


def test_duplicate_registration_raises(monkeypatch):
    monkeypatch.setattr("pyhazards.benchmarks.registry._BENCHMARK_REGISTRY", {})
    register_benchmark("dummy_benchmark", DummyBenchmark)

    with pytest.raises(ValueError):
        register_benchmark("dummy_benchmark", DummyBenchmark)


def test_aggregate_metrics_averages_results():
    benchmark = DummyBenchmark()
    metrics = benchmark.aggregate_metrics(
        [
            BenchmarkResult("dummy", "wildfire.danger", {"score": 1.0, "loss": 4.0}),
            BenchmarkResult("dummy", "wildfire.danger", {"score": 3.0, "loss": 2.0}),
        ]
    )
    assert metrics == {"loss": 3.0, "score": 2.0}
