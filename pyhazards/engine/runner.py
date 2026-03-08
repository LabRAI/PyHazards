from __future__ import annotations

from typing import Optional, Union

import torch.nn as nn

from ..benchmarks import Benchmark, BenchmarkRunSummary, run_benchmark
from ..configs import ExperimentConfig
from ..datasets import load_dataset
from ..datasets.base import DataBundle
from ..models import build_model


class BenchmarkRunner:
    """High-level runner that resolves datasets/models and executes a benchmark."""

    def __init__(self, benchmark: Optional[Union[str, Benchmark]] = None):
        self.benchmark = benchmark

    def run(
        self,
        experiment: ExperimentConfig,
        model: Optional[nn.Module] = None,
        data: Optional[DataBundle] = None,
        output_dir: Optional[str] = None,
    ) -> BenchmarkRunSummary:
        built_model = model or self._build_model(experiment)
        bundle = data or self._load_data(experiment)
        benchmark = self.benchmark or experiment.benchmark.name
        return run_benchmark(
            benchmark=benchmark,
            model=built_model,
            data=bundle,
            config=experiment,
            output_dir=output_dir,
        )

    def _build_model(self, experiment: ExperimentConfig) -> nn.Module:
        return build_model(
            name=experiment.model.name,
            task=experiment.model.task,
            **experiment.model.params,
        )

    def _load_data(self, experiment: ExperimentConfig) -> DataBundle:
        return load_dataset(
            experiment.dataset.name,
            **experiment.dataset.params,
        ).load()
