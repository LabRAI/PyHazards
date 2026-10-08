from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn

from ..configs import ExperimentConfig
from ..datasets.base import DataBundle
from .base import Benchmark
from .registry import build_benchmark
from .schemas import BenchmarkRunSummary


def resolve_benchmark(benchmark: Union[str, Benchmark]) -> Benchmark:
    if isinstance(benchmark, Benchmark):
        return benchmark
    return build_benchmark(benchmark)


def run_benchmark(
    benchmark: Union[str, Benchmark],
    model: nn.Module,
    data: DataBundle,
    config: ExperimentConfig,
    output_dir: str | None = None,
) -> BenchmarkRunSummary:
    benchmark_obj = resolve_benchmark(benchmark)
    # Score in inference mode (dropout off, BatchNorm running statistics) and restore the mode after.
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            result = benchmark_obj.evaluate(model=model, data=data, config=config)
    finally:
        model.train(was_training)
    metrics = benchmark_obj.aggregate_metrics([result])
    result.metrics = metrics
    report_dir = output_dir or config.report.output_dir
    report_paths = benchmark_obj.export_report(result, output_dir=report_dir, formats=config.report.formats)
    metadata = dict(result.metadata)
    metadata.setdefault("eval_split", config.benchmark.eval_split)
    return BenchmarkRunSummary(
        benchmark_name=result.benchmark_name,
        hazard_task=result.hazard_task,
        metrics=metrics,
        report_paths=report_paths,
        metadata=metadata,
    )
