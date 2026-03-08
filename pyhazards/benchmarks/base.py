from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Sequence

import torch.nn as nn

from ..configs import ExperimentConfig
from ..datasets.base import DataBundle
from ..reports import BenchmarkReport, export_report_bundle
from .schemas import BenchmarkResult


class Benchmark(ABC):
    """Shared benchmark contract for hazard evaluators."""

    name: str = "benchmark"
    hazard_task: str = ""

    @abstractmethod
    def evaluate(
        self,
        model: nn.Module,
        data: DataBundle,
        config: ExperimentConfig,
    ) -> BenchmarkResult:
        raise NotImplementedError

    def aggregate_metrics(self, results: Sequence[BenchmarkResult]) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for result in results:
            for key, value in result.metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
        return {
            key: totals[key] / counts[key]
            for key in sorted(totals.keys())
            if counts[key] > 0
        }

    def export_report(
        self,
        result: BenchmarkResult,
        output_dir: str,
        formats: Iterable[str],
    ) -> Dict[str, str]:
        report = BenchmarkReport(
            benchmark_name=result.benchmark_name,
            hazard_task=result.hazard_task,
            metrics=result.metrics,
            metadata=result.metadata,
            artifacts=result.artifacts,
        )
        return export_report_bundle(report, output_dir=output_dir, formats=list(formats))
