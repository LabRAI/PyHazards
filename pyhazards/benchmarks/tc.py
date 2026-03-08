from __future__ import annotations

import torch
import torch.nn as nn

from ..configs import ExperimentConfig
from ..datasets.base import DataBundle
from .base import Benchmark
from .registry import register_benchmark
from .schemas import BenchmarkResult


class TropicalCycloneBenchmark(Benchmark):
    name = "tc"
    hazard_task = "tc.track_intensity"
    metric_names_by_task = {
        "tc.track_intensity": ["track_error", "intensity_mae"],
    }

    def evaluate(self, model: nn.Module, data: DataBundle, config: ExperimentConfig) -> BenchmarkResult:
        split = data.get_split(config.benchmark.eval_split)
        preds = model(split.inputs)
        targets = split.targets

        track_error = torch.norm(preds[..., :2] - targets[..., :2], dim=-1).mean()
        intensity_mae = torch.mean(torch.abs(preds[..., 2] - targets[..., 2]))
        metrics = {
            "track_error": float(track_error.detach().cpu()),
            "intensity_mae": float(intensity_mae.detach().cpu()),
        }
        return BenchmarkResult(
            benchmark_name=self.name,
            hazard_task=config.benchmark.hazard_task,
            metrics=metrics,
            metadata={"split": config.benchmark.eval_split},
        )


register_benchmark(TropicalCycloneBenchmark.name, TropicalCycloneBenchmark)

__all__ = ["TropicalCycloneBenchmark"]
