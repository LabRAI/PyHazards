from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from ..configs import ExperimentConfig
from ..datasets.base import DataBundle
from .base import Benchmark
from .registry import register_benchmark
from .schemas import BenchmarkResult


class EarthquakeBenchmark(Benchmark):
    name = "earthquake"
    hazard_task = "earthquake.picking"
    metric_names_by_task = {
        "earthquake.picking": ["p_pick_mae", "s_pick_mae", "precision", "recall", "f1"],
        "earthquake.forecasting": ["mae", "mse"],
    }

    def evaluate(self, model: nn.Module, data: DataBundle, config: ExperimentConfig) -> BenchmarkResult:
        split = data.get_split(config.benchmark.eval_split)
        x = split.inputs
        y = split.targets
        preds = model(x)

        if config.benchmark.hazard_task == "earthquake.picking":
            mae = (preds - y).abs()
            tolerances = config.benchmark.params.get("detection_tolerances", [4.0, 8.0, 12.0])
            threshold_curve: Dict[str, float] = {}
            detection_rate = 0.0
            for tolerance in tolerances:
                hits = ((preds - y).abs() <= float(tolerance)).all(dim=1).float()
                hit_rate = float(hits.mean().detach().cpu())
                threshold_curve[str(tolerance)] = hit_rate
                if float(tolerance) == 8.0:
                    detection_rate = hit_rate

            metrics = {
                "p_pick_mae": float(mae[:, 0].mean().detach().cpu()),
                "s_pick_mae": float(mae[:, 1].mean().detach().cpu()),
                "mean_pick_mae": float(mae.mean().detach().cpu()),
                "precision": detection_rate,
                "recall": detection_rate,
                "f1": detection_rate,
            }
        else:
            mse = torch.mean((preds - y) ** 2)
            mae = torch.mean(torch.abs(preds - y))
            threshold_curve = {}
            metrics = {
                "mae": float(mae.detach().cpu()),
                "mse": float(mse.detach().cpu()),
            }

        return BenchmarkResult(
            benchmark_name=self.name,
            hazard_task=config.benchmark.hazard_task,
            metrics=metrics,
            metadata={
                "split": config.benchmark.eval_split,
                "threshold_curve": threshold_curve,
                "dataset_name": data.metadata.get("dataset"),
                "source_dataset": data.metadata.get("source_dataset", data.metadata.get("dataset")),
            },
        )

    def export_report(
        self,
        result: BenchmarkResult,
        output_dir: str,
        formats,
    ) -> Dict[str, str]:
        paths = super().export_report(result, output_dir=output_dir, formats=formats)
        if result.hazard_task == "earthquake.forecasting":
            target = Path(output_dir)
            target.mkdir(parents=True, exist_ok=True)
            pycsep_path = target / "earthquake_pycsep.json"
            payload = {
                "adapter": "pyCSEP-style",
                "benchmark_name": result.benchmark_name,
                "hazard_task": result.hazard_task,
                "metrics": result.metrics,
                "metadata": result.metadata,
            }
            pycsep_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            paths["pycsep"] = str(pycsep_path)
        return paths


register_benchmark(EarthquakeBenchmark.name, EarthquakeBenchmark)

__all__ = ["EarthquakeBenchmark"]
