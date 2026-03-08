from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from ..configs import ExperimentConfig
from ..datasets.base import DataBundle
from .base import Benchmark
from .registry import register_benchmark
from .schemas import BenchmarkResult


def _spread_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    targets = targets.float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou = float((intersection / union.clamp(min=1.0)).detach().cpu())
    f1 = float((2 * intersection / (preds.sum() + targets.sum()).clamp(min=1.0)).detach().cpu())
    return {"iou": iou, "f1": f1}


class WildfireBenchmark(Benchmark):
    name = "wildfire"
    hazard_task = "wildfire.danger"
    metric_names_by_task = {
        "wildfire.danger": ["accuracy", "macro_f1"],
        "wildfire.spread": ["iou", "f1"],
    }

    def evaluate(self, model: nn.Module, data: DataBundle, config: ExperimentConfig) -> BenchmarkResult:
        split = data.get_split(config.benchmark.eval_split)
        x = split.inputs
        y = split.targets
        logits = model(x)

        if config.benchmark.hazard_task == "wildfire.danger":
            preds = logits.argmax(dim=1)
            metrics = {
                "accuracy": float(accuracy_score(y.cpu().numpy(), preds.detach().cpu().numpy())),
                "macro_f1": float(
                    f1_score(y.cpu().numpy(), preds.detach().cpu().numpy(), average="macro")
                ),
            }
        else:
            metrics = _spread_metrics(logits, y)

        return BenchmarkResult(
            benchmark_name=self.name,
            hazard_task=config.benchmark.hazard_task,
            metrics=metrics,
            metadata={"split": config.benchmark.eval_split},
        )


register_benchmark(WildfireBenchmark.name, WildfireBenchmark)

__all__ = ["WildfireBenchmark"]
