from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..configs import ExperimentConfig
from ..datasets.base import DataBundle
from ..datasets.graph import graph_collate
from .base import Benchmark
from .registry import register_benchmark
from .schemas import BenchmarkResult


class FloodBenchmark(Benchmark):
    name = "flood"
    hazard_task = "flood.streamflow"
    metric_names_by_task = {
        "flood.streamflow": ["mae", "rmse", "nse", "kge"],
        "flood.inundation": ["pixel_mae", "iou", "f1"],
    }

    def evaluate(self, model: nn.Module, data: DataBundle, config: ExperimentConfig) -> BenchmarkResult:
        split = data.get_split(config.benchmark.eval_split)
        if (
            config.benchmark.hazard_task == "flood.streamflow"
            and hasattr(split.inputs, "__len__")
            and not isinstance(split.inputs, torch.Tensor)
        ):
            loader = DataLoader(split.inputs, batch_size=4, shuffle=False, collate_fn=graph_collate)
            preds_all = []
            target_all = []
            with torch.no_grad():
                for batch, target in loader:
                    preds_all.append(model(batch))
                    target_all.append(target)
            preds = torch.cat(preds_all, dim=0)
            targets = torch.cat(target_all, dim=0)
        else:
            preds = model(split.inputs)
            targets = split.targets

        if config.benchmark.hazard_task == "flood.inundation":
            pred_depth = preds.float()
            target_depth = targets.float()
            pred_mask = (pred_depth >= 0.5).float()
            target_mask = (target_depth > 0).float()
            intersection = (pred_mask * target_mask).sum()
            union = pred_mask.sum() + target_mask.sum() - intersection
            metrics: Dict[str, float] = {
                "pixel_mae": float(torch.mean(torch.abs(pred_depth - target_depth)).detach().cpu()),
                "iou": float((intersection / union.clamp(min=1.0)).detach().cpu()),
                "f1": float(
                    (
                        2 * intersection
                        / (pred_mask.sum() + target_mask.sum()).clamp(min=1.0)
                    ).detach().cpu()
                ),
            }
        else:
            mae = torch.mean(torch.abs(preds - targets))
            rmse = torch.sqrt(torch.mean((preds - targets) ** 2))
            target_mean = torch.mean(targets)
            denominator = torch.sum((targets - target_mean) ** 2).clamp(min=1e-6)
            nse = 1.0 - torch.sum((preds - targets) ** 2) / denominator
            pred_std = torch.std(preds).clamp(min=1e-6)
            target_std = torch.std(targets).clamp(min=1e-6)
            covariance = torch.mean((preds - torch.mean(preds)) * (targets - target_mean))
            correlation = covariance / (pred_std * target_std)
            alpha = pred_std / target_std
            beta = torch.mean(preds) / target_mean.clamp(min=1e-6)
            kge = 1.0 - torch.sqrt((correlation - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
            metrics = {
                "mae": float(mae.detach().cpu()),
                "rmse": float(rmse.detach().cpu()),
                "nse": float(nse.detach().cpu()),
                "kge": float(kge.detach().cpu()),
            }
        return BenchmarkResult(
            benchmark_name=self.name,
            hazard_task=config.benchmark.hazard_task,
            metrics=metrics,
            metadata={
                "split": config.benchmark.eval_split,
                "dataset_name": data.metadata.get("dataset"),
                "source_dataset": data.metadata.get("source_dataset", data.metadata.get("dataset")),
            },
        )


register_benchmark(FloodBenchmark.name, FloodBenchmark)

__all__ = ["FloodBenchmark"]
