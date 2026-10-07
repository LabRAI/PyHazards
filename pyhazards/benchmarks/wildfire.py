from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

from ..configs import ExperimentConfig
from ..datasets.base import DataBundle
from .base import Benchmark
from .registry import register_benchmark
from .schemas import BenchmarkResult


def _spread_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    # Pixel-level average precision is the primary WildfireSpreadTS metric.
    flat_targets = targets.detach().flatten().cpu().numpy() > 0.5
    flat_probs = probs.detach().flatten().cpu().numpy()
    average_precision = float(average_precision_score(flat_targets, flat_probs)) if flat_targets.any() else 0.0
    preds = (probs >= 0.5).float()
    targets = targets.float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou = float((intersection / union.clamp(min=1.0)).detach().cpu())
    f1 = float((2 * intersection / (preds.sum() + targets.sum()).clamp(min=1.0)).detach().cpu())
    burned_area_mae = float(
        torch.mean(torch.abs(preds.flatten(1).sum(dim=1) - targets.flatten(1).sum(dim=1))).detach().cpu()
    )
    return {
        "average_precision": average_precision,
        "iou": iou,
        "f1": f1,
        "burned_area_mae": burned_area_mae,
    }


def _binary_map_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Danger metrics for per-cell binary maps: one logit per cell, 0/1 targets of the same shape."""
    probs = torch.sigmoid(logits.detach().float()).flatten().cpu().numpy()
    y_true = targets.detach().float().flatten().cpu().numpy() > 0.5
    y_pred = probs >= 0.5
    try:
        auc = float(roc_auc_score(y_true, probs))
    except ValueError:
        auc = 0.0
    pr_auc = float(average_precision_score(y_true, probs)) if y_true.any() else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[False, True], zero_division=0)),
        "auc": auc,
        "pr_auc": pr_auc,
    }


def _batched_forward(model: nn.Module, inputs: torch.Tensor, batch_size: int | None) -> torch.Tensor:
    """Run ``model`` over ``inputs`` in chunks on the model's device and return CPU outputs."""
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else None
    if not batch_size or batch_size >= len(inputs):
        chunks = [inputs]
    else:
        chunks = list(torch.split(inputs, int(batch_size)))
    outputs = []
    for chunk in chunks:
        if device is not None and isinstance(chunk, torch.Tensor):
            chunk = chunk.to(device)
        outputs.append(model(chunk).detach().cpu())
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs)


def _danger_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    if targets.dtype in {torch.int32, torch.int64} or targets.ndim == 1:
        preds = logits.argmax(dim=1)
        probs = F.softmax(logits, dim=1)
        y_true = targets.detach().cpu().numpy()
        y_pred = preds.detach().cpu().numpy()
        y_score = probs.detach().cpu().numpy()
        one_hot = F.one_hot(targets.long(), num_classes=logits.size(1)).detach().cpu().numpy()
        try:
            auc = float(roc_auc_score(one_hot, y_score, average="macro", multi_class="ovr"))
        except ValueError:
            auc = 0.0
        try:
            pr_auc = float(average_precision_score(one_hot, y_score, average="macro"))
        except ValueError:
            pr_auc = 0.0
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "auc": auc,
            "pr_auc": pr_auc,
        }

    preds = logits.float()
    targets = targets.float()
    mae = torch.mean(torch.abs(preds - targets))
    rmse = torch.sqrt(torch.mean((preds - targets) ** 2))
    return {
        "mae": float(mae.detach().cpu()),
        "rmse": float(rmse.detach().cpu()),
    }


class WildfireBenchmark(Benchmark):
    name = "wildfire"
    hazard_task = "wildfire.danger"
    metric_names_by_task = {
        "wildfire.danger": ["accuracy", "macro_f1", "auc", "pr_auc", "mae", "rmse"],
        "wildfire.spread": ["average_precision", "iou", "f1", "burned_area_mae"],
    }

    def evaluate(self, model: nn.Module, data: DataBundle, config: ExperimentConfig) -> BenchmarkResult:
        split = data.get_split(config.benchmark.eval_split)
        x = split.inputs
        y = split.targets
        logits = _batched_forward(model, x, config.benchmark.params.get("batch_size"))

        if config.benchmark.hazard_task == "wildfire.danger":
            if data.label_spec.task_type == "segmentation":
                # Danger maps (e.g. the Track-O raster and temporal layouts): one logit per cell.
                if tuple(logits.shape) != tuple(y.shape):
                    raise ValueError(
                        "wildfire.danger maps need logits with the target shape (one logit per cell): "
                        f"got logits of shape {tuple(logits.shape)} for targets of shape {tuple(y.shape)}."
                    )
                metrics = _binary_map_metrics(logits, y)
            else:
                metrics = _danger_metrics(logits, y)
        else:
            metrics = _spread_metrics(logits, y)

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


register_benchmark(WildfireBenchmark.name, WildfireBenchmark)

__all__ = ["WildfireBenchmark"]
