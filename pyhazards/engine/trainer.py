from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset

from ..datasets.base import DataBundle
from ..metrics import (
    ClassificationMetrics,
    MetricBase,
    RegressionMetrics,
    SegmentationMetrics,
)
from ..utils.hardware import auto_device
from .distributed import select_strategy


class Trainer:
    """
    Lightweight training abstraction with a familiar API:
    fit -> evaluate -> predict.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device | str] = None,
        metrics: Optional[List[MetricBase]] = None,
        strategy: str = "auto",
        mixed_precision: bool = False,
    ):
        self.model = model
        self.device = torch.device(device) if device else auto_device()
        self.metrics = list(metrics) if metrics is not None else None
        self.strategy = select_strategy(strategy)
        self.mixed_precision = mixed_precision
        self.model.to(self.device)

    def fit(
        self,
        data: DataBundle,
        train_split: str = "train",
        val_split: Optional[str] = None,
        max_epochs: int = 1,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        collate_fn: Optional[Callable[[List[Any]], Any]] = None,
    ) -> None:
        """
        Minimal fit loop that works for tensor-based splits.
        Extend/replace with custom DataLoaders for complex data.

        Models that are not trained as one network by gradient descent declare a
        ``custom_fit_reason`` (scikit-learn estimator wrappers such as ``random_forest`` and
        ``xgboost``, and ``deep_ensemble``, whose members train independently); ``fit`` raises
        ``TypeError`` with that reason instead of running a loop that would not train them.
        ``evaluate`` and ``predict`` work for them.
        """
        custom_fit_reason = getattr(self.model, "custom_fit_reason", None)
        if custom_fit_reason:
            raise TypeError(f"Trainer.fit cannot train this model: {custom_fit_reason}")
        if optimizer is None or loss_fn is None:
            raise ValueError("optimizer and loss_fn must be provided.")

        train_split_data = data.get_split(train_split)
        train_loader = self._make_loader(train_split_data.inputs, train_split_data.targets, batch_size, num_workers, collate_fn)
        amp_enabled = self.mixed_precision and self.device.type == "cuda"
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
            use_new_amp = True
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
            use_new_amp = False

        self.model.train()
        for _ in range(max_epochs):
            for x, y in train_loader:
                x = self._to_device(x)
                y = self._to_device(y)
                optimizer.zero_grad()
                if use_new_amp:
                    with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                        out = self.model(x)
                        loss = loss_fn(out, y)
                else:
                    with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                        out = self.model(x)
                        loss = loss_fn(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        if val_split:
            self.evaluate(data, split=val_split)

    def evaluate(
        self,
        data: DataBundle,
        split: str = "test",
        batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Optional[Callable[[List[Any]], Any]] = None,
    ) -> Dict[str, float]:
        split_data = data.get_split(split)
        loader = self._make_loader(split_data.inputs, split_data.targets, batch_size, num_workers, collate_fn, shuffle=False)
        metrics = self._resolve_metrics(data)
        self.model.eval()
        for metric in metrics:
            metric.reset()
        with torch.no_grad():
            for x, y in loader:
                x = self._to_device(x)
                y = self._to_device(y)
                preds = self.model(x)
                for metric in metrics:
                    metric.update(preds, y)
        results: Dict[str, float] = {}
        for metric in metrics:
            results.update(metric.compute())
        return results

    def predict(
        self,
        data: DataBundle,
        split: str = "test",
        batch_size: int = 64,
        num_workers: int = 0,
        collate_fn: Optional[Callable[[List[Any]], Any]] = None,
    ) -> List[torch.Tensor]:
        split_data = data.get_split(split)
        loader = self._make_loader(split_data.inputs, split_data.targets, batch_size, num_workers, collate_fn, shuffle=False)
        self.model.eval()
        outputs: List[torch.Tensor] = []
        with torch.no_grad():
            for x, _ in loader:
                x = self._to_device(x)
                preds = self.model(x)
                outputs.append(preds.cpu())
        return outputs

    def save_checkpoint(self, path: str) -> None:
        torch.save({"model_state": self.model.state_dict()}, path)

    def _make_loader(
        self,
        inputs: Any,
        targets: Any,
        batch_size: int,
        num_workers: int,
        collate_fn: Optional[Callable[[List[Any]], Any]],
        shuffle: bool = True,
    ) -> Iterable:
        # Accept torch tensors
        if isinstance(inputs, torch.Tensor) and isinstance(targets, torch.Tensor):
            dataset = TensorDataset(inputs, targets)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
        # Accept torch.utils.data.Dataset directly (for complex dict/graph batches)
        if isinstance(inputs, Dataset):
            return DataLoader(inputs, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
        raise TypeError("Trainer only supports tensor pairs or torch Dataset inputs. Wrap custom logic in a Dataset.")

    def _to_device(self, obj: Any) -> Any:
        if obj is None:
            return None
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device)
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._to_device(o) for o in obj)
        if isinstance(obj, dict):
            return {k: self._to_device(v) for k, v in obj.items()}
        return obj

    def _resolve_metrics(self, data: DataBundle) -> List[MetricBase]:
        if self.metrics is not None:
            return self.metrics

        task_type = data.label_spec.task_type.lower()
        if task_type == "regression":
            return [RegressionMetrics()]
        if task_type == "classification":
            return [ClassificationMetrics()]
        if task_type == "segmentation":
            return [SegmentationMetrics()]
        return []
