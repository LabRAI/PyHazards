from __future__ import annotations

import inspect
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from ..datasets.base import DataBundle


def require_task(task: str, allowed: set[str], model_name: str) -> None:
    normalized = task.lower()
    if normalized not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(f"Model '{model_name}' does not support task={task!r}. Allowed tasks: {allowed_text}")


def filter_init_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(callable_obj)
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if accepts_kwargs:
        return dict(kwargs)
    allowed = {name for name in sig.parameters if name != 'self'}
    return {k: v for k, v in kwargs.items() if k in allowed}


class SegmentationPort(nn.Module):
    def __init__(self, model: nn.Module, out_channels: int = 1):
        super().__init__()
        self.model = model
        self.out_channels = int(out_channels)
        self.output_head = nn.Identity() if self.out_channels == 1 else nn.Conv2d(1, self.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return self.output_head(logits)


def _to_numpy_2d(x: torch.Tensor) -> np.ndarray:
    if not isinstance(x, torch.Tensor):
        raise TypeError('Expected torch.Tensor inputs for estimator-based models.')
    x_np = x.detach().cpu().float().numpy()
    if x_np.ndim == 1:
        x_np = x_np[:, None]
    if x_np.ndim > 2:
        x_np = x_np.reshape(x_np.shape[0], -1)
    return x_np


def _to_numpy_labels(y: torch.Tensor) -> np.ndarray:
    if not isinstance(y, torch.Tensor):
        raise TypeError('Expected torch.Tensor targets for estimator-based models.')
    y_np = y.detach().cpu().numpy()
    if y_np.ndim > 1:
        y_np = y_np.reshape(y_np.shape[0], -1)
        if y_np.shape[1] != 1:
            raise ValueError('Estimator-based models expect a single target column.')
        y_np = y_np[:, 0]
    return y_np.astype(np.int64)


class EstimatorPort(nn.Module):
    def __init__(self):
        super().__init__()
        self._is_fitted = False

    def fit_bundle(
        self,
        data: DataBundle,
        train_split: str = 'train',
        val_split: Optional[str] = None,
        **_: Any,
    ) -> None:
        train_data = data.get_split(train_split)
        x_train = _to_numpy_2d(train_data.inputs)
        y_train = _to_numpy_labels(train_data.targets)

        x_val = None
        y_val = None
        if val_split:
            val_data = data.get_split(val_split)
            x_val = _to_numpy_2d(val_data.inputs)
            y_val = _to_numpy_labels(val_data.targets)

        self._fit_numpy(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
        self._is_fitted = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_fitted:
            raise RuntimeError(
                f'{self.__class__.__name__} has not been fitted. Use Trainer.fit(...) with a tensor-backed DataBundle first.'
            )
        x_np = _to_numpy_2d(x)
        probs_pos = self._predict_positive_proba(x_np)
        probs = np.stack([1.0 - probs_pos, probs_pos], axis=-1).astype(np.float32)
        return torch.from_numpy(probs).to(x.device)

    def _fit_numpy(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
    ) -> None:
        raise NotImplementedError

    def _predict_positive_proba(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
