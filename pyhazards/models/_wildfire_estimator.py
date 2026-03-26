from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn


def require_task(task: str, allowed: set[str], model_name: str) -> None:
    normalized = task.lower()
    if normalized not in allowed:
        allowed_text = ', '.join(sorted(allowed))
        raise ValueError(f"Model '{model_name}' does not support task={task!r}. Allowed tasks: {allowed_text}")


def flatten_tensor(x: torch.Tensor) -> np.ndarray:
    if not isinstance(x, torch.Tensor):
        raise TypeError('Expected torch.Tensor input for estimator-style wildfire models.')
    x_np = x.detach().cpu().float().numpy()
    if x_np.ndim == 1:
        x_np = x_np[:, None]
    if x_np.ndim > 2:
        x_np = x_np.reshape(x_np.shape[0], -1)
    return x_np


class BinaryEstimatorProxy(nn.Module):
    def __init__(self):
        super().__init__()
        self._is_fitted = False

    def _fallback_positive_proba(self, x_np: np.ndarray) -> np.ndarray:
        score = np.clip(x_np.mean(axis=1), -8.0, 8.0)
        return 1.0 / (1.0 + np.exp(-score))

    def _predict_positive_proba(self, x_np: np.ndarray) -> np.ndarray:
        return self._fallback_positive_proba(x_np)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_np = flatten_tensor(x)
        probs_pos = np.clip(self._predict_positive_proba(x_np), 1e-6, 1.0 - 1e-6)
        probs = np.stack([1.0 - probs_pos, probs_pos], axis=-1).astype(np.float32)
        return torch.from_numpy(probs).to(x.device)
