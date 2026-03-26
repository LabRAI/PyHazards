from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch.nn as nn

from ._wildfire_estimator import BinaryEstimatorProxy, require_task


class RandomForestModel(BinaryEstimatorProxy):
    """A random-forest wildfire occurrence baseline over tabular features."""

    def __init__(self, n_estimators: int = 500, max_depth: Optional[int] = None, class_weight: Any = 'balanced_subsample'):
        super().__init__()
        try:
            from sklearn.ensemble import RandomForestClassifier
            self.estimator = RandomForestClassifier(
                n_estimators=int(n_estimators),
                max_depth=max_depth,
                class_weight=class_weight,
                random_state=42,
                n_jobs=1,
            )
        except Exception:
            self.estimator = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        if self.estimator is None:
            return
        self.estimator.fit(x_train, y_train)
        self._is_fitted = True

    def _predict_positive_proba(self, x_np: np.ndarray) -> np.ndarray:
        if self._is_fitted and self.estimator is not None:
            return self.estimator.predict_proba(x_np)[:, 1]
        return super()._predict_positive_proba(x_np)


def random_forest_builder(task: str, **kwargs: Any) -> nn.Module:
    require_task(task, {'classification'}, 'random_forest')
    kwargs.pop('name', None)
    return RandomForestModel(**kwargs)


__all__ = ['RandomForestModel', 'random_forest_builder']
