from __future__ import annotations

from typing import Any

import numpy as np
import torch.nn as nn

from ._wildfire_estimator import BinaryEstimatorProxy, require_task


class LogisticRegressionModel(BinaryEstimatorProxy):
    """A classical logistic baseline for wildfire occurrence probability."""

    def __init__(self, solver: str = 'lbfgs', max_iter: int = 500, class_weight: Any = 'balanced'):
        super().__init__()
        try:
            from sklearn.linear_model import LogisticRegression
            self.estimator = LogisticRegression(solver=solver, max_iter=int(max_iter), class_weight=class_weight)
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


def logistic_regression_builder(task: str, **kwargs: Any) -> nn.Module:
    require_task(task, {'classification'}, 'logistic_regression')
    return LogisticRegressionModel(**kwargs)


__all__ = ['LogisticRegressionModel', 'logistic_regression_builder']
