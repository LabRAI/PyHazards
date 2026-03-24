from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch.nn as nn

from ._wildfire_benchmark_utils import EstimatorPort, filter_init_kwargs, require_task


class LogisticRegressionModel(EstimatorPort):
    """A classical tabular binary-classification baseline for wildfire occurrence probability."""

    def __init__(self, solver: str = "lbfgs", max_iter: int = 500, class_weight: Any = "balanced"):
        super().__init__()
        from sklearn.linear_model import LogisticRegression

        self.estimator = LogisticRegression(
            solver=solver,
            max_iter=int(max_iter),
            class_weight=class_weight,
        )

    def _fit_numpy(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
    ) -> None:
                _ = x_val, y_val
                self.estimator.fit(x_train, y_train)

    def _predict_positive_proba(self, x: np.ndarray) -> np.ndarray:
                return self.estimator.predict_proba(x)[:, 1]


def logistic_regression_builder(task: str, **kwargs: Any) -> nn.Module:
    require_task(task, {"classification"}, "logistic_regression")
    build_kwargs = filter_init_kwargs(LogisticRegressionModel, kwargs)
    return LogisticRegressionModel(**build_kwargs)


__all__ = ["LogisticRegressionModel", "logistic_regression_builder"]
