from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch.nn as nn

from ._wildfire_benchmark_utils import EstimatorPort, filter_init_kwargs, require_task


class RandomForestModel(EstimatorPort):
    """A tree-ensemble baseline for wildfire occurrence probability over tabular features."""

    def __init__(self, n_estimators: int = 500, max_depth: Optional[int] = None, class_weight: Any = "balanced_subsample"):
        super().__init__()
        from sklearn.ensemble import RandomForestClassifier

        self.estimator = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=42,
            n_jobs=1,
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


def random_forest_builder(task: str, **kwargs: Any) -> nn.Module:
    require_task(task, {"classification"}, "random_forest")
    build_kwargs = filter_init_kwargs(RandomForestModel, kwargs)
    return RandomForestModel(**build_kwargs)


__all__ = ["RandomForestModel", "random_forest_builder"]
