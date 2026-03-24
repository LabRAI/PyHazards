from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch.nn as nn

from ._wildfire_benchmark_utils import EstimatorPort, filter_init_kwargs, require_task


class LightGBMModel(EstimatorPort):
    """A boosted-tree wildfire occurrence baseline using LightGBM binary classification."""

    def __init__(self, num_leaves: int = 63, learning_rate: float = 0.05, feature_fraction: float = 0.8, bagging_fraction: float = 0.8, num_boost_round: int = 800):
        super().__init__()
        self.params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": int(num_leaves),
            "learning_rate": float(learning_rate),
            "feature_fraction": float(feature_fraction),
            "bagging_fraction": float(bagging_fraction),
            "verbose": -1,
        }
        self.num_boost_round = int(num_boost_round)
        self.booster = None

    def _fit_numpy(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
    ) -> None:
        import lightgbm as lgb

        dtrain = lgb.Dataset(x_train, label=y_train)
        valid_sets = [dtrain]
        valid_names = ["train"]
        if x_val is not None and y_val is not None:
            dval = lgb.Dataset(x_val, label=y_val, reference=dtrain)
            valid_sets.append(dval)
            valid_names.append("val")
        self.booster = lgb.train(
            params=self.params,
            train_set=dtrain,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[lgb.log_evaluation(period=0)],
        )

    def _predict_positive_proba(self, x: np.ndarray) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("LightGBM booster is not fitted.")
        return np.asarray(self.booster.predict(x), dtype=np.float32)


def lightgbm_builder(task: str, **kwargs: Any) -> nn.Module:
    require_task(task, {"classification"}, "lightgbm")
    build_kwargs = filter_init_kwargs(LightGBMModel, kwargs)
    return LightGBMModel(**build_kwargs)


__all__ = ["LightGBMModel", "lightgbm_builder"]
