from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch.nn as nn

from ._wildfire_benchmark_utils import EstimatorPort, filter_init_kwargs, require_task


class XGBoostModel(EstimatorPort):
    """A boosted-tree wildfire occurrence baseline using a binary logistic objective."""

    def __init__(self, max_depth: int = 8, eta: float = 0.05, subsample: float = 0.8, colsample_bytree: float = 0.8, num_boost_round: int = 800):
        super().__init__()
        self.params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": int(max_depth),
            "eta": float(eta),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample_bytree),
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
        import xgboost as xgb

        dtrain = xgb.DMatrix(x_train, label=y_train)
        evals = [(dtrain, "train")]
        if x_val is not None and y_val is not None:
            dval = xgb.DMatrix(x_val, label=y_val)
            evals.append((dval, "val"))
        self.booster = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            verbose_eval=False,
        )

    def _predict_positive_proba(self, x: np.ndarray) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("XGBoost booster is not fitted.")
        import xgboost as xgb
        return np.asarray(self.booster.predict(xgb.DMatrix(x)), dtype=np.float32)


def xgboost_builder(task: str, **kwargs: Any) -> nn.Module:
    require_task(task, {"classification"}, "xgboost")
    build_kwargs = filter_init_kwargs(XGBoostModel, kwargs)
    return XGBoostModel(**build_kwargs)


__all__ = ["XGBoostModel", "xgboost_builder"]
