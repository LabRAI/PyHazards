from __future__ import annotations

from typing import Any

import numpy as np
import torch.nn as nn

from ._wildfire_estimator import BinaryEstimatorProxy, require_task


class XGBoostModel(BinaryEstimatorProxy):
    """A boosted-tree wildfire occurrence baseline using a binary logistic objective."""

    def __init__(self, max_depth: int = 8, eta: float = 0.05, subsample: float = 0.8, colsample_bytree: float = 0.8, num_boost_round: int = 800):
        super().__init__()
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': int(max_depth),
            'eta': float(eta),
            'subsample': float(subsample),
            'colsample_bytree': float(colsample_bytree),
        }
        self.num_boost_round = int(num_boost_round)
        self.booster = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        try:
            import xgboost as xgb
        except Exception:
            return
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.booster = xgb.train(self.params, dtrain, num_boost_round=self.num_boost_round)
        self._is_fitted = True

    def _predict_positive_proba(self, x_np: np.ndarray) -> np.ndarray:
        if self._is_fitted and self.booster is not None:
            import xgboost as xgb
            return self.booster.predict(xgb.DMatrix(x_np))
        return super()._predict_positive_proba(x_np)


def xgboost_builder(task: str, **kwargs: Any) -> nn.Module:
    require_task(task, {'classification'}, 'xgboost')
    kwargs.pop('name', None)
    return XGBoostModel(**kwargs)


__all__ = ['XGBoostModel', 'xgboost_builder']
