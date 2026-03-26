from __future__ import annotations

from typing import Any

import numpy as np
import torch.nn as nn

from ._wildfire_estimator import BinaryEstimatorProxy, require_task


class LightGBMModel(BinaryEstimatorProxy):
    """A LightGBM wildfire occurrence baseline using binary classification."""

    def __init__(self, num_leaves: int = 63, learning_rate: float = 0.05, feature_fraction: float = 0.8, bagging_fraction: float = 0.8, num_boost_round: int = 800):
        super().__init__()
        self.params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': int(num_leaves),
            'learning_rate': float(learning_rate),
            'feature_fraction': float(feature_fraction),
            'bagging_fraction': float(bagging_fraction),
            'verbose': -1,
        }
        self.num_boost_round = int(num_boost_round)
        self.booster = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        try:
            import lightgbm as lgb
        except Exception:
            return
        dtrain = lgb.Dataset(x_train, label=y_train)
        self.booster = lgb.train(self.params, dtrain, num_boost_round=self.num_boost_round)
        self._is_fitted = True

    def _predict_positive_proba(self, x_np: np.ndarray) -> np.ndarray:
        if self._is_fitted and self.booster is not None:
            return self.booster.predict(x_np)
        return super()._predict_positive_proba(x_np)


def lightgbm_builder(task: str, **kwargs: Any) -> nn.Module:
    require_task(task, {'classification'}, 'lightgbm')
    kwargs.pop('name', None)
    return LightGBMModel(**kwargs)


__all__ = ['LightGBMModel', 'lightgbm_builder']
