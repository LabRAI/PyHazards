"""Classical tree-ensemble baselines (random forest, XGBoost) behind the PyHazards model interface.

The registry builds ``nn.Module`` objects, so tree ensembles are wrapped in :class:`EstimatorModule`,
an ``nn.Module`` that holds a scikit-learn-compatible estimator. It is honest about what it is:

- it has no parameters and is **not** trained by gradient descent. Fit it with
  ``model.fit(inputs, targets)``; :class:`pyhazards.engine.Trainer` refuses to ``fit`` it (its
  ``evaluate`` and ``predict`` work on a fitted module);
- ``forward`` raises :class:`sklearn.exceptions.NotFittedError` until it is fitted (no fallback);
- ``forward`` returns torch tensors: ``(batch, n_classes)`` **log-probabilities** for
  classification (``log`` of ``predict_proba``, the convention of ``wildfire_forecasting``; columns
  follow ``estimator.classes_``, and a class with probability 0 gets ``-inf``), and
  ``(batch, n_targets)`` predictions for regression. Outputs carry no gradient;
- ``state_dict()`` is empty, so torch checkpoints do not hold the fitted trees. Use
  :meth:`EstimatorModule.save` / :meth:`EstimatorModule.load` (joblib).

Inputs are either ready-made features ``(batch, n_features)`` or, when a ``daily_layout`` is set,
the daily tensor ``(batch, time, n_dynamic + n_static + n_land_cover)`` that the
``wildfire_forecasting`` LSTM reads. Daily tensors are turned into the instance features of
Kondylatos et al. (2022) by :func:`kondylatos_instance_features`.

``random_forest`` reproduces the random forest of Kondylatos, Prapas, Ronco, Papoutsis, Camps-Valls,
Piles, Fernandez-Torres & Carvalhais, "Wildfire Danger Prediction and Understanding With Deep
Learning", GRL 49(17), e2022GL099368 (2022), as configured in the official notebook
``notebooks/RF.ipynb`` of Orion-AI-Lab/wildfire_forecasting (MIT License, Copyright (c) 2022 iprapas,
commit 2b18bcf194284d56bd4e1774f7d5315db94cba34): scikit-learn ``RandomForestClassifier(
n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=123)`` on the
35 instance features. The feature construction follows the notebook's loop. ``xgboost`` wraps
``xgboost.XGBClassifier`` with the library defaults (the paper's XGBoost hyperparameters are in its
Supporting Information, which has not been read), and needs the optional ``xgboost`` package.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

ESTIMATOR_TASKS = ("classification", "regression")

# Kondylatos et al. (2022): 10 dynamic variables, 5 static variables and the 10 Corine Land Cover
# fractions per day, concatenated in that order (as combine_dynamic_static_inputs does for the LSTM).
KONDYLATOS_DAILY_LAYOUT = (10, 5, 10)

# notebooks/RF.ipynb (cells 4 and 5).
KONDYLATOS_RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 123,
}

_SAVE_FORMAT = "pyhazards.EstimatorModule"
_SAVE_VERSION = 1

ArrayLike = Union[torch.Tensor, np.ndarray, Sequence[Any]]


def _to_numpy(values: ArrayLike) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _check_layout(layout: Sequence[int]) -> Tuple[int, int, int]:
    if len(layout) != 3 or any(int(size) < 0 for size in layout) or int(layout[0]) < 1:
        raise ValueError(
            "daily_layout must be (n_dynamic, n_static, n_land_cover) with n_dynamic >= 1, "
            f"got {tuple(layout)!r}"
        )
    return int(layout[0]), int(layout[1]), int(layout[2])


def kondylatos_instance_features(
    daily: ArrayLike,
    n_dynamic: int = 10,
    n_static: int = 5,
    n_land_cover: int = 10,
) -> np.ndarray:
    """Instance features of Kondylatos et al. (2022) from a daily tensor.

    ``daily`` is ``(batch, time, n_dynamic + n_static + n_land_cover)``, oldest day first, with the
    dynamic variables, the static variables and the land-cover fractions concatenated in that order
    (the static and land-cover columns are repeated over time, as the official
    ``combine_dynamic_static_inputs`` builds the LSTM input). Returns a float64 array
    ``(batch, 2 * n_dynamic + n_static + n_land_cover)``, 35 columns for the paper's layout::

        [nanmean of the dynamic variables over the days, dynamic variables on the last day (t-1),
         static variables, land-cover fractions]

    exactly as ``notebooks/RF.ipynb`` concatenates them (NaNs are ignored by the mean; a variable
    that is NaN on every day gives NaN, which the notebook also passes on). Static and land-cover
    values are read from the last day. The computation is done in float64 like the notebook, so
    the features match it bit for bit when ``daily`` holds the same values.
    """
    n_dynamic, n_static, n_land_cover = _check_layout((n_dynamic, n_static, n_land_cover))
    array = np.asarray(_to_numpy(daily), dtype=np.float64)
    expected = n_dynamic + n_static + n_land_cover
    if array.ndim != 3 or array.shape[1] < 1 or array.shape[2] != expected:
        raise ValueError(
            "kondylatos_instance_features expects a daily tensor of shape (batch, time, "
            f"{expected}) = (batch, time, {n_dynamic} dynamic + {n_static} static + "
            f"{n_land_cover} land cover), got shape {tuple(array.shape)}."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN columns, as in the notebook
        dynamic_mean = np.nanmean(array[:, :, :n_dynamic], axis=1)
    last_day = array[:, -1, :]  # dynamic at t-1, then static, then land cover
    return np.concatenate([dynamic_mean, last_day], axis=1)


class EstimatorModule(nn.Module):
    """``nn.Module`` holding a fitted-or-not scikit-learn-compatible estimator.

    Parameters
    ----------
    estimator:
        Any object with ``fit`` / ``predict`` (and ``predict_proba`` for classification), e.g. a
        scikit-learn or ``xgboost`` estimator.
    task:
        ``"classification"`` or ``"regression"``.
    daily_layout:
        ``(n_dynamic, n_static, n_land_cover)``. When set, 3-D input ``(batch, time, sum)`` is
        converted with :func:`kondylatos_instance_features`; 2-D input is always used as features.
    name:
        Label used in messages (the registry name).

    Saving and loading use joblib, i.e. pickle: :meth:`load` can execute arbitrary code from the
    file, so only load files you trust.
    """

    def __init__(
        self,
        estimator: Any,
        task: str = "classification",
        daily_layout: Optional[Sequence[int]] = None,
        name: Optional[str] = None,
    ):
        super().__init__()
        task = task.lower()
        if task not in ESTIMATOR_TASKS:
            raise ValueError(f"EstimatorModule supports task in {ESTIMATOR_TASKS}, got {task!r}.")
        if task == "classification" and not hasattr(estimator, "predict_proba"):
            raise ValueError(f"{type(estimator).__name__} has no predict_proba; it cannot back a classifier.")
        self.estimator = estimator
        self.task = task
        self.daily_layout = _check_layout(daily_layout) if daily_layout is not None else None
        self.name = name or type(estimator).__name__

    @property
    def custom_fit_reason(self) -> str:
        """Why :class:`pyhazards.engine.Trainer` must not ``fit`` this module (it raises with this text)."""
        return (
            f"'{self.name}' wraps a {type(self.estimator).__name__} and is not trained by gradient "
            "descent. Fit it on the whole training split with model.fit(inputs, targets), e.g. "
            "model.fit(bundle.get_split('train').inputs, bundle.get_split('train').targets); "
            "Trainer.evaluate and Trainer.predict work on the fitted model."
        )

    @property
    def is_fitted(self) -> bool:
        from sklearn.exceptions import NotFittedError  # scikit-learn is imported only when used
        from sklearn.utils.validation import check_is_fitted

        try:
            check_is_fitted(self.estimator)
        except NotFittedError:
            return False
        return True

    def features(self, inputs: ArrayLike) -> np.ndarray:
        """The 2-D feature matrix the estimator sees for ``inputs``."""
        array = _to_numpy(inputs)
        if array.ndim == 2:
            return array
        if array.ndim == 3 and self.daily_layout is not None:
            return kondylatos_instance_features(array, *self.daily_layout)
        expected = "(batch, features)"
        if self.daily_layout is not None:
            expected += " or a daily tensor (batch, time, {n})".format(n=sum(self.daily_layout))
        raise ValueError(f"{self.name} expects input shape {expected}, got shape {tuple(array.shape)}.")

    def fit(self, inputs: ArrayLike, targets: ArrayLike, **fit_kwargs: Any) -> "EstimatorModule":
        """Fit the estimator on all of ``inputs`` / ``targets`` (tensors or arrays) and return ``self``.

        Classification targets are class labels ``(batch,)`` (``(batch, 1)`` is flattened, as the
        notebook's ``y_train.ravel()``). Extra keyword arguments go to ``estimator.fit``.
        """
        features = self.features(inputs)
        labels = _to_numpy(targets)
        if self.task == "classification":
            if labels.ndim == 2 and labels.shape[1] == 1:
                labels = labels.ravel()
            if labels.ndim != 1:
                raise ValueError(
                    f"{self.name} classification targets must have shape (batch,), got shape {tuple(labels.shape)}."
                )
        if labels.shape[0] != features.shape[0]:
            raise ValueError(
                f"{self.name} got {features.shape[0]} input rows but targets of shape {tuple(labels.shape)}."
            )
        self.estimator.fit(features, labels, **fit_kwargs)
        return self

    def _fitted_features(self, inputs: ArrayLike) -> np.ndarray:
        from sklearn.exceptions import NotFittedError

        if not self.is_fitted:
            raise NotFittedError(
                f"{self.name} is not fitted yet: call model.fit(inputs, targets) before predicting."
            )
        features = self.features(inputs)
        expected = getattr(self.estimator, "n_features_in_", None)
        if expected is not None and features.shape[1] != expected:
            raise ValueError(
                f"{self.name} was fitted on {expected} features, but the input gives "
                f"{features.shape[1]} (input shape {tuple(_to_numpy(inputs).shape)})."
            )
        return features

    def predict_proba(self, inputs: ArrayLike) -> np.ndarray:
        """Class probabilities from ``estimator.predict_proba`` (classification only)."""
        if self.task != "classification":
            raise ValueError(f"{self.name} is a regressor; predict_proba needs task='classification'.")
        return self.estimator.predict_proba(self._fitted_features(inputs))

    def predict(self, inputs: ArrayLike) -> np.ndarray:
        """``estimator.predict``: class labels for classification, values for regression."""
        return self.estimator.predict(self._fitted_features(inputs))

    def forward(self, x: ArrayLike) -> torch.Tensor:
        if self.task == "classification":
            with np.errstate(divide="ignore"):
                values = np.log(self.predict_proba(x))
        else:
            values = self.predict(x)
            if values.ndim == 1:
                values = values.reshape(-1, 1)
        if isinstance(x, torch.Tensor):
            dtype = x.dtype if x.is_floating_point() else torch.get_default_dtype()
            return torch.as_tensor(values, dtype=dtype, device=x.device)
        return torch.as_tensor(values, dtype=torch.get_default_dtype())

    def save(self, path: Union[str, Path]) -> None:
        """Write the module (estimator, task, layout, name) to ``path`` with joblib."""
        import joblib

        joblib.dump(
            {
                "format": _SAVE_FORMAT,
                "version": _SAVE_VERSION,
                "name": self.name,
                "task": self.task,
                "daily_layout": self.daily_layout,
                "estimator": self.estimator,
            },
            path,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EstimatorModule":
        """Load a module written by :meth:`save`.

        This unpickles the file (joblib), which can run arbitrary code: only load trusted files.
        """
        import joblib

        payload = joblib.load(path)
        if not isinstance(payload, dict) or payload.get("format") != _SAVE_FORMAT:
            raise ValueError(f"{path} was not written by EstimatorModule.save.")
        return cls(
            payload["estimator"],
            task=payload["task"],
            daily_layout=payload["daily_layout"],
            name=payload["name"],
        )

    def extra_repr(self) -> str:
        return f"name={self.name!r}, task={self.task!r}, fitted={self.is_fitted}, estimator={self.estimator!r}"


def _estimator_task(task: str, name: str) -> str:
    task = task.lower()
    if task not in ESTIMATOR_TASKS:
        raise ValueError(f"{name} supports task in {ESTIMATOR_TASKS}, got {task!r}.")
    return task


def random_forest_builder(
    task: str,
    n_estimators: int = 100,
    max_depth: Optional[int] = 10,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: Optional[int] = 123,
    daily_layout: Optional[Sequence[int]] = KONDYLATOS_DAILY_LAYOUT,
    **kwargs: Any,
) -> EstimatorModule:
    """Random forest (Breiman, 2001) with the Kondylatos et al. (2022) notebook configuration.

    ``task="classification"`` builds ``RandomForestClassifier``, ``task="regression"``
    ``RandomForestRegressor``, both with the notebook's hyperparameters unless overridden. Other
    keyword arguments (``n_jobs``, ``class_weight``, ...) go to the scikit-learn estimator.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    kwargs.pop("name", None)
    task = _estimator_task(task, "random_forest")
    estimator_cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
    estimator = estimator_cls(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        **kwargs,
    )
    return EstimatorModule(estimator, task=task, daily_layout=daily_layout, name="random_forest")


def _import_xgboost():
    try:
        import xgboost
    except ImportError as exc:
        raise ImportError(
            "The 'xgboost' model needs the optional xgboost package: pip install 'pyhazards[xgboost]' "
            "(or the CPU-only build, pip install xgboost-cpu)."
        ) from exc
    return xgboost


def xgboost_builder(
    task: str,
    daily_layout: Optional[Sequence[int]] = KONDYLATOS_DAILY_LAYOUT,
    **kwargs: Any,
) -> EstimatorModule:
    """XGBoost (Chen & Guestrin, 2016) on the Kondylatos et al. (2022) instance features.

    Builds ``xgboost.XGBClassifier`` (``task="classification"``) or ``XGBRegressor`` with the library
    defaults; keyword arguments (``n_estimators``, ``max_depth``, ``learning_rate``, ``n_jobs``, ...)
    are passed to it. ``xgboost`` is imported only here.
    """
    kwargs.pop("name", None)
    task = _estimator_task(task, "xgboost")
    xgboost = _import_xgboost()
    estimator_cls = xgboost.XGBClassifier if task == "classification" else xgboost.XGBRegressor
    return EstimatorModule(estimator_cls(**kwargs), task=task, daily_layout=daily_layout, name="xgboost")


__all__ = [
    "ESTIMATOR_TASKS",
    "EstimatorModule",
    "KONDYLATOS_DAILY_LAYOUT",
    "KONDYLATOS_RF_PARAMS",
    "kondylatos_instance_features",
    "random_forest_builder",
    "xgboost_builder",
]
