"""Daily streamflow samples in the NeuralHydrology layout.

A streamflow sample is one basin and one target day: the ``seq_length`` days of dynamic forcings that
end on that day, ``x_d`` of shape ``(seq_length, n_dynamic)``, the basin's static attributes, ``x_s`` of
shape ``(n_static,)``, and the discharge series ``y`` of shape ``(seq_length, n_targets)`` over the same
days. Models predict every step; training uses the last ``predict_last_n`` steps and evaluation the last
one. This is the setup of Kratzert et al. (HESS 2019) and of NeuralHydrology.

:func:`build_streamflow_bundle` turns per-basin daily DataFrames and an attribute table into a
:class:`~pyhazards.datasets.base.DataBundle` the way NeuralHydrology's ``BaseDataset`` does (from
``neuralhydrology/datasetzoo/basedataset.py`` at commit ea94a40, BSD-3-Clause, Copyright (c) 2021,
NeuralHydrology; reimplemented for one input frequency without NeuralHydrology's config object):

- each period is read with a warm-up of ``seq_length - predict_last_n`` days before its start date,
  reindexed to a gap-free daily range, and targets inside the warm-up are set to NaN;
- dynamic inputs and targets are standardised with the mean and population standard deviation over all
  training basins and days (warm-up included, NaN skipped); static attributes with the mean and sample
  standard deviation over the training basins, after sorting them alphabetically by name, which is the
  order NeuralHydrology feeds them to its models;
- training samples are dropped when any dynamic input in the window is NaN or when all targets of the
  last ``predict_last_n`` steps are NaN; validation and test keep every sample with a full window, so
  that predictions exist for every day of the period (NaN inputs give NaN predictions, which the
  metrics skip).

Splits hold a :class:`StreamflowWindows` dataset; the bundle metadata carry the scaler (needed to put
predictions back into discharge units) and the feature names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset

from ..base import DataBundle, DataSplit, FeatureSpec, LabelSpec

PERIOD_NAMES = ("train", "val", "test")
Period = Tuple[str, str]


def _timestamp(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _validate_periods(periods: Mapping[str, Sequence[Any]]) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    if "train" not in periods:
        raise ValueError("periods needs a 'train' entry (start, end); its data define the scaler.")
    parsed: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for name, value in periods.items():
        if name not in PERIOD_NAMES:
            raise ValueError(f"Unknown period {name!r}; expected a subset of {PERIOD_NAMES}.")
        if len(value) != 2:
            raise ValueError(f"Period {name!r} must be (start, end), got {value!r}.")
        start, end = _timestamp(value[0]), _timestamp(value[1])
        if end < start:
            raise ValueError(f"Period {name!r} ends ({end.date()}) before it starts ({start.date()}).")
        parsed[name] = (start, end)
    return parsed


def period_frame(
    df: pd.DataFrame,
    columns: Sequence[str],
    target_variables: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    seq_length: int,
    predict_last_n: int,
    add_missing_targets: bool = False,
) -> pd.DataFrame:
    """One basin's period with warm-up, on a gap-free daily index, targets NaN before ``start``."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Basin data must be indexed by a pandas DatetimeIndex of days.")
    df = df.copy()
    if add_missing_targets:
        for target in target_variables:
            if target not in df.columns:
                df[target] = np.nan
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"The following features are not available in the data: {missing}. "
            f"These are the available features: {df.columns.tolist()}"
        )
    df = df[list(columns)].sort_index()
    warmup_start = start - pd.Timedelta(days=seq_length - predict_last_n)
    full_range = pd.date_range(start=warmup_start, end=end, freq="1D")
    df = df.loc[warmup_start: end + pd.Timedelta(days=1, seconds=-1)]
    df = df[~df.index.duplicated(keep="first")]
    df = df.reindex(pd.DatetimeIndex(full_range, name="date"))
    df.loc[df.index < start, list(target_variables)] = np.nan
    return df.astype(np.float32)


@dataclass
class StreamflowScaler:
    """Centre and scale of every input; applied as ``(value - center) / scale``."""

    dynamic_center: np.ndarray
    dynamic_scale: np.ndarray
    target_center: np.ndarray
    target_scale: np.ndarray
    static_mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    static_std: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def to_dict(self) -> Dict[str, List[float]]:
        return {key: [float(v) for v in np.asarray(value)] for key, value in self.__dict__.items()}

    def rescale_target(self, values: np.ndarray) -> np.ndarray:
        """Normalised targets or predictions ``(..., n_targets)`` back to physical units."""
        return values * self.target_scale + self.target_center


class StreamflowWindows(TorchDataset):
    """Lazy sliding windows over per-basin arrays.

    Item ``i`` is ``({"x_d": (seq_length, n_dynamic), "x_s": (n_static,)}, y)`` with ``y`` of shape
    ``(seq_length, n_targets)``; ``x_s`` is left out when there are no static attributes.
    ``sample_basin[i]`` indexes ``basins`` and ``sample_date[i]`` is the date of the window's last day.
    """

    def __init__(
        self,
        basins: Sequence[str],
        x_d: Sequence[np.ndarray],
        y: Sequence[np.ndarray],
        dates: Sequence[np.ndarray],
        x_s: Optional[np.ndarray],
        seq_length: int,
        predict_last_n: int,
        drop_invalid: bool,
    ):
        if not (len(basins) == len(x_d) == len(y) == len(dates)):
            raise ValueError("basins, x_d, y and dates must have one entry per basin.")
        if x_s is not None and x_s.shape[0] != len(basins):
            raise ValueError(f"x_s must have shape (n_basins, n_static); got {x_s.shape} for {len(basins)} basins.")
        self.basins = list(basins)
        self._x_d = [torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)) for a in x_d]
        self._y = [torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)) for a in y]
        self._dates = [np.asarray(d, dtype="datetime64[ns]") for d in dates]
        self._x_s = None if x_s is None else torch.from_numpy(np.ascontiguousarray(x_s, dtype=np.float32))
        self.seq_length = int(seq_length)
        self.predict_last_n = int(predict_last_n)
        basin_index: List[np.ndarray] = []
        end_index: List[np.ndarray] = []
        for i, (xd, yy) in enumerate(zip(x_d, y)):
            ends = np.arange(self.seq_length - 1, len(xd))
            if drop_invalid and ends.size:
                ends = ends[self._valid(np.asarray(xd), np.asarray(yy), ends)]
            basin_index.append(np.full(ends.shape, i, dtype=np.int64))
            end_index.append(ends.astype(np.int64))
        self.sample_basin = np.concatenate(basin_index) if basin_index else np.zeros(0, dtype=np.int64)
        self._end = np.concatenate(end_index) if end_index else np.zeros(0, dtype=np.int64)
        self.sample_date = (
            np.array([self._dates[b][e] for b, e in zip(self.sample_basin, self._end)], dtype="datetime64[ns]")
            if len(self._end)
            else np.zeros(0, dtype="datetime64[ns]")
        )

    def _valid(self, x_d: np.ndarray, y: np.ndarray, ends: np.ndarray) -> np.ndarray:
        # NeuralHydrology's _validate_samples: any NaN input in the window or an all-NaN target
        # over the last predict_last_n steps invalidates a training sample.
        nan_inputs = np.isnan(x_d).any(axis=1).astype(np.int64)
        cumulative = np.concatenate([[0], np.cumsum(nan_inputs)])
        window_nan = cumulative[ends + 1] - cumulative[ends + 1 - self.seq_length] > 0
        valid_target = ~np.isnan(y).all(axis=1)
        cumulative_t = np.concatenate([[0], np.cumsum(valid_target.astype(np.int64))])
        any_target = cumulative_t[ends + 1] - cumulative_t[ends + 1 - self.predict_last_n] > 0
        if self.predict_last_n == 0:
            any_target = np.ones_like(any_target, dtype=bool)
        return ~window_nan & any_target

    def __len__(self) -> int:
        return int(self._end.size)

    def __getitem__(self, item: int):
        basin = int(self.sample_basin[item])
        end = int(self._end[item]) + 1
        start = end - self.seq_length
        inputs = {"x_d": self._x_d[basin][start:end]}
        if self._x_s is not None:
            inputs["x_s"] = self._x_s[basin]
        return inputs, self._y[basin][start:end]

    def __repr__(self) -> str:
        return (
            f"StreamflowWindows(basins={len(self.basins)}, samples={len(self)}, "
            f"seq_length={self.seq_length}, predict_last_n={self.predict_last_n})"
        )


def build_streamflow_bundle(
    basin_frames: Mapping[str, pd.DataFrame],
    attributes: Optional[pd.DataFrame],
    dynamic_inputs: Sequence[str],
    static_attributes: Sequence[str],
    target_variables: Sequence[str],
    periods: Mapping[str, Sequence[Any]],
    seq_length: int,
    predict_last_n: int = 1,
    dataset_name: str = "streamflow",
    metadata: Optional[Mapping[str, Any]] = None,
) -> DataBundle:
    """Normalise, window and split per-basin daily data (see the module docstring).

    ``basin_frames`` maps basin ids to DataFrames indexed by date with the dynamic inputs and targets as
    columns; ``attributes`` is indexed by basin id. ``periods`` maps ``train`` (required), ``val`` and
    ``test`` to ``(start, end)`` dates, both inclusive.
    """
    if seq_length < 1:
        raise ValueError(f"seq_length must be positive, got {seq_length}.")
    if not 0 <= predict_last_n <= seq_length:
        raise ValueError(f"predict_last_n must be in [0, seq_length], got {predict_last_n}.")
    if not dynamic_inputs:
        raise ValueError("At least one dynamic input is required.")
    if not target_variables:
        raise ValueError("At least one target variable is required.")
    if not basin_frames:
        raise ValueError("No basins to load.")
    parsed = _validate_periods(periods)
    dynamic_inputs = list(dynamic_inputs)
    target_variables = list(target_variables)
    columns = dynamic_inputs + [t for t in target_variables if t not in dynamic_inputs]
    basins = list(basin_frames)

    frames: Dict[str, List[pd.DataFrame]] = {}
    for name, (start, end) in parsed.items():
        frames[name] = [
            period_frame(
                basin_frames[basin],
                columns,
                target_variables,
                start,
                end,
                seq_length,
                predict_last_n,
                add_missing_targets=name != "train",
            )
            for basin in basins
        ]

    train_values = np.concatenate([frame.to_numpy(dtype=np.float64) for frame in frames["train"]], axis=0)
    with np.errstate(invalid="ignore"):
        center = np.nanmean(train_values, axis=0)
        scale = np.nanstd(train_values, axis=0)
    dyn_idx = [columns.index(c) for c in dynamic_inputs]
    tgt_idx = [columns.index(c) for c in target_variables]
    for label, values in (("center", center), ("scale", scale)):
        bad = [columns[i] for i in range(len(columns)) if not np.isfinite(values[i])]
        if bad:
            raise ValueError(f"Training data give no finite {label} for {bad}; check the training period.")

    static_names = sorted(static_attributes)
    x_s = None
    static_mean = static_std = np.zeros(0)
    if static_names:
        if attributes is None:
            raise ValueError("static_attributes were requested but no attribute table was given.")
        missing = [a for a in static_names if a not in attributes.columns]
        if missing:
            raise ValueError(f"Static attributes {missing} are missing.")
        missing_basins = [b for b in basins if b not in attributes.index]
        if missing_basins:
            raise ValueError(f"Some basins are missing static attributes: {missing_basins}")
        table = attributes.loc[basins, static_names].astype(np.float64)
        static_mean = table.mean().to_numpy()
        static_std = table.std().to_numpy()
        bad = [name for name, std in zip(static_names, static_std) if not np.isfinite(std) or std == 0]
        if bad:
            raise ValueError(
                f"Static attributes {bad} have a zero or undefined standard deviation over the training basins."
            )
        x_s = ((table - static_mean) / static_std).to_numpy(dtype=np.float32)

    scaler = StreamflowScaler(
        dynamic_center=center[dyn_idx].astype(np.float32),
        dynamic_scale=scale[dyn_idx].astype(np.float32),
        target_center=center[tgt_idx].astype(np.float32),
        target_scale=scale[tgt_idx].astype(np.float32),
        static_mean=static_mean.astype(np.float32),
        static_std=static_std.astype(np.float32),
    )

    splits: Dict[str, DataSplit] = {}
    for name in parsed:
        x_d, y, dates = [], [], []
        for frame in frames[name]:
            values = frame.to_numpy(dtype=np.float32)
            x_d.append((values[:, dyn_idx] - scaler.dynamic_center) / scaler.dynamic_scale)
            y.append((values[:, tgt_idx] - scaler.target_center) / scaler.target_scale)
            dates.append(frame.index.to_numpy())
        windows = StreamflowWindows(
            basins, x_d, y, dates, x_s, seq_length, predict_last_n, drop_invalid=name == "train"
        )
        splits[name] = DataSplit(
            inputs=windows,
            targets=None,
            metadata={"period": [str(parsed[name][0].date()), str(parsed[name][1].date())]},
        )

    info: Dict[str, Any] = {
        "dataset": dataset_name,
        "source_dataset": dataset_name,
        "hazard_task": "flood.streamflow",
        "basins": basins,
        "dynamic_inputs": dynamic_inputs,
        "static_attributes": static_names,
        "target_variables": target_variables,
        "seq_length": int(seq_length),
        "predict_last_n": int(predict_last_n),
        "periods": {name: [str(s.date()), str(e.date())] for name, (s, e) in parsed.items()},
        "scaler": scaler.to_dict(),
    }
    info.update(dict(metadata or {}))
    return DataBundle(
        splits=splits,
        feature_spec=FeatureSpec(
            input_dim=len(dynamic_inputs),
            description="Daily dynamic forcings x_d (seq_length, n_dynamic) and static attributes x_s (n_static,).",
            extra={"n_dynamic": len(dynamic_inputs), "n_static": len(static_names), "seq_length": int(seq_length)},
        ),
        label_spec=LabelSpec(
            num_targets=len(target_variables),
            task_type="regression",
            description="Daily discharge over the input window (normalised; see metadata['scaler']).",
        ),
        metadata=info,
    )


def scaler_from_metadata(metadata: Mapping[str, Any]) -> StreamflowScaler:
    raw = metadata.get("scaler")
    if raw is None:
        raise ValueError("Streamflow bundles carry their scaler in metadata['scaler'].")
    return StreamflowScaler(**{key: np.asarray(value, dtype=np.float32) for key, value in raw.items()})


def read_basin_list(basins: Any) -> List[str]:
    """Basin ids from a list or from a text file with one id per line (NeuralHydrology basin files)."""
    if basins is None:
        raise ValueError("basins is required: a list of basin ids or a path to a basin file.")
    if isinstance(basins, (str, bytes)) or hasattr(basins, "__fspath__"):
        from pathlib import Path

        path = Path(basins)
        if not path.is_file():
            raise FileNotFoundError(f"Basin file not found: {path}")
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(basin) for basin in basins]
    if not ids:
        raise ValueError("basins is empty.")
    return ids


__all__ = [
    "PERIOD_NAMES",
    "StreamflowScaler",
    "StreamflowWindows",
    "build_streamflow_bundle",
    "period_frame",
    "read_basin_list",
    "scaler_from_metadata",
]
