from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch

from .base import DataBundle, DataSplit, Dataset, FeatureSpec, LabelSpec

CauseMode = Literal["paper5", "keep_all"]
Region = Literal["US", "CA"]
WeeklyFeatures = Literal["counts", "counts+time"]

PAPER5_CAUSES = [
    "Debris and open burning",
    "Natural",
    "Arson/incendiarism",
    "Equipment and vehicle use",
    "Recreation and ceremony",
]

CAUSE_SYNONYMS = {
    "Debris/open burning": "Debris and open burning",
    "Debris and Open Burning": "Debris and open burning",
    "Arson": "Arson/incendiarism",
    "Equipment/vehicle use": "Equipment and vehicle use",
    "Recreation/ceremony": "Recreation and ceremony",
}

SIZE_GROUPS = ["A", "B", "C", "D", "EFG"]


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "FPA-FOD dataset support requires pandas. Install pandas or xarray's pandas dependency first."
        ) from exc
    return pd


def _minmax_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mins = np.nanmin(x, axis=0)
    maxs = np.nanmax(x, axis=0)
    maxs = np.where(maxs == mins, mins + 1.0, maxs)
    return mins, maxs


def _minmax_apply(x: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    return (x - mins) / (maxs - mins)


def _stratified_split_indices(
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0.")

    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for class_id in np.unique(y):
        idx = np.where(y == class_id)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(train_ratio * n))
        n_val = int(round(val_ratio * n))
        n_test = max(0, n - n_train - n_val)

        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train : n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val : n_train + n_val + n_test].tolist())

    train = np.array(train_idx, dtype=np.int64)
    val = np.array(val_idx, dtype=np.int64)
    test = np.array(test_idx, dtype=np.int64)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _chronological_split_indices(
    n: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0.")

    n_train = int(math.floor(train_ratio * n))
    n_val = int(math.floor(val_ratio * n))
    n_test = n - n_train - n_val

    train = np.arange(0, n_train, dtype=np.int64)
    val = np.arange(n_train, n_train + n_val, dtype=np.int64)
    test = np.arange(n_train + n_val, n_train + n_val + n_test, dtype=np.int64)
    return train, val, test


def _load_fpa_fod_table(path: str):
    pd = _require_pandas()

    if not os.path.exists(path):
        raise FileNotFoundError(f"Data path not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in {".sqlite", ".db"}:
        import sqlite3

        con = sqlite3.connect(path)
        try:
            return pd.read_sql_query("SELECT * FROM Fires", con)
        finally:
            con.close()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension for FPA-FOD data: {ext}")


def _coerce_required_columns(df, required: List[str]):
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def _encode_states(states) -> Tuple[np.ndarray, Dict[str, int]]:
    values = sorted(states.dropna().astype(str).unique().tolist())
    mapping = {value: index for index, value in enumerate(values)}
    encoded = states.astype(str).map(mapping).astype("int64").to_numpy()
    return encoded, mapping


def _normalize_cause_strings(values):
    return values.astype(str).str.strip().map(lambda value: CAUSE_SYNONYMS.get(value, value))


def _impute_numeric(df, columns: List[str]):
    pd = _require_pandas()
    filled = df.copy()
    medians: Dict[str, float] = {}
    for column in columns:
        series = pd.to_numeric(filled[column], errors="coerce")
        median = float(series.median())
        if math.isnan(median):
            median = 0.0
        medians[column] = median
        filled[column] = series.fillna(median)
    return filled, medians


def _micro_tabular_df(seed: int = 1337, n: int = 200):
    pd = _require_pandas()

    rng = np.random.default_rng(seed)
    states = np.array(["CA", "TX", "FL", "NY", "WA", "CO"])
    causes = np.array(PAPER5_CAUSES)

    years = rng.integers(2010, 2019, size=n)
    discovery_doy = rng.integers(1, 366, size=n)
    discovery_time = rng.integers(0, 2400, size=n)
    containment_doy = np.clip(discovery_doy + rng.integers(0, 30, size=n), 1, 366)
    containment_time = rng.integers(0, 2400, size=n)

    state = rng.choice(states, size=n, replace=True)
    latitude = rng.uniform(25.0, 49.0, size=n)
    longitude = rng.uniform(-124.0, -67.0, size=n)
    california_mask = state == "CA"
    latitude[california_mask] = rng.uniform(32.0, 42.0, size=california_mask.sum())
    longitude[california_mask] = rng.uniform(-124.5, -114.0, size=california_mask.sum())

    cause = rng.choice(causes, size=n, replace=True)
    size_class = rng.choice(
        ["A", "B", "C", "D", "E", "F", "G"],
        size=n,
        p=[0.38, 0.42, 0.12, 0.04, 0.02, 0.01, 0.01],
    )

    return pd.DataFrame(
        {
            "FIRE_YEAR": years,
            "STATE": state,
            "DISCOVERY_DOY": discovery_doy,
            "DISCOVERY_TIME": discovery_time,
            "CONT_DOY": containment_doy,
            "CONT_TIME": containment_time,
            "LATITUDE": latitude,
            "LONGITUDE": longitude,
            "NWCG_GENERAL_CAUSE": cause,
            "FIRE_SIZE_CLASS": size_class,
        }
    )


def _micro_weekly_counts(seed: int = 1337, weeks: int = 120, region: Region = "US"):
    pd = _require_pandas()

    rng = np.random.default_rng(seed)
    dates = pd.date_range(pd.Timestamp("2016-01-04"), periods=weeks, freq="W-MON")
    time = np.arange(weeks)
    base = 50 + 20 * np.sin(2 * np.pi * time / 52.0)
    if region == "CA":
        base = base * 1.3

    a = np.maximum(0, base + rng.normal(0, 8, size=weeks)).astype(int)
    b = np.maximum(0, base * 0.8 + rng.normal(0, 7, size=weeks)).astype(int)
    c = np.maximum(0, base * 0.2 + rng.normal(0, 3, size=weeks)).astype(int)
    d = np.maximum(0, base * 0.05 + rng.normal(0, 2, size=weeks)).astype(int)
    efg = np.maximum(0, base * 0.03 + rng.normal(0, 2, size=weeks)).astype(int)

    return pd.DataFrame({"week_start": dates, "A": a, "B": b, "C": c, "D": d, "EFG": efg})


class FPAFODTabularDataset(Dataset):
    """Incident-level tabular dataset for wildfire cause or size classification."""

    name = "fpa_fod_tabular"

    def __init__(
        self,
        task: Literal["cause", "size"] = "cause",
        region: Region = "US",
        cause_mode: CauseMode = "paper5",
        data_path: Optional[str] = None,
        micro: bool = False,
        normalize: bool = False,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        seed: int = 1337,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(cache_dir=cache_dir)
        self.task = task
        self.region = region
        self.cause_mode = cause_mode
        self.data_path = data_path
        self.micro = micro
        self.normalize = normalize
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def _load(self) -> DataBundle:
        if self.micro:
            df = _micro_tabular_df(seed=self.seed)
            source = "micro_synthetic"
        else:
            if not self.data_path:
                raise ValueError("data_path is required when micro=False")
            df = _load_fpa_fod_table(self.data_path)
            source = self.data_path

        required = [
            "FIRE_YEAR",
            "STATE",
            "DISCOVERY_DOY",
            "DISCOVERY_TIME",
            "CONT_DOY",
            "CONT_TIME",
            "LATITUDE",
            "LONGITUDE",
            "NWCG_GENERAL_CAUSE",
            "FIRE_SIZE_CLASS",
        ]
        df = _coerce_required_columns(df, required)

        if self.region == "CA":
            df = df[df["STATE"].astype(str) == "CA"].copy()

        df, numeric_impute = _impute_numeric(
            df,
            columns=[
                "FIRE_YEAR",
                "DISCOVERY_DOY",
                "DISCOVERY_TIME",
                "CONT_DOY",
                "CONT_TIME",
                "LATITUDE",
                "LONGITUDE",
            ],
        )

        state_encoded, state_mapping = _encode_states(df["STATE"])
        numeric_features = [
            "FIRE_YEAR",
            "DISCOVERY_DOY",
            "DISCOVERY_TIME",
            "CONT_DOY",
            "CONT_TIME",
            "LATITUDE",
            "LONGITUDE",
        ]
        feature_names = numeric_features + ["STATE_ID"]
        x_numeric = df[numeric_features].to_numpy(dtype=np.float32)
        x = np.concatenate([x_numeric, state_encoded.astype(np.float32).reshape(-1, 1)], axis=1)

        metadata: Dict[str, Any] = {
            "dataset": self.name,
            "source": source,
            "region": self.region,
            "task": self.task,
            "micro": self.micro,
            "seed": self.seed,
            "state_mapping": state_mapping,
            "numeric_impute_medians": numeric_impute,
        }

        if self.task == "cause":
            causes = _normalize_cause_strings(df["NWCG_GENERAL_CAUSE"])
            if self.cause_mode == "paper5":
                mask = causes.isin(PAPER5_CAUSES)
                metadata["dropped_non_paper5_causes"] = int((~mask).sum())
                if int(mask.sum()) == 0:
                    raise RuntimeError("cause_mode='paper5' kept zero rows after cause normalization.")
                causes = causes.loc[mask]
                x = x[mask.to_numpy()]

            classes = sorted(causes.unique().tolist())
            label_mapping = {label: index for index, label in enumerate(classes)}
            y = causes.map(label_mapping).astype("int64").to_numpy()
            train_idx, val_idx, test_idx = _stratified_split_indices(
                y=y,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                seed=self.seed,
            )
            label_spec = LabelSpec(
                num_targets=len(classes),
                task_type="classification",
                description="NWCG_GENERAL_CAUSE mapped to class ids.",
                extra={"classes": classes, "label_mapping": label_mapping},
            )
            metadata["label_mapping"] = label_mapping
        elif self.task == "size":
            grouped = df["FIRE_SIZE_CLASS"].astype(str).str.strip().replace({"E": "EFG", "F": "EFG", "G": "EFG"})
            mask = grouped.isin(SIZE_GROUPS)
            metadata["dropped_unknown_size_class"] = int((~mask).sum())
            grouped = grouped.loc[mask]
            x = x[mask.to_numpy()]
            label_mapping = {label: index for index, label in enumerate(SIZE_GROUPS)}
            y = grouped.map(label_mapping).astype("int64").to_numpy()
            train_idx, val_idx, test_idx = _stratified_split_indices(
                y=y,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                seed=self.seed,
            )
            label_spec = LabelSpec(
                num_targets=len(SIZE_GROUPS),
                task_type="classification",
                description="FIRE_SIZE_CLASS grouped into A/B/C/D/EFG and mapped to class ids.",
                extra={"classes": SIZE_GROUPS, "label_mapping": label_mapping},
            )
            metadata["label_mapping"] = label_mapping
        else:
            raise ValueError(f"Unsupported tabular task: {self.task}")

        if self.normalize:
            mins, maxs = _minmax_fit(x[train_idx])
            x = _minmax_apply(x, mins, maxs).astype(np.float32)
            metadata["normalization"] = {"mins": mins.tolist(), "maxs": maxs.tolist()}
        else:
            metadata["normalization"] = None

        splits = {
            "train": DataSplit(
                inputs=torch.as_tensor(x[train_idx], dtype=torch.float32),
                targets=torch.as_tensor(y[train_idx], dtype=torch.long),
                metadata={"source": source},
            ),
            "val": DataSplit(
                inputs=torch.as_tensor(x[val_idx], dtype=torch.float32),
                targets=torch.as_tensor(y[val_idx], dtype=torch.long),
                metadata={"source": source},
            ),
            "test": DataSplit(
                inputs=torch.as_tensor(x[test_idx], dtype=torch.float32),
                targets=torch.as_tensor(y[test_idx], dtype=torch.long),
                metadata={"source": source},
            ),
        }

        return DataBundle(
            splits=splits,
            feature_spec=FeatureSpec(
                input_dim=int(splits["train"].inputs.shape[1]),
                description="Incident-level FPA-FOD features for classification.",
                extra={"feature_names": feature_names, "dtype": "float32"},
            ),
            label_spec=label_spec,
            metadata=metadata,
        )


class FPAFODWeeklyDataset(Dataset):
    """Weekly count forecasting dataset derived from FPA-FOD incident records."""

    name = "fpa_fod_weekly"

    def __init__(
        self,
        region: Region = "US",
        data_path: Optional[str] = None,
        micro: bool = False,
        lookback_weeks: int = 50,
        features: WeeklyFeatures = "counts",
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        seed: int = 1337,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(cache_dir=cache_dir)
        self.region = region
        self.data_path = data_path
        self.micro = micro
        self.lookback_weeks = lookback_weeks
        self.features = features
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def _weekly_table(self):
        pd = _require_pandas()

        if self.micro:
            return _micro_weekly_counts(seed=self.seed, region=self.region), "micro_synthetic"

        if not self.data_path:
            raise ValueError("data_path is required when micro=False")

        df = _load_fpa_fod_table(self.data_path)
        required = ["FIRE_YEAR", "STATE", "DISCOVERY_DOY", "FIRE_SIZE_CLASS"]
        df = _coerce_required_columns(df, required)

        if self.region == "CA":
            df = df[df["STATE"].astype(str) == "CA"].copy()

        fire_year = pd.to_numeric(df["FIRE_YEAR"], errors="coerce")
        discovery_doy = pd.to_numeric(df["DISCOVERY_DOY"], errors="coerce").fillna(1)
        base = pd.to_datetime(fire_year.astype("Int64").astype(str) + "-01-01", errors="coerce")
        discovery_dt = base + pd.to_timedelta(discovery_doy.astype(int) - 1, unit="D")
        week_start = discovery_dt.dt.to_period("W-MON").dt.start_time
        size_class = df["FIRE_SIZE_CLASS"].astype(str).str.strip().replace({"E": "EFG", "F": "EFG", "G": "EFG"})
        size_class = size_class.where(size_class.isin(SIZE_GROUPS), other=np.nan)

        weekly = (
            df.assign(_week_start=week_start, _size=size_class)
            .dropna(subset=["_week_start", "_size"])
            .groupby(["_week_start", "_size"])
            .size()
            .unstack("_size", fill_value=0)
            .reset_index()
            .rename(columns={"_week_start": "week_start"})
            .sort_values("week_start")
            .reset_index(drop=True)
        )
        for size_group in SIZE_GROUPS:
            if size_group not in weekly.columns:
                weekly[size_group] = 0
        return weekly, self.data_path

    def _load(self) -> DataBundle:
        weekly, source = self._weekly_table()
        lookback = int(self.lookback_weeks)
        if len(weekly) <= lookback:
            raise ValueError(f"Not enough weeks ({len(weekly)}) for lookback={lookback}")

        counts = weekly[SIZE_GROUPS].to_numpy(dtype=np.float32)
        if self.features == "counts":
            features = counts
            feature_names = list(SIZE_GROUPS)
        elif self.features == "counts+time":
            week_of_year = weekly["week_start"].dt.isocalendar().week.to_numpy(dtype=np.float32)
            sin = np.sin(2 * np.pi * week_of_year / 52.0).reshape(-1, 1).astype(np.float32)
            cos = np.cos(2 * np.pi * week_of_year / 52.0).reshape(-1, 1).astype(np.float32)
            features = np.concatenate([counts, sin, cos], axis=1)
            feature_names = list(SIZE_GROUPS) + ["woy_sin", "woy_cos"]
        else:
            raise ValueError(f"Unsupported feature mode: {self.features}")

        x_windows: List[np.ndarray] = []
        y_targets: List[np.ndarray] = []
        sample_weeks: List[str] = []
        for index in range(lookback, len(weekly)):
            x_windows.append(features[index - lookback : index])
            y_targets.append(counts[index])
            sample_weeks.append(str(weekly.loc[index, "week_start"]))

        x = np.stack(x_windows, axis=0).astype(np.float32)
        y = np.stack(y_targets, axis=0).astype(np.float32)
        train_idx, val_idx, test_idx = _chronological_split_indices(
            n=int(x.shape[0]),
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
        )

        splits = {
            "train": DataSplit(
                inputs=torch.as_tensor(x[train_idx], dtype=torch.float32),
                targets=torch.as_tensor(y[train_idx], dtype=torch.float32),
                metadata={"source": source, "region": self.region},
            ),
            "val": DataSplit(
                inputs=torch.as_tensor(x[val_idx], dtype=torch.float32),
                targets=torch.as_tensor(y[val_idx], dtype=torch.float32),
                metadata={"source": source, "region": self.region},
            ),
            "test": DataSplit(
                inputs=torch.as_tensor(x[test_idx], dtype=torch.float32),
                targets=torch.as_tensor(y[test_idx], dtype=torch.float32),
                metadata={"source": source, "region": self.region},
            ),
        }

        return DataBundle(
            splits=splits,
            feature_spec=FeatureSpec(
                input_dim=int(splits["train"].inputs.shape[-1]),
                description="Weekly FPA-FOD feature windows for next-week forecasting.",
                extra={
                    "feature_names": feature_names,
                    "lookback_weeks": lookback,
                    "dtype": "float32",
                    "region": self.region,
                },
            ),
            label_spec=LabelSpec(
                num_targets=len(SIZE_GROUPS),
                task_type="regression",
                description="Next-week counts per size group (A, B, C, D, EFG).",
                extra={"targets": list(SIZE_GROUPS), "dtype": "float32"},
            ),
            metadata={
                "dataset": self.name,
                "source": source,
                "region": self.region,
                "micro": self.micro,
                "seed": self.seed,
                "lookback_weeks": lookback,
                "features_mode": self.features,
                "week_start_for_each_sample": sample_weeks,
            },
        )


def _default_dataset_path() -> Path:
    return Path("data/fpa_fod.sqlite")


def build_tabular_inspection_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pyhazards.datasets.fpa_fod_tabular.inspection",
        description="Inspect the FPA-FOD tabular dataset and print split/label summary.",
    )
    parser.add_argument("--path", default=str(_default_dataset_path()), help="Path to the FPA-FOD sqlite/csv/parquet file.")
    parser.add_argument("--task", choices=["cause", "size"], default="cause", help="Tabular classification target.")
    parser.add_argument("--region", choices=["US", "CA"], default="US", help="Geographic subset.")
    parser.add_argument("--cause-mode", choices=["paper5", "keep_all"], default="paper5", help="Cause label mapping mode.")
    parser.add_argument("--micro", action="store_true", help="Use deterministic synthetic data instead of a real file.")
    parser.add_argument("--normalize", action="store_true", help="Apply train-fit min/max normalization.")
    return parser


def build_weekly_inspection_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pyhazards.datasets.fpa_fod_weekly.inspection",
        description="Inspect the FPA-FOD weekly forecasting dataset and print split/shape summary.",
    )
    parser.add_argument("--path", default=str(_default_dataset_path()), help="Path to the FPA-FOD sqlite/csv/parquet file.")
    parser.add_argument("--region", choices=["US", "CA"], default="US", help="Geographic subset.")
    parser.add_argument("--features", choices=["counts", "counts+time"], default="counts", help="Weekly feature mode.")
    parser.add_argument("--lookback-weeks", type=int, default=50, help="Sequence length used to predict the next week.")
    parser.add_argument("--micro", action="store_true", help="Use deterministic synthetic data instead of a real file.")
    return parser


def inspect_fpa_fod_tabular(argv: list[str] | None = None) -> int:
    args = build_tabular_inspection_parser().parse_args(argv)
    dataset = FPAFODTabularDataset(
        task=args.task,
        region=args.region,
        cause_mode=args.cause_mode,
        data_path=args.path,
        micro=args.micro,
        normalize=args.normalize,
    )
    bundle = dataset.load()
    print(f"[OK] Loaded dataset: {dataset.name}")
    print(f"[OK] Source: {bundle.metadata['source']}")
    print(f"[OK] Task: {bundle.metadata['task']}")
    print(f"[OK] Input dim: {bundle.feature_spec.input_dim}")
    print(f"[OK] Num targets: {bundle.label_spec.num_targets}")
    for split_name, split in bundle.splits.items():
        print(f"[OK] {split_name}: inputs={tuple(split.inputs.shape)} targets={tuple(split.targets.shape)}")
    mapping = bundle.metadata.get("label_mapping")
    if mapping:
        print(f"[OK] Label mapping: {mapping}")
    return 0


def inspect_fpa_fod_weekly(argv: list[str] | None = None) -> int:
    args = build_weekly_inspection_parser().parse_args(argv)
    dataset = FPAFODWeeklyDataset(
        region=args.region,
        data_path=args.path,
        micro=args.micro,
        features=args.features,
        lookback_weeks=args.lookback_weeks,
    )
    bundle = dataset.load()
    print(f"[OK] Loaded dataset: {dataset.name}")
    print(f"[OK] Source: {bundle.metadata['source']}")
    print(f"[OK] Lookback weeks: {bundle.metadata['lookback_weeks']}")
    print(f"[OK] Feature mode: {bundle.metadata['features_mode']}")
    print(f"[OK] Input dim: {bundle.feature_spec.input_dim}")
    print(f"[OK] Num targets: {bundle.label_spec.num_targets}")
    for split_name, split in bundle.splits.items():
        print(f"[OK] {split_name}: inputs={tuple(split.inputs.shape)} targets={tuple(split.targets.shape)}")
    return 0


__all__ = [
    "CAUSE_SYNONYMS",
    "PAPER5_CAUSES",
    "SIZE_GROUPS",
    "FPAFODTabularDataset",
    "FPAFODWeeklyDataset",
    "build_tabular_inspection_parser",
    "build_weekly_inspection_parser",
    "inspect_fpa_fod_tabular",
    "inspect_fpa_fod_weekly",
]
