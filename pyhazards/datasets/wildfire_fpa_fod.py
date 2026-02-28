"""
FPA-FOD Wildfire datasets for PyHazards.

Implements:
  - FPAFODWildfireTabular: per-incident tabular classification for (cause | size)
  - FPAFODWildfireWeekly: weekly time-series forecasting (lookback -> next-week counts by size group)

Contract (PyHazards-canonical):
  - These datasets are DataBundle-style (NOT torch Dataset):
      ds.load() -> DataBundle
      bundle.splits["train"/"val"/"test"] are DataSplit objects with:
        - .inputs  : torch.FloatTensor
        - .targets : torch.LongTensor (classification) or torch.FloatTensor (regression)
  - The engine.Trainer expects DataBundle.get_split(name) -> DataSplit.

Non-negotiable:
  - Includes deterministic MICRO synthetic datasets for unit tests.

Note:
  - Real data must be provided by the user (licensing/size). Supported: .sqlite/.db (Fires table), .csv, .parquet.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .base import DataBundle, DataSplit, Dataset, FeatureSpec, LabelSpec

# -----------------------------------------------------------------------------
# Types / constants
# -----------------------------------------------------------------------------

CauseMode = Literal["paper5", "keep_all"]
Region = Literal["US", "CA"]
WeeklyFeatures = Literal["counts", "counts+time"]

# These strings must match NWCG_GENERAL_CAUSE values (with synonyms mapped below)
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


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _minmax_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mins = np.nanmin(x, axis=0)
    maxs = np.nanmax(x, axis=0)
    maxs = np.where(maxs == mins, mins + 1.0, maxs)
    return mins, maxs


def _minmax_apply(x: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    return (x - mins) / (maxs - mins)


def _stratified_split_indices(
    y: np.ndarray, train_ratio: float, val_ratio: float, test_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6
    rng = np.random.default_rng(seed)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    classes = np.unique(y)
    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        n = len(idx_c)
        n_train = int(round(train_ratio * n))
        n_val = int(round(val_ratio * n))
        n_test = max(0, n - n_train - n_val)

        train_idx.extend(idx_c[:n_train].tolist())
        val_idx.extend(idx_c[n_train:n_train + n_val].tolist())
        test_idx.extend(idx_c[n_train + n_val:n_train + n_val + n_test].tolist())

    train_idx = np.array(train_idx, dtype=np.int64)
    val_idx = np.array(val_idx, dtype=np.int64)
    test_idx = np.array(test_idx, dtype=np.int64)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx


def _chronological_split_indices(
    n: int, train_ratio: float, val_ratio: float, test_ratio: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6
    n_train = int(math.floor(train_ratio * n))
    n_val = int(math.floor(val_ratio * n))
    n_test = n - n_train - n_val

    train_idx = np.arange(0, n_train, dtype=np.int64)
    val_idx = np.arange(n_train, n_train + n_val, dtype=np.int64)
    test_idx = np.arange(n_train + n_val, n_train + n_val + n_test, dtype=np.int64)
    return train_idx, val_idx, test_idx


def _encode_states(states: pd.Series) -> Tuple[np.ndarray, Dict[str, int]]:
    uniq = sorted([s for s in states.dropna().astype(str).unique().tolist()])
    mapping = {s: i for i, s in enumerate(uniq)}
    enc = states.astype(str).map(mapping).astype("int64").to_numpy()
    return enc, mapping


def _normalize_cause_strings(x: pd.Series) -> pd.Series:
    s = x.astype(str).str.strip()
    s = s.map(lambda v: CAUSE_SYNONYMS.get(v, v))
    return s


def _coerce_required_columns(df: pd.DataFrame, required: List[str]) -> pd.DataFrame:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Have: {list(df.columns)[:40]}...")
    return df


def _impute_numeric(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    medians: Dict[str, float] = {}
    out = df.copy()
    for c in cols:
        med = float(pd.to_numeric(out[c], errors="coerce").median())
        if math.isnan(med):
            med = 0.0
        medians[c] = med
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(med)
    return out, medians


# -----------------------------------------------------------------------------
# MICRO synthetic data
# -----------------------------------------------------------------------------

def _micro_tabular_df(seed: int = 1337, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    states = np.array(["CA", "TX", "FL", "NY", "WA", "CO"])
    causes = np.array(PAPER5_CAUSES)

    years = rng.integers(2010, 2019, size=n)
    doy = rng.integers(1, 366, size=n)
    disc_time = rng.integers(0, 2400, size=n)
    cont_doy = np.clip(doy + rng.integers(0, 30, size=n), 1, 366)
    cont_time = rng.integers(0, 2400, size=n)

    st = rng.choice(states, size=n, replace=True)
    lat = rng.uniform(25.0, 49.0, size=n)
    lon = rng.uniform(-124.0, -67.0, size=n)
    ca_mask = (st == "CA")
    lat[ca_mask] = rng.uniform(32.0, 42.0, size=ca_mask.sum())
    lon[ca_mask] = rng.uniform(-124.5, -114.0, size=ca_mask.sum())

    cause = rng.choice(causes, size=n, replace=True)
    size_raw = rng.choice(
        ["A", "B", "C", "D", "E", "F", "G"],
        size=n,
        p=[0.38, 0.42, 0.12, 0.04, 0.02, 0.01, 0.01],
    )

    return pd.DataFrame({
        "FIRE_YEAR": years,
        "STATE": st,
        "DISCOVERY_DOY": doy,
        "DISCOVERY_TIME": disc_time,
        "CONT_DOY": cont_doy,
        "CONT_TIME": cont_time,
        "LATITUDE": lat,
        "LONGITUDE": lon,
        "NWCG_GENERAL_CAUSE": cause,
        "FIRE_SIZE_CLASS": size_raw,
    })


def _micro_weekly_counts(seed: int = 1337, weeks: int = 120, region: Region = "US") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2016-01-04")
    dates = pd.date_range(start, periods=weeks, freq="W-MON")

    t = np.arange(weeks)
    base = 50 + 20 * np.sin(2 * np.pi * t / 52.0)
    if region == "CA":
        base = base * 1.3

    A = np.maximum(0, base + rng.normal(0, 8, size=weeks)).astype(int)
    B = np.maximum(0, base * 0.8 + rng.normal(0, 7, size=weeks)).astype(int)
    C = np.maximum(0, base * 0.2 + rng.normal(0, 3, size=weeks)).astype(int)
    D = np.maximum(0, base * 0.05 + rng.normal(0, 2, size=weeks)).astype(int)
    EFG = np.maximum(0, base * 0.03 + rng.normal(0, 2, size=weeks)).astype(int)

    return pd.DataFrame({"week_start": dates, "A": A, "B": B, "C": C, "D": D, "EFG": EFG})


# -----------------------------------------------------------------------------
# Real-data loader
# -----------------------------------------------------------------------------

def _load_fpa_fod_any(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data path not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext in [".sqlite", ".db"]:
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

    raise ValueError(f"Unsupported file extension: {ext} for path {path}")


# -----------------------------------------------------------------------------
# Dataset: Tabular classification
# -----------------------------------------------------------------------------

class FPAFODWildfireTabular(Dataset):
    name = "wildfire_fpa_fod_tabular"

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
            df = _micro_tabular_df(seed=self.seed, n=200)
            source = "micro_synthetic"
        else:
            if not self.data_path:
                raise ValueError("data_path is required when micro=False")
            df = _load_fpa_fod_any(self.data_path)
            source = self.data_path

        required = [
            "FIRE_YEAR", "STATE", "DISCOVERY_DOY", "DISCOVERY_TIME",
            "CONT_DOY", "CONT_TIME", "LATITUDE", "LONGITUDE",
            "NWCG_GENERAL_CAUSE", "FIRE_SIZE_CLASS",
        ]
        df = _coerce_required_columns(df, required)

        if self.region == "CA":
            df = df[df["STATE"].astype(str) == "CA"].copy()

        df, numeric_impute = _impute_numeric(
            df,
            cols=["FIRE_YEAR", "DISCOVERY_DOY", "DISCOVERY_TIME", "CONT_DOY", "CONT_TIME", "LATITUDE", "LONGITUDE"],
        )

        state_enc, state_mapping = _encode_states(df["STATE"])
        feature_cols_numeric = ["FIRE_YEAR", "DISCOVERY_DOY", "DISCOVERY_TIME", "CONT_DOY", "CONT_TIME", "LATITUDE", "LONGITUDE"]

        x_num = df[feature_cols_numeric].to_numpy(dtype=np.float32)
        x_state = state_enc.astype(np.float32).reshape(-1, 1)
        x = np.concatenate([x_num, x_state], axis=1).astype(np.float32)

        feature_names = feature_cols_numeric + ["STATE_ID"]

        metadata: Dict[str, Any] = {
            "dataset": self.name,
            "source": source,
            "region": self.region,
            "task": self.task,
            "micro": bool(self.micro),
            "seed": int(self.seed),
            "state_mapping": state_mapping,
            "numeric_impute_medians": numeric_impute,
        }

        # ---- Targets + split indices ----
        if self.task == "cause":
            causes = _normalize_cause_strings(df["NWCG_GENERAL_CAUSE"])

            if self.cause_mode == "paper5":
                mask = causes.isin(PAPER5_CAUSES)
                metadata["dropped_non_paper5_causes"] = int((~mask).sum())
                if int(mask.sum()) == 0:
                    top = causes.value_counts().head(20).to_dict()
                    raise RuntimeError(
                        "cause_mode=paper5 kept 0 rows. Your NWCG_GENERAL_CAUSE strings don't match PAPER5_CAUSES.\n"
                        f"Top values: {top}\n"
                        f"Expected one of: {PAPER5_CAUSES}\n"
                        "Fix CAUSE_SYNONYMS / PAPER5_CAUSES or run with cause_mode='keep_all'."
                    )
                df = df.loc[mask].copy()
                causes = causes.loc[mask]
                x = x[mask.to_numpy()]

            classes = sorted(causes.unique().tolist())
            label_mapping = {c: i for i, c in enumerate(classes)}
            y = causes.map(label_mapping).astype("int64").to_numpy()

            metadata["label_mapping"] = label_mapping
            train_idx, val_idx, test_idx = _stratified_split_indices(
                y=y, train_ratio=self.train_ratio, val_ratio=self.val_ratio, test_ratio=self.test_ratio, seed=self.seed
            )

            label_spec = LabelSpec(
                num_targets=len(classes),
                task_type="classification",
                description="NWCG_GENERAL_CAUSE mapped to integer class IDs.",
                extra={"classes": classes, "label_mapping": label_mapping},
            )

        elif self.task == "size":
            size_raw = df["FIRE_SIZE_CLASS"].astype(str).str.strip()
            size_grp = size_raw.replace({"E": "EFG", "F": "EFG", "G": "EFG"})
            mask = size_grp.isin(SIZE_GROUPS)
            metadata["dropped_unknown_size_class"] = int((~mask).sum())

            size_grp = size_grp.loc[mask]
            x = x[mask.to_numpy()]

            label_mapping = {c: i for i, c in enumerate(SIZE_GROUPS)}
            y = size_grp.map(label_mapping).astype("int64").to_numpy()

            metadata["label_mapping"] = label_mapping
            train_idx, val_idx, test_idx = _stratified_split_indices(
                y=y, train_ratio=self.train_ratio, val_ratio=self.val_ratio, test_ratio=self.test_ratio, seed=self.seed
            )

            label_spec = LabelSpec(
                num_targets=len(SIZE_GROUPS),
                task_type="classification",
                description="FIRE_SIZE_CLASS grouped to A/B/C/D/EFG and mapped to integer class IDs.",
                extra={"classes": SIZE_GROUPS, "label_mapping": label_mapping},
            )

        else:
            raise ValueError(f"Unknown task: {self.task}")

        # ---- Normalization (fit on train only) ----
        norm_stats = None
        if self.normalize:
            mins, maxs = _minmax_fit(x[train_idx])
            x = _minmax_apply(x, mins, maxs).astype(np.float32)
            norm_stats = {"mins": mins.tolist(), "maxs": maxs.tolist()}
        metadata["normalization"] = norm_stats

        # ---- Build canonical splits as DataSplit(torch tensors) ----
        X_train = torch.as_tensor(x[train_idx], dtype=torch.float32)
        y_train = torch.as_tensor(y[train_idx], dtype=torch.long)

        X_val = torch.as_tensor(x[val_idx], dtype=torch.float32)
        y_val = torch.as_tensor(y[val_idx], dtype=torch.long)

        X_test = torch.as_tensor(x[test_idx], dtype=torch.float32)
        y_test = torch.as_tensor(y[test_idx], dtype=torch.long)

        splits: Dict[str, DataSplit] = {
            "train": DataSplit(inputs=X_train, targets=y_train, metadata={"source": source}),
            "val": DataSplit(inputs=X_val, targets=y_val, metadata={"source": source}),
            "test": DataSplit(inputs=X_test, targets=y_test, metadata={"source": source}),
        }

        feature_spec = FeatureSpec(
            input_dim=int(X_train.shape[1]),
            description="FPA FOD tabular features (engineered from incident records).",
            extra={"feature_names": feature_names, "dtype": "float32"},
        )

        return DataBundle(
            splits=splits,
            feature_spec=feature_spec,
            label_spec=label_spec,
            metadata=metadata,
        )


# -----------------------------------------------------------------------------
# Dataset: Weekly forecasting
# -----------------------------------------------------------------------------

class FPAFODWildfireWeekly(Dataset):
    name = "wildfire_fpa_fod_weekly"

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

    def _load(self) -> DataBundle:
        if self.micro:
            wk = _micro_weekly_counts(seed=self.seed, weeks=120, region=self.region)
            source = "micro_synthetic"
        else:
            if not self.data_path:
                raise ValueError("data_path is required when micro=False")
            df = _load_fpa_fod_any(self.data_path)
            source = self.data_path

            required = ["FIRE_YEAR", "STATE", "DISCOVERY_DOY", "FIRE_SIZE_CLASS"]
            df = _coerce_required_columns(df, required)

            if self.region == "CA":
                df = df[df["STATE"].astype(str) == "CA"].copy()

            year = pd.to_numeric(df["FIRE_YEAR"], errors="coerce")
            doy = pd.to_numeric(df["DISCOVERY_DOY"], errors="coerce")
            base = pd.to_datetime(year.astype("Int64").astype(str) + "-01-01", errors="coerce")
            disc_dt = base + pd.to_timedelta(doy.fillna(1).astype(int) - 1, unit="D")
            df = df.assign(_disc_dt=disc_dt)

            week_start = df["_disc_dt"].dt.to_period("W-MON").dt.start_time

            size_raw = df["FIRE_SIZE_CLASS"].astype(str).str.strip()
            size_grp = size_raw.replace({"E": "EFG", "F": "EFG", "G": "EFG"})
            size_grp = size_grp.where(size_grp.isin(SIZE_GROUPS), other=np.nan)

            df = df.assign(_week_start=week_start, _size=size_grp).dropna(subset=["_week_start", "_size"])

            wk = (
                df.groupby(["_week_start", "_size"])
                .size()
                .unstack("_size", fill_value=0)
                .reset_index()
                .rename(columns={"_week_start": "week_start"})
            )
            for c in SIZE_GROUPS:
                if c not in wk.columns:
                    wk[c] = 0
            wk = wk.sort_values("week_start").reset_index(drop=True)

        lookback = int(self.lookback_weeks)
        if len(wk) <= lookback:
            raise ValueError(f"Not enough weeks ({len(wk)}) for lookback={lookback}")

        counts = wk[SIZE_GROUPS].to_numpy(dtype=np.float32)  # (T, 5)

        if self.features == "counts":
            feats = counts
            feature_names = SIZE_GROUPS
        elif self.features == "counts+time":
            woy = wk["week_start"].dt.isocalendar().week.to_numpy(dtype=np.float32)
            sin = np.sin(2 * np.pi * woy / 52.0).reshape(-1, 1).astype(np.float32)
            cos = np.cos(2 * np.pi * woy / 52.0).reshape(-1, 1).astype(np.float32)
            feats = np.concatenate([counts, sin, cos], axis=1).astype(np.float32)
            feature_names = SIZE_GROUPS + ["woy_sin", "woy_cos"]
        else:
            raise ValueError(f"Unknown features mode: {self.features}")

        X_list: List[np.ndarray] = []
        Y_list: List[np.ndarray] = []
        week_starts: List[pd.Timestamp] = []

        for i in range(lookback, len(wk)):
            X_list.append(feats[i - lookback:i])
            Y_list.append(counts[i])
            week_starts.append(wk.loc[i, "week_start"])

        X = np.stack(X_list, axis=0).astype(np.float32)  # (N, lookback, input_dim)
        Y = np.stack(Y_list, axis=0).astype(np.float32)  # (N, 5)

        n = int(X.shape[0])
        train_idx, val_idx, test_idx = _chronological_split_indices(
            n=n, train_ratio=self.train_ratio, val_ratio=self.val_ratio, test_ratio=self.test_ratio
        )

        # ---- Canonical splits as DataSplit(torch tensors) ----
        X_train = torch.as_tensor(X[train_idx], dtype=torch.float32)
        y_train = torch.as_tensor(Y[train_idx], dtype=torch.float32)

        X_val = torch.as_tensor(X[val_idx], dtype=torch.float32)
        y_val = torch.as_tensor(Y[val_idx], dtype=torch.float32)

        X_test = torch.as_tensor(X[test_idx], dtype=torch.float32)
        y_test = torch.as_tensor(Y[test_idx], dtype=torch.float32)

        splits: Dict[str, DataSplit] = {
            "train": DataSplit(inputs=X_train, targets=y_train, metadata={"source": source, "region": self.region}),
            "val": DataSplit(inputs=X_val, targets=y_val, metadata={"source": source, "region": self.region}),
            "test": DataSplit(inputs=X_test, targets=y_test, metadata={"source": source, "region": self.region}),
        }

        feature_spec = FeatureSpec(
            input_dim=int(X_train.shape[2]),
            description="Weekly sequence features derived from FPA FOD incident aggregation.",
            extra={
                "feature_names": feature_names,
                "lookback_weeks": lookback,
                "dtype": "float32",
                "region": self.region,
            },
        )

        label_spec = LabelSpec(
            num_targets=5,
            task_type="regression",
            description="Next-week counts per size group (A,B,C,D,EFG).",
            extra={"targets": SIZE_GROUPS, "dtype": "float32"},
        )

        metadata: Dict[str, Any] = {
            "dataset": self.name,
            "source": source,
            "region": self.region,
            "micro": bool(self.micro),
            "seed": int(self.seed),
            "lookback_weeks": lookback,
            "features_mode": self.features,
            "week_start_for_each_sample": [str(ts) for ts in week_starts],
        }

        return DataBundle(
            splits=splits,
            feature_spec=feature_spec,
            label_spec=label_spec,
            metadata=metadata,
        )
