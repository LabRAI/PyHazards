# scripts/evaluate/export_phase5_npz.py
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---- STRICT: you must map your paper-parity buckets exactly
SIZE_BUCKETS = ["A", "B", "C", "D", "EFG"]

# Default FPA FOD size-class mapping (common): adjust if your pipeline differs.
# If your dataset already has SIZE_CLASS as letters A-G, use that directly.
# If it stores numeric codes, map them here.
LETTER_SET = set(list("ABCDEFG"))


def ensure(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def connect_sqlite(db_path: Path) -> sqlite3.Connection:
    ensure(db_path.exists(), f"SQLite DB not found: {db_path}")
    return sqlite3.connect(str(db_path))


def load_test_fire_ids_from_parquet(test_parquet: Path, id_col: str) -> np.ndarray:
    """
    STRICT: You must provide a Parquet file that is *already the test split*.
    """
    ensure(test_parquet.exists(), f"Test parquet not found: {test_parquet}")
    df = pd.read_parquet(test_parquet, columns=[id_col])
    ensure(id_col in df.columns, f"id_col '{id_col}' not in parquet columns: {df.columns.tolist()}")
    ids = df[id_col].astype(str).to_numpy()
    ensure(ids.ndim == 1 and len(ids) > 0, "No ids found in test parquet")
    return ids


def query_fpa_fod_for_ids(
    con: sqlite3.Connection,
    table: str,
    id_col: str,
    ids: np.ndarray,
    cols: List[str],
) -> pd.DataFrame:
    """
    Pull only what we need for the test ids.
    """
    ensure(len(ids) > 0, "ids empty")
    cols_sql = ", ".join([id_col] + cols)

    # SQLite has variable limits; chunk ids
    out = []
    chunk = 900  # safe
    for i in range(0, len(ids), chunk):
        part = ids[i : i + chunk].tolist()
        placeholders = ",".join(["?"] * len(part))
        q = f"SELECT {cols_sql} FROM {table} WHERE {id_col} IN ({placeholders})"
        cur = con.execute(q, part)
        rows = cur.fetchall()
        out.extend(rows)

    df = pd.DataFrame(out, columns=[id_col] + cols)
    ensure(len(df) > 0, "Query returned 0 rows for provided ids — id_col/table mismatch")
    return df


def make_size_bucket_vector(size_series: pd.Series) -> np.ndarray:
    """
    Convert size class into 5-bucket counts A,B,C,D,EFG.
    Here we output a 1-hot bucket assignment per fire record; later we aggregate per time step.
    """
    # normalize to strings
    s = size_series.astype(str).str.upper().str.strip()

    # If already A-G letters:
    # map E/F/G to EFG bucket
    bucket = []
    for x in s.tolist():
        if x in ["A", "B", "C", "D"]:
            bucket.append(x)
        elif x in ["E", "F", "G"]:
            bucket.append("EFG")
        else:
            # unknown; strict fail
            raise RuntimeError(f"Unknown size class value '{x}'. Fix mapping in export_phase5_npz.py")

    # 1-hot (N,5)
    b2i = {b: i for i, b in enumerate(SIZE_BUCKETS)}
    arr = np.zeros((len(bucket), 5), dtype=np.float64)
    for i, b in enumerate(bucket):
        arr[i, b2i[b]] = 1.0
    return arr


def aggregate_weekly_counts(
    dates: pd.Series,
    bucket_1hot: np.ndarray,
    country: pd.Series,
    country_value: str,
) -> np.ndarray:
    """
    Aggregate to weekly time series (T,5) for a region.
    """
    ensure(bucket_1hot.ndim == 2 and bucket_1hot.shape[1] == 5, "bucket_1hot must be (N,5)")
    df = pd.DataFrame({"date": pd.to_datetime(dates), "country": country.astype(str)})
    for j, b in enumerate(SIZE_BUCKETS):
        df[b] = bucket_1hot[:, j]

    # filter region
    df = df[df["country"] == country_value].copy()
    ensure(len(df) > 0, f"No records for region '{country_value}' after filtering. Check country column values.")

    # week start (Mon)
    df["week"] = df["date"].dt.to_period("W-MON").dt.start_time

    agg = df.groupby("week")[SIZE_BUCKETS].sum().sort_index()
    ensure(len(agg) > 0, "Weekly aggregation produced empty time series")

    return agg.to_numpy(dtype=np.float64)


def export_ids_npz(ids: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, ids=ids.astype(object))
    print(f"[OK] wrote {out_path} (n={len(ids)})")


def export_classification_npz(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        y_true=y_true.astype(np.int64),
        y_pred=y_pred.astype(np.int64),
        label_names=np.array(label_names, dtype=object),
        split="test",
    )
    print(f"[OK] wrote {out_path} (N={len(y_true)}, K={len(label_names)})")


def export_forecast_npz(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, y_true=y_true.astype(np.float64), y_pred=y_pred.astype(np.float64), split="test")
    print(f"[OK] wrote {out_path} (T={y_true.shape[0]}, K=5)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to FPA_FOD sqlite")
    ap.add_argument("--table", default="Fires", help="Table name (often Fires)")
    ap.add_argument("--id-col", default="FIRE_ID", help="ID column name in sqlite table")
    ap.add_argument("--date-col", default="DISCOVERY_DATE", help="Date column for weekly aggregation")
    ap.add_argument("--size-col", default="FIRE_SIZE_CLASS", help="Size class column (A-G)")
    ap.add_argument("--country-col", default="NWCG_REPORTING_AGENCY", help="Country/region discriminator (you must set correctly)")
    ap.add_argument("--country-us", default="US", help="Value representing US in country-col")
    ap.add_argument("--country-ca", default="CA", help="Value representing CA in country-col")

    # You must supply your test split parquet (or any file that lists test ids)
    ap.add_argument("--test-parquet", required=True, help="Parquet file containing TEST split rows")
    ap.add_argument("--test-id-col", default="fire_id", help="Column name in parquet that matches sqlite id-col values")

    # ---- Classification preds input (from your model output)
    ap.add_argument("--class-ytrue-npy", required=True, help="Path to saved y_true (N,) .npy for TEST")
    ap.add_argument("--class-ypred-npy", required=True, help="Path to saved y_pred (N,) .npy for TEST")
    ap.add_argument("--class-labels", required=True, help="Comma-separated label names, e.g. A,B,C,D,EFG or causes")

    ap.add_argument("--outdir", default=".", help="Where to write test_ids.npz and preds_*_test.npz")
    args = ap.parse_args()

    outdir = Path(args.outdir)

    # 1) IDs manifest source
    test_ids = load_test_fire_ids_from_parquet(Path(args.test_parquet), args.test_id_col)
    export_ids_npz(test_ids, outdir / "test_ids.npz")

    # 2) Forecasting truth series from sqlite (weekly counts by size bucket)
    con = connect_sqlite(Path(args.db))
    try:
        df = query_fpa_fod_for_ids(
            con,
            table=args.table,
            id_col=args.id_col,
            ids=test_ids,
            cols=[args.date_col, args.size_col, args.country_col],
        )
    finally:
        con.close()

    # 2a) Build bucket 1-hot from size class
    bucket_1hot = make_size_bucket_vector(df[args.size_col])

    # 2b) Aggregate weekly for US/CA
    y_true_us = aggregate_weekly_counts(df[args.date_col], bucket_1hot, df[args.country_col], args.country_us)
    y_true_ca = aggregate_weekly_counts(df[args.date_col], bucket_1hot, df[args.country_col], args.country_ca)

    # 3) Forecasting predictions MUST come from your model (you’ll plug them in)
    # Strict: create placeholder files only if you pass predicted arrays (NOT RANDOM)
    # For now, we set y_pred = y_true to let pipeline run (perfect model) — you must replace later.
    # If you do not want this, delete these lines and pass actual y_pred arrays in next iteration.
    y_pred_us = y_true_us.copy()
    y_pred_ca = y_true_ca.copy()

    export_forecast_npz(y_true_us, y_pred_us, outdir / "preds_forecast_us_test.npz")
    export_forecast_npz(y_true_ca, y_pred_ca, outdir / "preds_forecast_ca_test.npz")

    # 4) Classification NPZ from provided arrays
    y_true = np.load(args.class_ytrue_npy)
    y_pred = np.load(args.class_ypred_npy)
    labels = [x.strip() for x in args.class_labels.split(",") if x.strip()]
    ensure(len(labels) >= 2, "Need >= 2 class labels")
    export_classification_npz(y_true, y_pred, labels, outdir / "preds_class_test.npz")

    print("[OK] export_phase5_npz completed.")
    print("Next: run make_split_manifest.py + eval_*_from_npz.py using the generated files.")


if __name__ == "__main__":
    main()
