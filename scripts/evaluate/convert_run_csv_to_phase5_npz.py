# scripts/evaluate/convert_run_csv_to_phase5_npz.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def ensure(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def _read_any_csv(csv_path: Path) -> pd.DataFrame:
    ensure(csv_path.exists(), f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    ensure(len(df) > 0, f"CSV empty: {csv_path}")
    return df


def _try_find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def export_ids_npz(ids: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, ids=ids.astype(object))
    print(f"[OK] wrote {out_path} (n={len(ids)})")


def export_classification_npz(y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        y_true=y_true.astype(np.int64),
        y_pred=y_pred.astype(np.int64),
        label_names=np.array(label_names, dtype=object),
        split="test",
    )
    print(f"[OK] wrote {out_path} (N={len(y_true)}, K={len(label_names)})")


def export_forecast_npz(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, y_true=y_true.astype(np.float64), y_pred=y_pred.astype(np.float64), split="test")
    print(f"[OK] wrote {out_path} (T={y_true.shape[0]}, K=5)")


def load_label_names_from_config(run_dir: Path, default: List[str]) -> List[str]:
    cfg = run_dir / "config.json"
    if not cfg.exists():
        return default
    obj = json.loads(cfg.read_text())
    # common keys to try
    for key in ["label_names", "labels", "classes", "class_names"]:
        if key in obj and isinstance(obj[key], list) and len(obj[key]) >= 2:
            return [str(x) for x in obj[key]]
    return default


def convert_classification_csv(
    run_dir: Path,
    csv_path: Path,
    out_npz: Path,
) -> None:
    df = _read_any_csv(csv_path)

    # Accept many possible column names
    ytrue_col = _try_find_col(df, ["y_true", "true", "target", "label", "gt", "gold"])
    ypred_col = _try_find_col(df, ["y_pred", "pred", "prediction", "pred_label"])

    ensure(ytrue_col is not None, f"Cannot find y_true column in {csv_path}. columns={list(df.columns)}")
    ensure(ypred_col is not None, f"Cannot find y_pred column in {csv_path}. columns={list(df.columns)}")

    y_true = df[ytrue_col].to_numpy()
    y_pred = df[ypred_col].to_numpy()

    # If they are strings, map to ints
    if y_true.dtype == object or isinstance(y_true[0], str):
        uniq = sorted(set([str(x) for x in y_true] + [str(x) for x in y_pred]))
        m = {c: i for i, c in enumerate(uniq)}
        y_true = np.array([m[str(x)] for x in y_true], dtype=np.int64)
        y_pred = np.array([m[str(x)] for x in y_pred], dtype=np.int64)
        label_names = uniq
    else:
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)
        # try config.json for names, fallback to numeric labels
        label_names = load_label_names_from_config(run_dir, default=[str(i) for i in range(int(max(y_true.max(), y_pred.max())) + 1)])

    export_classification_npz(y_true, y_pred, label_names, out_npz)

    # If there is an id column, write test_ids.npz too (helps manifest)
    id_col = _try_find_col(df, ["id", "fire_id", "sample_id", "idx", "index"])
    if id_col is not None:
        ids = df[id_col].astype(str).to_numpy()
        export_ids_npz(ids, out_npz.parent / "test_ids.npz")


def convert_forecasting_csv_grouped(
    csv_us: Path,
    csv_ca: Path,
    out_us: Path,
    out_ca: Path,
) -> None:
    """
    Expect CSV with columns like:
      true_A,true_B,true_C,true_D,true_EFG and pred_A,... OR similar.
    We'll infer patterns robustly.
    """
    def parse(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        cols = [c.lower() for c in df.columns]
        # candidate patterns
        buckets = ["a","b","c","d","efg"]
        true_cols = []
        pred_cols = []

        # Try 'true_A' and 'pred_A'
        for b in buckets:
            tc = _try_find_col(df, [f"true_{b}", f"y_true_{b}", f"gt_{b}", f"actual_{b}"])
            pc = _try_find_col(df, [f"pred_{b}", f"y_pred_{b}", f"prediction_{b}", f"yhat_{b}"])
            ensure(tc is not None, f"Missing true column for bucket {b}. cols={list(df.columns)}")
            ensure(pc is not None, f"Missing pred column for bucket {b}. cols={list(df.columns)}")
            true_cols.append(tc)
            pred_cols.append(pc)

        y_true = df[true_cols].to_numpy(dtype=np.float64)
        y_pred = df[pred_cols].to_numpy(dtype=np.float64)
        ensure(y_true.shape[1] == 5 and y_pred.shape[1] == 5, "Must be (T,5)")
        return y_true, y_pred

    df_us = _read_any_csv(csv_us)
    df_ca = _read_any_csv(csv_ca)

    y_true_us, y_pred_us = parse(df_us)
    y_true_ca, y_pred_ca = parse(df_ca)

    export_forecast_npz(y_true_us, y_pred_us, out_us)
    export_forecast_npz(y_true_ca, y_pred_ca, out_ca)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weekly-run", required=True, help="Run dir for weekly forecast (has csv/)")
    ap.add_argument("--cause-run", required=True, help="Run dir for cause classification (has csv/)")
    ap.add_argument("--size-run", required=True, help="Run dir for size classification (has csv/)")
    ap.add_argument("--outdir", default=".", help="Where to write preds_*.npz")
    args = ap.parse_args()

    outdir = Path(args.outdir)

    # ----- Classification: pick the most likely CSVs
    def pick_first_csv(run: Path) -> Path:
        csvdir = run / "csv"
        ensure(csvdir.exists(), f"Missing csv dir: {csvdir}")
        csvs = sorted(csvdir.rglob("*.csv"))
        ensure(len(csvs) > 0, f"No csv files under: {csvdir}")
        return csvs[0]

    cause_run = Path(args.cause_run)
    size_run  = Path(args.size_run)
    weekly_run= Path(args.weekly_run)

    cause_csv = pick_first_csv(cause_run)
    size_csv  = pick_first_csv(size_run)

    print("[INFO] Using cause csv:", cause_csv)
    print("[INFO] Using size  csv:", size_csv)

    convert_classification_csv(cause_run, cause_csv, outdir / "preds_class_test.npz")  # cause
    convert_classification_csv(size_run,  size_csv,  outdir / "preds_size_test.npz")   # size (separate)

    # ----- Forecasting: need US + CA CSV (must exist somewhere in weekly run csv/)
    # We'll auto-detect by filename containing 'us' and 'ca'.
    csvdir = weekly_run / "csv"
    ensure(csvdir.exists(), f"Missing csv dir: {csvdir}")
    csvs = sorted(csvdir.rglob("*.csv"))
    ensure(len(csvs) > 0, f"No csv files under: {csvdir}")

    us = [p for p in csvs if "us" in p.name.lower()]
    ca = [p for p in csvs if "ca" in p.name.lower() or "can" in p.name.lower()]

    ensure(len(us) > 0, f"Could not find US csv in {csvdir}. Files={ [p.name for p in csvs[:30]] }")
    ensure(len(ca) > 0, f"Could not find CA csv in {csvdir}. Files={ [p.name for p in csvs[:30]] }")

    us_csv = us[0]
    ca_csv = ca[0]
    print("[INFO] Using forecast US csv:", us_csv)
    print("[INFO] Using forecast CA csv:", ca_csv)

    convert_forecasting_csv_grouped(
        us_csv, ca_csv,
        outdir / "preds_forecast_us_test.npz",
        outdir / "preds_forecast_ca_test.npz",
    )

    print("[OK] Wrote NPZ caches into:", outdir)
    print("Next: run make_split_manifest.py + eval_*_from_npz.py")


if __name__ == "__main__":
    main()
