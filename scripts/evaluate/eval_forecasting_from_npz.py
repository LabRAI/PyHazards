# scripts/evaluate/eval_forecasting_from_npz.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pyhazards.metrics.forecasting import evaluate_forecast_buckets
from common_eval import (
    build_run_meta,
    ensure,
    write_json,
    write_phase5_readme,
)

BUCKETS = ["A", "B", "C", "D", "EFG"]


def _load_region_npz(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    ensure(path.exists(), f"preds npz not found: {path}")
    d = np.load(str(path), allow_pickle=True)
    ensure("split" in d.files, f"{path} missing key 'split'")
    ensure("y_true" in d.files and "y_pred" in d.files, f"{path} must contain y_true and y_pred")

    split = str(d["split"])
    ensure(split == "test", f"{path}: split must be 'test' but got '{split}'")

    y_true = np.asarray(d["y_true"], dtype=np.float64)
    y_pred = np.asarray(d["y_pred"], dtype=np.float64)

    # HARD VALIDATION
    ensure(y_true.ndim == 2 and y_pred.ndim == 2, f"{path}: y_true/y_pred must be 2D (T,5)")
    ensure(y_true.shape == y_pred.shape, f"{path}: y_true/y_pred shape mismatch")
    ensure(y_true.shape[1] == 5, f"{path}: must have 5 columns for {BUCKETS}")
    ensure(y_true.shape[0] > 0, f"{path}: empty test timeline not allowed")
    ensure(np.isfinite(y_true).all() and np.isfinite(y_pred).all(), f"{path}: contains NaN/Inf")

    # If these are counts, enforce nonneg (strict)
    ensure((y_true >= 0).all(), f"{path}: y_true has negative values (counts must be non-negative)")

    return y_true, y_pred, split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--us", required=True, help="NPZ for United States test: y_true(T,5), y_pred(T,5), split='test'")
    ap.add_argument("--ca", required=True, help="NPZ for California test: y_true(T,5), y_pred(T,5), split='test'")
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split-manifest", default="outputs/phase5/split_manifest.json")
    ap.add_argument("--outdir", default="outputs/phase5/forecasting")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    us_path = Path(args.us)
    ca_path = Path(args.ca)

    y_true_us, y_pred_us, _ = _load_region_npz(us_path)
    y_true_ca, y_pred_ca, _ = _load_region_npz(ca_path)

    us = evaluate_forecast_buckets(y_true_us, y_pred_us, BUCKETS)
    ca = evaluate_forecast_buckets(y_true_ca, y_pred_ca, BUCKETS)

    # Save per-region json
    write_json(out_dir / "forecast_metrics_us.json", {k: {"rmse": v.rmse, "r2": v.r2} for k, v in us.items()})
    write_json(out_dir / "forecast_metrics_ca.json", {k: {"rmse": v.rmse, "r2": v.r2} for k, v in ca.items()})

    # Table10-like CSV
    table_path = out_dir / "table10_like.csv"
    with table_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "bucket", "rmse", "r2"])
        for region, metrics in [("United States", us), ("California", ca)]:
            for bucket in ["all"] + BUCKETS:
                w.writerow([region, bucket, metrics[bucket].rmse, metrics[bucket].r2])

    # Meta/README (tie to US preds path as primary)
    meta = build_run_meta(
        repo_root=repo_root,
        dataset_id=args.dataset_id,
        split="test",
        split_manifest_path=Path(args.split_manifest),
        checkpoint=args.checkpoint,
        preds_path=us_path,
    )
    write_json(out_dir / "run_meta.json", meta.__dict__)

    extra = f"""
- inputs:
  - US preds: {us_path}
  - CA preds: {ca_path}
- buckets: {BUCKETS}
- outputs:
  - forecast_metrics_us.json
  - forecast_metrics_ca.json
  - table10_like.csv
"""
    write_phase5_readme(out_dir, "Forecasting (Table10 parity)", meta, extra)

    print(f"[OK] Saved forecasting evaluation to: {out_dir}")
    print(f"[OK] Table10-like CSV: {table_path}")


if __name__ == "__main__":
    main()
