# scripts/evaluate/eval_forecasting.py
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from pyhazards.metrics.forecasting import evaluate_forecast_buckets


BUCKETS = ["A", "B", "C", "D", "EFG"]


def _save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2))


def main():
    out_dir = Path("outputs/phase5/forecasting")
    out_dir.mkdir(parents=True, exist_ok=True)

    # TODO: produce TEST-ONLY arrays for each region:
    # y_true_us, y_pred_us: shape (T, 5) for A,B,C,D,EFG
    # y_true_ca, y_pred_ca: shape (T, 5)
    raise_not_implemented = False
    if raise_not_implemented:
        raise RuntimeError("TODO: wire dataset + inference to produce y_true/y_pred for US and CA test splits.")

    # Example placeholders (DELETE)
    y_true_us = np.random.poisson(10, size=(100, 5))
    y_pred_us = y_true_us + np.random.normal(0, 2, size=(100, 5))
    y_true_ca = np.random.poisson(5, size=(100, 5))
    y_pred_ca = y_true_ca + np.random.normal(0, 1, size=(100, 5))

    us = evaluate_forecast_buckets(y_true_us, y_pred_us, BUCKETS)
    ca = evaluate_forecast_buckets(y_true_ca, y_pred_ca, BUCKETS)

    # Save region JSON
    _save_json(out_dir / "forecast_metrics_us.json", {k: vars(v) for k, v in us.items()})
    _save_json(out_dir / "forecast_metrics_ca.json", {k: vars(v) for k, v in ca.items()})

    # Table 10-like CSV (paper parity)
    table_path = out_dir / "table10_like.csv"
    with table_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "bucket", "rmse", "r2"])
        for region, metrics in [("United States", us), ("California", ca)]:
            for bucket in ["all"] + BUCKETS:
                w.writerow([region, bucket, metrics[bucket].rmse, metrics[bucket].r2])

    print(f"[OK] Saved forecasting evaluation to: {out_dir}")
    print(f"[OK] Table10-like CSV: {table_path}")

def r2_score(y_true, y_pred):
    # y_true, y_pred: 1D numpy arrays
    import numpy as np
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


if __name__ == "__main__":
    main()
