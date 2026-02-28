# scripts/evaluate/extract_table10_from_two_runs.py
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from common_eval import ensure, write_json

BUCKETS = ["A","B","C","D","EFG"]


def load_metrics(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "metrics.json"
    ensure(p.exists(), f"Missing metrics.json in {run_dir}")
    return json.loads(p.read_text())


def extract_table10_like(m: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Your schema:
      m["test"]["rmse"] float (ALL)
      m["test"]["rmse_per_group"] list len=5 (A,B,C,D,EFG)
      m["test"]["r2"] is NOT present, so we cannot invent it.
    The paper wants RMSE and R². If R² doesn't exist, we hard-fail.
    """
    ensure("test" in m and isinstance(m["test"], dict), "metrics.json missing dict 'test'")
    t = m["test"]

    # Must have rmse and rmse_per_group
    ensure("rmse" in t, "test.rmse missing")
    ensure("rmse_per_group" in t, "test.rmse_per_group missing")
    ensure(isinstance(t["rmse_per_group"], list) and len(t["rmse_per_group"]) == 5, "rmse_per_group must be list len=5")

    # R2 requirement: must be present overall + per-group, OR we fail (paper parity)
    r2 = t.get("r2", None)
    r2pg = t.get("r2_per_group", None)
    ensure(r2 is not None, "test.r2 missing — cannot match paper Table10 without R². Re-run weekly evaluation to log R².")
    ensure(isinstance(r2pg, list) and len(r2pg) == 5, "test.r2_per_group missing/invalid — need list len=5")

    out: Dict[str, Dict[str, float]] = {}
    out["all"] = {"rmse": float(t["rmse"]), "r2": float(r2)}
    for i, b in enumerate(BUCKETS):
        out[b] = {"rmse": float(t["rmse_per_group"][i]), "r2": float(r2pg[i])}
    return out


def write_table_csv(us: Dict[str, Dict[str, float]], ca: Dict[str, Dict[str, float]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "bucket", "rmse", "r2"])
        for region, metrics in [("United States", us), ("CA", ca)]:
            for b in ["all"] + BUCKETS:
                w.writerow([region, b, metrics[b]["rmse"], metrics[b]["r2"]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--us-run", required=True)
    ap.add_argument("--ca-run", required=True)
    ap.add_argument("--outdir", default="outputs/phase5/forecasting")
    args = ap.parse_args()

    us_run = Path(args.us_run)
    ca_run = Path(args.ca_run)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    us_m = load_metrics(us_run)
    ca_m = load_metrics(ca_run)

    us = extract_table10_like(us_m)
    ca = extract_table10_like(ca_m)

    write_json(outdir / "forecast_metrics_us.json", us)
    write_json(outdir / "forecast_metrics_ca.json", ca)
    write_table_csv(us, ca, outdir / "table10_like.csv")

    print("[OK] wrote outputs/phase5/forecasting/table10_like.csv")
    print("[OK] wrote forecast_metrics_us.json / forecast_metrics_ca.json")


if __name__ == "__main__":
    main()
