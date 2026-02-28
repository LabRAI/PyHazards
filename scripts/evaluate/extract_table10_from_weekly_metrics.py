# scripts/evaluate/extract_table10_from_weekly_metrics.py
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

from common_eval import ensure, write_json

BUCKETS = ["A","B","C","D","EFG"]


def load_json(p: Path) -> Dict[str, Any]:
    ensure(p.exists(), f"Missing: {p}")
    return json.loads(p.read_text())


def find_region_block(test: Dict[str, Any], region: str) -> Optional[Dict[str, Any]]:
    # search multiple schemas
    paths = [
        (region,),
        ("per_group", region),
        ("groups", region),
        ("by_group", region),
        ("region", region),
    ]
    for path in paths:
        cur = test
        ok = True
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                ok = False
                break
            cur = cur[k]
        if ok and isinstance(cur, dict):
            return cur
    return None


def normalize_bucket_metrics(block: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Accept formats:
      block[bucket] = {rmse:..., r2:...}
      OR block["rmse"][bucket] and block["r2"][bucket]
      OR flat keys like "{bucket}_rmse"
    """
    out: Dict[str, Dict[str, float]] = {}

    # format 1: nested by bucket
    if all(isinstance(block.get(b), dict) for b in ["all"] + BUCKETS):
        for b in ["all"] + BUCKETS:
            rmse = block[b].get("rmse")
            r2   = block[b].get("r2") or block[b].get("r_squared") or block[b].get("r2_score")
            if rmse is None or r2 is None:
                continue
            out[b] = {"rmse": float(rmse), "r2": float(r2)}
        if len(out) >= 6:
            return out

    # format 2: block has rmse/r2 dicts
    if isinstance(block.get("rmse"), dict) and isinstance(block.get("r2"), dict):
        for b in ["all"] + BUCKETS:
            if b in block["rmse"] and b in block["r2"]:
                out[b] = {"rmse": float(block["rmse"][b]), "r2": float(block["r2"][b])}
        if len(out) >= 6:
            return out

    # format 3: flat
    for b in ["all"] + BUCKETS:
        rmse = None
        r2 = None
        for k in [f"{b}_rmse", f"rmse_{b}", f"{b}.rmse"]:
            if k in block: rmse = block[k]
        for k in [f"{b}_r2", f"r2_{b}", f"{b}.r2"]:
            if k in block: r2 = block[k]
        if rmse is not None and r2 is not None:
            out[b] = {"rmse": float(rmse), "r2": float(r2)}
    return out


def require_complete(region_metrics: Dict[str, Dict[str, float]], region: str) -> None:
    for b in ["all"] + BUCKETS:
        ensure(b in region_metrics, f"{region}: missing bucket '{b}'")
        ensure("rmse" in region_metrics[b] and "r2" in region_metrics[b], f"{region}/{b}: missing rmse/r2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="weekly run metrics.json path")
    ap.add_argument("--outdir", default="outputs/phase5/forecasting")
    ap.add_argument("--region-us", default="US")
    ap.add_argument("--region-ca", default="CA")
    args = ap.parse_args()

    m = load_json(Path(args.metrics))
    ensure("test" in m and isinstance(m["test"], dict), "metrics.json missing dict key 'test'")

    test = m["test"]

    us_block = find_region_block(test, args.region_us)
    ca_block = find_region_block(test, args.region_ca)

    ensure(us_block is not None, f"Could not find region block for {args.region_us} inside metrics['test']")
    ensure(ca_block is not None, f"Could not find region block for {args.region_ca} inside metrics['test']")

    us = normalize_bucket_metrics(us_block)
    ca = normalize_bucket_metrics(ca_block)

    ensure(len(us) >= 6, f"US block found but could not parse bucket metrics. keys={list(us_block.keys())}")
    ensure(len(ca) >= 6, f"CA block found but could not parse bucket metrics. keys={list(ca_block.keys())}")

    require_complete(us, "US")
    require_complete(ca, "CA")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    write_json(outdir / "forecast_metrics_us.json", us)
    write_json(outdir / "forecast_metrics_ca.json", ca)

    table = outdir / "table10_like.csv"
    with table.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "bucket", "rmse", "r2"])
        for region_name, reg in [("United States", us), ("CA", ca)]:
            for b in ["all"] + BUCKETS:
                w.writerow([region_name, b, reg[b]["rmse"], reg[b]["r2"]])

    print(f"[OK] wrote {table}")
    print(f"[OK] wrote forecast_metrics_us.json / forecast_metrics_ca.json")


if __name__ == "__main__":
    main()
