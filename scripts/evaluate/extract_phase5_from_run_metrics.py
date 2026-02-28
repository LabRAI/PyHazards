# scripts/evaluate/extract_phase5_from_run_metrics.py
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from common_eval import ensure, write_json, build_run_meta, write_phase5_readme


BUCKETS = ["A", "B", "C", "D", "EFG"]


def load_json(p: Path) -> Dict[str, Any]:
    ensure(p.exists(), f"Missing file: {p}")
    return json.loads(p.read_text())


def find_any(d: Dict[str, Any], keys) -> Optional[Any]:
    for k in keys:
        if k in d:
            return d[k]
    return None


def normalize_region_key(k: str) -> str:
    s = k.lower().replace(" ", "").replace("_", "")
    if s in ["us", "unitedstates", "usa"]:
        return "United States"
    if s in ["ca", "canada"]:
        return "California"  # NOTE: paper says CA (Canada). If your run uses CA=Canada, keep "Canada".
    return k


def extract_forecasting_table10(metrics: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    We accept multiple possible shapes.
    REQUIRED output:
      us_metrics: dict bucket-> {rmse,r2}
      ca_metrics: dict bucket-> {rmse,r2}
    """
    # Common patterns we try:
    # 1) metrics["forecast"] = {"US": {"all": {...}, "A": {...}, ...}, "CA": {...}}
    # 2) metrics["table10"] = {"US": {...}, "CA": {...}}
    # 3) flat keys like "us/all/rmse" etc.

    container = find_any(metrics, ["forecast", "forecasting", "table10", "table_10", "paper_table10"])
    if isinstance(container, dict):
        # try region dict
        us = None
        ca = None
        for rk, rv in container.items():
            if not isinstance(rv, dict):
                continue
            name = rk.lower()
            if name in ["us", "usa", "united_states", "unitedstates"]:
                us = rv
            if name in ["ca", "canada", "california"]:
                ca = rv
        if us is not None and ca is not None:
            return us, ca

    # Try flat schema: keys contain 'us'/'ca' and 'rmse'/'r2'
    def build_region(prefix: str) -> Dict[str, Any]:
        out = {}
        for b in ["all"] + BUCKETS:
            rmse = find_any(metrics, [f"{prefix}/{b}/rmse", f"{prefix}_{b}_rmse", f"{prefix}.{b}.rmse"])
            r2   = find_any(metrics, [f"{prefix}/{b}/r2",   f"{prefix}_{b}_r2",   f"{prefix}.{b}.r2"])
            if rmse is None or r2 is None:
                continue
            out[b] = {"rmse": float(rmse), "r2": float(r2)}
        return out

    us = build_region("us")
    ca = build_region("ca")
    ensure(len(us) >= 2 and len(ca) >= 2,
           "Could not extract forecasting metrics for US/CA from metrics.json. "
           "Expected something like metrics['forecast']['US']['all']['rmse'] etc.")
    return us, ca


def ensure_has_buckets(region_metrics: Dict[str, Any], region_name: str) -> None:
    for b in ["all"] + BUCKETS:
        ensure(b in region_metrics, f"{region_name} missing bucket '{b}' in extracted metrics.")
        ensure("rmse" in region_metrics[b] and "r2" in region_metrics[b],
               f"{region_name}/{b} missing rmse/r2.")


def extract_classification(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    We want accuracy + per-class precision/recall/f1 + confusion matrix if present.
    We accept:
      - metrics["test"] dict
      - metrics["eval"] dict
      - sklearn-style report dict
    """
    container = find_any(metrics, ["test", "eval", "evaluation", "metrics"])
    if isinstance(container, dict):
        metrics = container

    # try common keys
    acc = find_any(metrics, ["test/acc", "acc", "accuracy", "test_accuracy", "test.acc"])
    report = find_any(metrics, ["classification_report", "report", "per_class", "class_report"])
    cm = find_any(metrics, ["confusion", "confusion_matrix", "cm"])
    labels = find_any(metrics, ["labels", "label_names", "classes", "class_names"])

    out = {}
    if acc is not None:
        out["accuracy"] = float(acc)
    if report is not None:
        out["classification_report"] = report
    if cm is not None:
        out["confusion_matrix"] = cm
    if labels is not None:
        out["labels"] = labels

    ensure(out, "No classification metrics found in metrics.json (acc/report/cm/labels missing).")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weekly-run", required=True)
    ap.add_argument("--cause-run", required=True)
    ap.add_argument("--size-run", required=True)
    ap.add_argument("--dataset-id", default="fpa_fod_v20221014")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    weekly = Path(args.weekly_run)
    cause  = Path(args.cause_run)
    size   = Path(args.size_run)

    # ---------- Forecasting extraction ----------
    weekly_metrics = load_json(weekly / "metrics.json")
    us_m, ca_m = extract_forecasting_table10(weekly_metrics)

    # In your repo, "CA" might mean California in US-only weekly series OR Canada.
    # We keep label as "CA" but write region string "CA" in CSV if you want.
    # For now: United States and CA.
    ensure_has_buckets(us_m, "US")
    ensure_has_buckets(ca_m, "CA")

    out_fore = Path("outputs/phase5/forecasting")
    out_fore.mkdir(parents=True, exist_ok=True)

    write_json(out_fore / "forecast_metrics_us.json", us_m)
    write_json(out_fore / "forecast_metrics_ca.json", ca_m)

    table_path = out_fore / "table10_like.csv"
    with table_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "bucket", "rmse", "r2"])
        for region_name, metrics in [("United States", us_m), ("CA", ca_m)]:
            for bucket in ["all"] + BUCKETS:
                w.writerow([region_name, bucket, metrics[bucket]["rmse"], metrics[bucket]["r2"]])

    # README + meta
    meta = build_run_meta(
        repo_root=repo_root,
        dataset_id=args.dataset_id,
        split="test",  # as reported by run; we rely on run’s own split
        split_manifest_path=Path("outputs/phase5/split_manifest.json") if Path("outputs/phase5/split_manifest.json").exists() else Path("outputs/phase5/split_manifest.json"),
        checkpoint=str((weekly / "checkpoints" / "best.ckpt") if (weekly / "checkpoints" / "best.ckpt").exists() else weekly),
        preds_path=weekly / "metrics.json",
    )
    # NOTE: if split_manifest doesn't exist, we still write README; manifest creation handled later
    write_json(out_fore / "run_meta.json", meta.__dict__)
    extra = f"""
- source_run: {weekly}
- extracted_from: metrics.json
- outputs:
  - forecast_metrics_us.json
  - forecast_metrics_ca.json
  - table10_like.csv
"""
    write_phase5_readme(out_fore, "Forecasting (Table10 parity) — extracted", meta, extra)

    print(f"[OK] Forecasting Phase5 artifacts written to {out_fore}")
    print(f"[OK] {table_path}")

    # ---------- Classification extraction (cause/size) ----------
    def do_cls(run_dir: Path, out_dir: Path, ckpt_hint: str):
        m = load_json(run_dir / "metrics.json")
        cls = extract_classification(m)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "metrics_extracted.json", cls)

        meta2 = build_run_meta(
            repo_root=repo_root,
            dataset_id=args.dataset_id,
            split="test",
            split_manifest_path=Path("outputs/phase5/split_manifest.json") if Path("outputs/phase5/split_manifest.json").exists() else Path("outputs/phase5/split_manifest.json"),
            checkpoint=ckpt_hint,
            preds_path=run_dir / "metrics.json",
        )
        write_json(out_dir / "run_meta.json", meta2.__dict__)
        extra2 = f"""
- source_run: {run_dir}
- extracted_from: metrics.json
- keys_present: {list(cls.keys())}
"""
        write_phase5_readme(out_dir, f"Classification — extracted", meta2, extra2)
        print(f"[OK] Classification extracted to {out_dir}")

    do_cls(
        cause,
        Path("outputs/phase5/classification_cause"),
        str(cause / "checkpoints" / "best.ckpt") if (cause / "checkpoints" / "best.ckpt").exists() else str(cause),
    )
    do_cls(
        size,
        Path("outputs/phase5/classification_size"),
        str(size / "checkpoints" / "best.ckpt") if (size / "checkpoints" / "best.ckpt").exists() else str(size),
    )

    print("[DONE] Extraction complete. Next: if classification metrics lack confusion/report, we must run inference from ckpt.")


if __name__ == "__main__":
    main()
