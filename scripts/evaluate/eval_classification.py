# scripts/evaluate/eval_classification.py
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from pyhazards.metrics.classification import evaluate_multiclass
from pyhazards.plotting.confusion import plot_confusion_matrix


def ensure(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def _find_run_dir_from_checkpoint(ckpt_path: Path) -> Path:
    """
    Expected checkpoint layout (your tree confirms this):
      outputs/train_wildfire_fpa_{task}/{RUN_ID}/checkpoints/best.ckpt
    So run_dir = ckpt_path.parents[1]
    """
    ckpt_path = ckpt_path.resolve()
    ensure(ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}")
    ensure(ckpt_path.suffix == ".ckpt", f"Expected a .ckpt file, got: {ckpt_path}")
    run_dir = ckpt_path.parent.parent  # .../RUN_ID
    ensure(run_dir.exists(), f"Run dir not found from checkpoint: {run_dir}")
    return run_dir


def _load_json_if_exists(p: Path) -> Optional[Dict[str, Any]]:
    if p.exists():
        return json.loads(p.read_text())
    return None


def _extract_from_run_dir(run_dir: Path) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """
    We try to reconstruct y_true/y_pred/labels from training-run artifacts.

    Supported sources (in order):
      1) confusion_matrix.json if present (preferred)
      2) metrics.json if it contains a confusion matrix block (best effort)
      3) FAIL with a clear error message
    """
    cmj = run_dir / "confusion_matrix.json"
    mj = run_dir / "metrics.json"

    # 1) confusion_matrix.json (your training runs have this)
    cm_obj = _load_json_if_exists(cmj)
    if cm_obj is not None:
        # We accept a few common schemas:
        # A) {"confusion": [[...]], "labels": ["A","B",...]}
        # B) {"matrix": [[...]], "labels": [...]}
        # C) {"confusion_matrix": [[...]], "label_names": [...]}
        mat = (
            cm_obj.get("confusion")
            or cm_obj.get("matrix")
            or cm_obj.get("confusion_matrix")
            or cm_obj.get("cm")
        )
        labels = cm_obj.get("labels") or cm_obj.get("label_names") or cm_obj.get("classes")
        ensure(mat is not None and labels is not None, f"Unrecognized schema in {cmj}")
        confusion = np.array(mat, dtype=int)
        label_names = [str(x) for x in labels]

        # From confusion alone we cannot recover the exact per-example y_true/y_pred,
        # but we *can* generate a synthetic sample expansion that reproduces the confusion exactly.
        # This makes evaluate_multiclass + plots consistent with the run’s confusion matrix.
        y_true_list = []
        y_pred_list = []
        for i in range(confusion.shape[0]):
            for j in range(confusion.shape[1]):
                c = int(confusion[i, j])
                if c <= 0:
                    continue
                y_true_list.extend([i] * c)
                y_pred_list.extend([j] * c)

        ensure(len(y_true_list) > 0, f"Confusion matrix in {cmj} is empty")
        return np.array(y_true_list, dtype=int), np.array(y_pred_list, dtype=int), label_names

    # 2) metrics.json fallback (best-effort)
    m_obj = _load_json_if_exists(mj)
    if m_obj is not None:
        # Some pipelines store confusion under test.metrics or test.confusion etc.
        test = m_obj.get("test")
        if isinstance(test, dict):
            cand_blocks = [test, test.get("metrics", {}) if isinstance(test.get("metrics"), dict) else {}]
            for blk in cand_blocks:
                if not isinstance(blk, dict):
                    continue
                mat = blk.get("confusion") or blk.get("confusion_matrix") or blk.get("cm")
                labels = blk.get("labels") or blk.get("label_names") or blk.get("classes")
                if mat is not None and labels is not None:
                    confusion = np.array(mat, dtype=int)
                    label_names = [str(x) for x in labels]
                    y_true_list = []
                    y_pred_list = []
                    for i in range(confusion.shape[0]):
                        for j in range(confusion.shape[1]):
                            c = int(confusion[i, j])
                            if c <= 0:
                                continue
                            y_true_list.extend([i] * c)
                            y_pred_list.extend([j] * c)
                    ensure(len(y_true_list) > 0, f"Confusion matrix in {mj} is empty")
                    return np.array(y_true_list, dtype=int), np.array(y_pred_list, dtype=int), label_names

    raise RuntimeError(
        "Could not find confusion_matrix.json (preferred) or a usable confusion block inside metrics.json.\n"
        f"Looked in:\n  - {cmj}\n  - {mj}\n\n"
        "Fix: ensure your training run writes confusion_matrix.json, or extend this script for your schema."
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Phase5 classification evaluation exporter.\n\n"
            "This script is intentionally strict + reproducible.\n"
            "It exports classification metrics + confusion plots to the requested outdir.\n"
            "It derives results from the training run directory inferred from the checkpoint path."
        )
    )
    ap.add_argument("--dataset-id", required=True, help="Dataset ID string (stored into output metadata).")
    ap.add_argument("--checkpoint", required=True, help="Path to best.ckpt (used to locate the training run dir).")
    ap.add_argument("--split-manifest", required=True, help="Path to outputs/phase5/split_manifest.json (stored into output metadata).")
    ap.add_argument("--outdir", required=True, help="Output directory for this evaluation (e.g., outputs/phase5/classification_cause).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite outdir if it exists.")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    split_manifest = Path(args.split_manifest)
    out_dir = Path(args.outdir)

    ensure(split_manifest.exists(), f"split manifest not found: {split_manifest}")
    run_dir = _find_run_dir_from_checkpoint(ckpt)

    if out_dir.exists():
        ensure(args.overwrite, f"outdir exists: {out_dir} (use --overwrite to replace)")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract y_true/y_pred/labels from training run artifacts
    y_true, y_pred, label_names = _extract_from_run_dir(run_dir)

    res = evaluate_multiclass(y_true, y_pred, label_names)

    # Extra metadata for traceability
    meta = {
        "dataset_id": args.dataset_id,
        "checkpoint": str(ckpt),
        "run_dir": str(run_dir),
        "split_manifest": str(split_manifest),
        "cwd": os.getcwd(),
    }

    metrics_payload = {
        "meta": meta,
        "accuracy": float(res.accuracy),
        "report": res.report_dict,  # per-class + macro/weighted
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
    (out_dir / "classification_report.txt").write_text(res.report_csv)

    plot_confusion_matrix(
        res.confusion,
        res.labels,
        title="Confusion Matrix",
        out_path=str(out_dir / "confusion_matrix.png"),
        normalize=False,
    )
    plot_confusion_matrix(
        res.confusion_normalized,
        res.labels,
        title="Confusion Matrix (Normalized)",
        out_path=str(out_dir / "confusion_matrix_normalized.png"),
        normalize=True,
    )

    print(f"[OK] Saved classification evaluation to: {out_dir}")


if __name__ == "__main__":
    main()
