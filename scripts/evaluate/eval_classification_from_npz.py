# scripts/evaluate/eval_classification_from_npz.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pyhazards.metrics.classification import evaluate_multiclass
from pyhazards.plotting.confusion import plot_confusion_matrix
from common_eval import (
    build_run_meta,
    ensure,
    write_json,
    write_phase5_readme,
)


def _load_npz(path: Path) -> dict:
    ensure(path.exists(), f"preds npz not found: {path}")
    d = np.load(str(path), allow_pickle=True)
    return {k: d[k] for k in d.files}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="NPZ with y_true,y_pred,label_names,split")
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split-manifest", default="outputs/phase5/split_manifest.json")
    ap.add_argument("--outdir", default="outputs/phase5/classification")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    preds_path = Path(args.preds)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    blob = _load_npz(preds_path)

    required = ["y_true", "y_pred", "label_names", "split"]
    for k in required:
        ensure(k in blob, f"{preds_path} missing key '{k}' (required: {required})")

    split = str(blob["split"])
    ensure(split == "test", f"split must be 'test' but got '{split}'")

    y_true = np.asarray(blob["y_true"])
    y_pred = np.asarray(blob["y_pred"])
    label_names = [str(x) for x in np.asarray(blob["label_names"]).tolist()]

    # HARD VALIDATION
    ensure(y_true.ndim == 1, "y_true must be 1D (N,)")
    ensure(y_pred.ndim == 1, "y_pred must be 1D (N,)")
    ensure(len(y_true) == len(y_pred), "y_true and y_pred length mismatch")
    ensure(len(y_true) > 0, "empty test set not allowed")
    ensure(len(label_names) >= 2, "need >=2 classes")
    ensure(np.isfinite(y_true.astype(float)).all(), "y_true has NaN/Inf")
    ensure(np.isfinite(y_pred.astype(float)).all(), "y_pred has NaN/Inf")

    K = len(label_names)
    ensure(np.issubdtype(y_true.dtype, np.integer), "y_true must be integer labels")
    ensure(np.issubdtype(y_pred.dtype, np.integer), "y_pred must be integer labels")
    ensure(int(y_true.min()) >= 0 and int(y_true.max()) < K, "y_true labels out of range")
    ensure(int(y_pred.min()) >= 0 and int(y_pred.max()) < K, "y_pred labels out of range")

    res = evaluate_multiclass(y_true, y_pred, label_names)

    meta = build_run_meta(
        repo_root=repo_root,
        dataset_id=args.dataset_id,
        split="test",
        split_manifest_path=Path(args.split_manifest),
        checkpoint=args.checkpoint,
        preds_path=preds_path,
    )

    # Save
    write_json(out_dir / "run_meta.json", meta.__dict__)
    write_json(
        out_dir / "metrics.json",
        {
            "accuracy": res.accuracy,
            "labels": res.labels,
            "classification_report": res.report_dict,
            "confusion_matrix": res.confusion.tolist(),
            "confusion_matrix_normalized": res.confusion_normalized.tolist(),
        },
    )
    (out_dir / "classification_report.txt").write_text(res.report_csv)

    plot_confusion_matrix(
        res.confusion, res.labels,
        title="Confusion Matrix",
        out_path=str(out_dir / "confusion_matrix.png"),
        normalize=False,
    )
    plot_confusion_matrix(
        res.confusion_normalized, res.labels,
        title="Confusion Matrix",
        out_path=str(out_dir / "confusion_matrix_normalized.png"),
        normalize=True,
    )

    extra = f"""
- N_test: {len(y_true)}
- accuracy: {res.accuracy:.6f}
- outputs:
  - metrics.json
  - classification_report.txt
  - confusion_matrix.png
  - confusion_matrix_normalized.png
"""
    write_phase5_readme(out_dir, "Classification", meta, extra)

    print(f"[OK] Saved classification evaluation to: {out_dir}")


if __name__ == "__main__":
    main()
