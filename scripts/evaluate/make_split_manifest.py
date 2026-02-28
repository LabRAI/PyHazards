# scripts/evaluate/make_split_manifest.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from common_eval import ensure, sha256_json, write_json, utc_now_iso


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", required=True, help="e.g., fpa_fod_v20221014")
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--ids-npz", required=True, help="NPZ containing array 'ids' for this split")
    ap.add_argument("--out", default="outputs/phase5/split_manifest.json")
    args = ap.parse_args()

    ids_path = Path(args.ids_npz)
    ensure(ids_path.exists(), f"ids npz not found: {ids_path}")

    data = np.load(str(ids_path), allow_pickle=True)
    ensure("ids" in data.files, f"{ids_path} must contain key 'ids'")

    ids = data["ids"]
    ensure(ids.ndim == 1, "ids must be 1D")
    ensure(len(ids) > 0, "ids cannot be empty")

    # Stable JSONable representation
    # If ids are large, we store hash + count + sample (first/last 5)
    ids_list = [str(x) for x in ids.tolist()]
    ids_sha = sha256_json(ids_list)

    manifest: Dict[str, Any] = {
        "created_utc": utc_now_iso(),
        "dataset_id": args.dataset_id,
        "split": args.split,
        "n": int(len(ids_list)),
        "ids_sha256": ids_sha,
        "ids_preview_first5": ids_list[:5],
        "ids_preview_last5": ids_list[-5:],
        "source_ids_npz": str(ids_path),
    }

    out_path = Path(args.out)
    write_json(out_path, manifest)

    print(f"[OK] wrote {out_path}")
    print(f"[OK] ids_sha256 = {ids_sha}")


if __name__ == "__main__":
    main()
