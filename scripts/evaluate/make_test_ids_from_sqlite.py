# scripts/evaluate/make_test_ids_from_sqlite.py
from __future__ import annotations

import argparse
import hashlib
import sqlite3
from pathlib import Path
from typing import List, Tuple

import numpy as np


def ensure(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def stable_hash(s: str) -> int:
    # deterministic across machines
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to FPA_FOD sqlite")
    ap.add_argument("--table", default="Fires")
    ap.add_argument("--id-col", default="FIRE_ID")
    ap.add_argument("--where", default="", help="Optional SQL WHERE clause (without 'WHERE')")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", default="phase5_seed_v1", help="String seed for deterministic hashing split")
    ap.add_argument("--out", default="test_ids.npz")
    args = ap.parse_args()

    db = Path(args.db)
    ensure(db.exists(), f"db not found: {db}")
    ensure(0.0 < args.test_frac < 1.0, "--test-frac must be in (0,1)")

    con = sqlite3.connect(str(db))
    try:
        where_sql = f" WHERE {args.where} " if args.where.strip() else ""
        q = f"SELECT {args.id_col} FROM {args.table} {where_sql}"
        rows = con.execute(q).fetchall()
    finally:
        con.close()

    ensure(len(rows) > 0, f"query returned 0 rows: {q}")

    ids = [str(r[0]) for r in rows]
    ensure(len(ids) == len(set(ids)), "duplicate IDs detected; id-col/table wrong?")

    # deterministic split by hash(seed + id)
    scored = []
    for fid in ids:
        hv = stable_hash(args.seed + "::" + fid)
        scored.append((hv, fid))
    scored.sort(key=lambda x: x[0])

    n = len(scored)
    n_test = int(round(n * args.test_frac))
    ensure(n_test > 0, "test set ended up empty; increase --test-frac")

    test_ids = np.array([fid for _, fid in scored[:n_test]], dtype=object)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, ids=test_ids)

    print(f"[OK] wrote {out}")
    print(f"[OK] n_total={n}  n_test={len(test_ids)}  test_frac={len(test_ids)/n:.4f}")
    print(f"[OK] seed='{args.seed}'")
    if args.where.strip():
        print(f"[OK] where='{args.where}'")


if __name__ == "__main__":
    main()

