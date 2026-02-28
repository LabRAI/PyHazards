# scripts/evaluate/common_eval.py
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit_hash(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "UNKNOWN"


def pip_freeze_head(n: int = 80) -> str:
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
        lines = out.decode().splitlines()
        return "\n".join(lines[:n])
    except Exception:
        return "pip-freeze-unavailable"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    b = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def ensure(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


@dataclass(frozen=True)
class RunMeta:
    created_utc: str
    git_commit: str
    python: str
    platform: str
    pip_freeze_head: str
    dataset_id: str
    split: str
    split_manifest_path: str
    split_manifest_sha256: str
    checkpoint: str
    preds_path: str


def build_run_meta(
    *,
    repo_root: Path,
    dataset_id: str,
    split: str,
    split_manifest_path: Path,
    checkpoint: str,
    preds_path: Path,
) -> RunMeta:
    ensure(split_manifest_path.exists(), f"split_manifest not found: {split_manifest_path}")
    return RunMeta(
        created_utc=utc_now_iso(),
        git_commit=git_commit_hash(repo_root),
        python=sys.version.replace("\n", " "),
        platform=f"{platform.platform()} | {platform.machine()}",
        pip_freeze_head=pip_freeze_head(),
        dataset_id=dataset_id,
        split=split,
        split_manifest_path=str(split_manifest_path),
        split_manifest_sha256=sha256_file(split_manifest_path),
        checkpoint=checkpoint,
        preds_path=str(preds_path),
    )


def write_phase5_readme(out_dir: Path, title: str, meta: RunMeta, extra_lines: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "README.md"
    txt = []
    txt.append(f"# Phase 5 — {title}")
    txt.append("")
    txt.append("## Repro metadata")
    txt.append(f"- created_utc: {meta.created_utc}")
    txt.append(f"- git_commit: {meta.git_commit}")
    txt.append(f"- python: {meta.python}")
    txt.append(f"- platform: {meta.platform}")
    txt.append(f"- dataset_id: {meta.dataset_id}")
    txt.append(f"- split: {meta.split}")
    txt.append(f"- split_manifest_path: {meta.split_manifest_path}")
    txt.append(f"- split_manifest_sha256: {meta.split_manifest_sha256}")
    txt.append(f"- checkpoint: {meta.checkpoint}")
    txt.append(f"- preds_path: {meta.preds_path}")
    txt.append("")
    txt.append("## pip freeze (head)")
    txt.append("```")
    txt.append(meta.pip_freeze_head)
    txt.append("```")
    txt.append("")
    txt.append("## Notes / results")
    txt.append(extra_lines.strip() + "\n")
    p.write_text("\n".join(txt))
