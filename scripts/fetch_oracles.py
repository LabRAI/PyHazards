"""Fetch the pinned reference implementations used by tests/oracle.

usage: python scripts/fetch_oracles.py [--dest DIR] [NAME ...]

DIR defaults to $PYHAZARDS_ORACLE_DIR, then ~/.cache/pyhazards-oracles. Git repositories are
checked out at the pinned commit; assets are downloaded and verified against their sha256 (saved
under the URL's file name, or ``filename`` when the URL has none, e.g. Google Drive downloads).
Assets marked ``large: true`` (multi-GB checkpoints) are fetched only when named explicitly or
with ``--large``; the tests that need them are skipped when they are absent.
Run the oracle tests afterwards with ``PYHAZARDS_ORACLE_DIR=DIR python -m pytest tests/oracle``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import yaml

MANIFEST = Path(__file__).resolve().parent.parent / "tests" / "oracle" / "repos.yaml"


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def fetch_repo(dest: Path, name: str, url: str, commit: str) -> None:
    target = dest / name
    if (target / ".git").exists():
        try:
            if _git("rev-parse", "HEAD", cwd=target) == commit:
                print(f"ok      {name} @ {commit[:7]}")
                return
        except subprocess.CalledProcessError:
            pass
    target.mkdir(parents=True, exist_ok=True)
    if not (target / ".git").exists():
        _git("init", "-q", cwd=target)
        _git("remote", "add", "origin", url, cwd=target)
    _git("fetch", "-q", "--depth", "1", "origin", commit, cwd=target)
    _git("checkout", "-q", "--detach", "FETCH_HEAD", cwd=target)
    head = _git("rev-parse", "HEAD", cwd=target)
    if head != commit:
        raise RuntimeError(f"{name}: expected {commit}, got {head}")
    print(f"fetched {name} @ {commit[:7]}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_asset(dest: Path, name: str, url: str, sha256: str, extract: bool = False, filename: str | None = None) -> None:
    target = dest / name
    target.mkdir(parents=True, exist_ok=True)
    archive = target / (filename or Path(url.split("?")[0]).name)
    if not archive.exists() or _sha256(archive) != sha256:
        print(f"download {name}")
        urllib.request.urlretrieve(url, archive)
    actual = _sha256(archive)
    if actual != sha256:
        raise RuntimeError(f"{name}: sha256 mismatch ({actual})")
    if extract and zipfile.is_zipfile(archive):
        marker = target / ".extracted"
        if not marker.exists():
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(target)
            marker.write_text(sha256)
    print(f"ok      {name}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="repos/assets to fetch (default: all but large assets)")
    parser.add_argument("--large", action="store_true", help="also fetch assets marked large: true")
    parser.add_argument(
        "--dest",
        default=os.environ.get("PYHAZARDS_ORACLE_DIR", str(Path.home() / ".cache" / "pyhazards-oracles")),
    )
    args = parser.parse_args(argv)
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    wanted = set(args.names)
    known = set(manifest.get("repos", {})) | set(manifest.get("assets", {}))
    unknown = wanted - known
    if unknown:
        print(f"unknown names: {sorted(unknown)}; known: {sorted(known)}", file=sys.stderr)
        return 2
    for name, spec in manifest.get("repos", {}).items():
        if not wanted or name in wanted:
            fetch_repo(dest, name, spec["url"], spec["commit"])
    for name, spec in manifest.get("assets", {}).items():
        if name in wanted or (not wanted and (args.large or not spec.get("large", False))):
            fetch_asset(dest, name, spec["url"], spec["sha256"], spec.get("extract", False), spec.get("filename"))
    print(f"oracles in {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
