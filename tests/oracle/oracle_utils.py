"""Shared helpers for oracle tests: locate pinned reference code and import it in isolation.

The oracle tests are skipped unless ``PYHAZARDS_ORACLE_DIR`` points at a directory filled by
``scripts/fetch_oracles.py``. With ``PYHAZARDS_ORACLE_REQUIRED=1`` (the Oracle CI workflow) a
missing reference fails the test instead.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

MANIFEST = yaml.safe_load((Path(__file__).parent / "repos.yaml").read_text(encoding="utf-8"))


def missing(reason: str) -> None:
    """Skip, or fail when the oracle run is mandatory."""
    if os.environ.get("PYHAZARDS_ORACLE_REQUIRED") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


def oracle_dir() -> Path:
    root = os.environ.get("PYHAZARDS_ORACLE_DIR")
    if not root:
        missing("PYHAZARDS_ORACLE_DIR is not set; run scripts/fetch_oracles.py first")
    return Path(root).expanduser()


def oracle_repo(name: str) -> Path:
    """Path of a pinned reference repository, checked to be at the pinned commit."""
    path = oracle_dir() / name
    if not (path / ".git").exists():
        missing(f"oracle repo {name!r} is missing; run scripts/fetch_oracles.py {name}")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()
    expected = MANIFEST["repos"][name]["commit"]
    assert head == expected, f"oracle repo {name} is at {head}, expected pinned {expected}"
    return path


def oracle_asset(name: str) -> Path:
    path = oracle_dir() / name
    if not path.exists():
        missing(f"oracle asset {name!r} is missing; run scripts/fetch_oracles.py {name}")
    return path


def import_from(root: Path, module: str) -> ModuleType:
    """Import ``module`` with ``root`` on sys.path, then drop it from sys.modules.

    Reference repositories reuse generic package names such as ``src``; removing them after
    import keeps one repository's modules from shadowing another's.
    """
    before = set(sys.modules)
    sys.path.insert(0, str(root))
    try:
        return importlib.import_module(module)
    finally:
        sys.path.remove(str(root))
        for name in set(sys.modules) - before:
            if name.split(".")[0] == module.split(".")[0]:
                del sys.modules[name]


def oracle_package(name: str, version: str) -> ModuleType:
    """Import a pip-installed reference package at an exact version (see requirements.txt)."""
    try:
        module = importlib.import_module(name)
    except ImportError:
        missing(f"{name} is not installed; pip install -r tests/oracle/requirements.txt")
    if getattr(module, "__version__", None) != version:
        missing(f"needs {name}=={version}, found {getattr(module, '__version__', None)}")
    return module
