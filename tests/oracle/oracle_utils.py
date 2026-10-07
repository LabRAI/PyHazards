"""Shared helpers for oracle tests: locate pinned reference code and import it in isolation.

The oracle tests are skipped unless ``PYHAZARDS_ORACLE_DIR`` points at a directory filled by
``scripts/fetch_oracles.py``. With ``PYHAZARDS_ORACLE_REQUIRED=1`` (the Oracle CI workflow) a
missing reference fails the test instead. Multi-GB assets (``large: true`` in repos.yaml) are not
fetched in CI: tests that need them skip when the asset is absent, unless
``PYHAZARDS_ORACLE_LARGE=1`` makes them mandatory too.
"""

from __future__ import annotations

import importlib
import importlib.metadata
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


def oracle_large_asset(name: str) -> Path:
    """Path of a ``large: true`` asset; skipped when absent unless ``PYHAZARDS_ORACLE_LARGE=1``."""
    path = oracle_dir() / name
    if not path.exists():
        reason = f"large oracle asset {name!r} is missing; run scripts/fetch_oracles.py {name}"
        if os.environ.get("PYHAZARDS_ORACLE_LARGE") == "1":
            pytest.fail(reason)
        pytest.skip(reason)
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


def oracle_package(
    name: str, version: str, requirements: str = "requirements.txt", distribution: str | None = None
) -> ModuleType:
    """Import a pip-installed reference package at an exact version (see ``requirements``).

    ``distribution`` is the PyPI name when it differs from the module name (``forefire`` installs
    ``pyforefire``).
    """
    try:
        module = importlib.import_module(name)
    except ImportError:
        missing(f"{name} is not installed; pip install -r tests/oracle/{requirements}")
    found = getattr(module, "__version__", None)
    if found is None:
        try:
            found = importlib.metadata.version(distribution or name)
        except importlib.metadata.PackageNotFoundError:
            pass
    if found != version:
        missing(f"needs {name}=={version}, found {found}")
    return module
