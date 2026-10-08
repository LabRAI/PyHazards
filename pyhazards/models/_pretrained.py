"""Download helper for official pretrained weights (pinned URL + sha256).

Weights are never bundled with PyHazards. They are fetched on first use from the URLs pinned in each
model module, checked against their sha256 and cached under ``<torch hub dir>/checkpoints/pyhazards``.
"""

from __future__ import annotations

import hashlib
import urllib.error
from pathlib import Path
from typing import Sequence

import torch


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_download(family: str, filename: str, urls: Sequence[str], sha256: str) -> Path:
    """Local path of a pinned file, downloaded into the torch hub cache on first use."""
    path = Path(torch.hub.get_dir()) / "checkpoints" / "pyhazards" / family / filename
    if path.exists() and sha256_of(path) == sha256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in urls:
        try:
            torch.hub.download_url_to_file(url, str(path), hash_prefix=sha256, progress=False)
            if sha256_of(path) == sha256:
                return path
            errors.append(f"{url}: sha256 mismatch")
        except (urllib.error.URLError, RuntimeError, OSError) as error:  # HTTP errors or hash mismatch
            errors.append(f"{url}: {error}")
    raise RuntimeError(f"Could not download {family}/{filename}:\n" + "\n".join(errors))


__all__ = ["cached_download", "sha256_of"]
