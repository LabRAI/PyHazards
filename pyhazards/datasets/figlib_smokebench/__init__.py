"""SmokeBench evaluation set: HPWREN FIgLib frames with the SmokeyNet smoke boxes.

SmokeBench (Qi, Li & Barnes, WACV 2026, arXiv:2512.11215) uses "a subset of 5,046 images with
ground-truth bounding box annotations provided in [Dewangan et al. 2022]" as smoke images and 1,000
smoke-free images. SmokeBench itself released no data, image list or split (its evaluation script,
github.com/SoraLink/MLLM, expects them in a git-ignored ``dataset/`` folder). The pieces come from:

* boxes: ``bbox_labels`` of ``data/metadata.pkl`` in the SmokeyNet repository (Dewangan et al.,
  Remote Sensing 14:1007, 2022; gitlab.nrp-nautilus.io/anshumand/pytorch-lightning-smoke-detection,
  Apache-2.0), pinned to commit d8d52b9 and checked by sha256. It holds exactly 5,046 boxes, one per
  image, on 145 FIgLib sequences, and their area quintiles are exactly SmokeBench's area bins
  (42, 4356, 11232, 28565, 83157, 2232020), so this is the SmokeBench smoke set;
* images: the FIgLib per-sequence archives on the HPWREN CDN (``HPWREN-FIgLib-Data/Tar``). HPWREN
  makes the data public and asks that derivative work credit https://www.hpwren.ucsd.edu/; no
  licence file is given, so PyHazards downloads from the official source and redistributes nothing;
* smoke-free images: frames before the visible plume (negative offset in the file name
  ``<timestamp>_<offset>.jpg``). The authors drew 1,000 of them with an unseeded ``random.sample``
  over all FIgLib sequences they had downloaded and did not release the list, so it cannot be
  rebuilt; PyHazards draws ``negatives`` frames with a seeded generator from the pre-ignition frames
  of the selected sequences.

Boxes are ``[x1, y1, x2, y2]`` integer pixel coordinates of the original frame (the rounded extremes
of the annotated polygon). Frames are 1600 x 1200, 2048 x 1536 or 3072 x 2048 RGB JPEGs.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import pickle
import random
import re
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

SMOKEYNET_REPO_URL = "https://gitlab.nrp-nautilus.io/anshumand/pytorch-lightning-smoke-detection"
SMOKEYNET_COMMIT = "d8d52b957891de10d4bef08943535fa4f9c3cd43"
SMOKEYNET_METADATA_URL = f"{SMOKEYNET_REPO_URL}/-/raw/{SMOKEYNET_COMMIT}/data/metadata.pkl"
SMOKEYNET_METADATA_SHA256 = "c3465c121669b0f92426451799140b0d30bc7caf7512aff9326fefa698ecb1a6"
METADATA_FILENAME = "smokeynet_metadata.pkl"
FIGLIB_ARCHIVE_URL = "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/Tar/{sequence}.tgz"
FIGLIB_HOME_URL = "https://www.hpwren.ucsd.edu/FIgLib/"

NUM_SMOKE_IMAGES = 5046
NUM_SEQUENCES = 145
NUM_NEGATIVES = 1000

_FRAME_NAME = re.compile(r"^(\d+)_([+-])(\d+)\.jpg$")
_COMPLETE_MARKER = ".complete"

PathLike = Union[str, Path]


@dataclass(frozen=True)
class SmokeSample:
    """One SmokeBench image: ``image_id`` is ``"<sequence>/<frame stem>"``."""

    image_id: str
    path: Path
    label: bool
    boxes: Tuple[Tuple[int, int, int, int], ...] = ()

    @property
    def sequence(self) -> str:
        return self.image_id.split("/")[0]

    @property
    def offset_seconds(self) -> int:
        """Seconds from the first visible plume (negative before ignition)."""
        match = _FRAME_NAME.match(self.image_id.split("/")[1] + ".jpg")
        if match is None:
            raise ValueError(f"unexpected FIgLib frame name {self.image_id!r}")
        sign = -1 if match.group(2) == "-" else 1
        return sign * int(match.group(3))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


_PICKLE_ALLOWED = {
    ("builtins", "dict"),
    ("builtins", "list"),
    ("builtins", "tuple"),
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
}


class _MetadataUnpickler(pickle.Unpickler):
    """Unpickler that only builds containers and numpy arrays/scalars (no arbitrary code)."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _PICKLE_ALLOWED:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name} from the SmokeyNet metadata")
        try:
            return getattr(importlib.import_module(module), name)
        except ImportError:
            swapped = module.replace("numpy.core", "numpy._core") if "numpy.core" in module else module.replace("numpy._core", "numpy.core")
            return getattr(importlib.import_module(swapped), name)


def load_smokeynet_boxes(path: PathLike, *, verify: bool = True) -> Dict[str, List[List[int]]]:
    """``{"<sequence>/<frame>": [[x1, y1, x2, y2], ...]}`` from SmokeyNet's ``metadata.pkl``."""
    path = Path(path)
    if verify:
        actual = _sha256(path)
        if actual != SMOKEYNET_METADATA_SHA256:
            raise ValueError(f"{path} has sha256 {actual}, expected the pinned {SMOKEYNET_METADATA_SHA256}")
    metadata = _MetadataUnpickler(io.BytesIO(path.read_bytes())).load()
    boxes = metadata["bbox_labels"]
    return {
        str(image_id): [[int(v) for v in np.asarray(box).ravel()] for box in image_boxes]
        for image_id, image_boxes in boxes.items()
    }


def _download_file(url: str, target: Path) -> None:
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
    tmp.replace(target)


def download_metadata(root: PathLike) -> Path:
    """Fetch the pinned SmokeyNet ``metadata.pkl`` into ``root`` (verified by sha256)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / METADATA_FILENAME
    if not target.exists() or _sha256(target) != SMOKEYNET_METADATA_SHA256:
        _download_file(SMOKEYNET_METADATA_URL, target)
    actual = _sha256(target)
    if actual != SMOKEYNET_METADATA_SHA256:
        raise ValueError(f"downloaded {target} has sha256 {actual}, expected {SMOKEYNET_METADATA_SHA256}")
    return target


def download_sequence(root: PathLike, sequence: str) -> int:
    """Stream one FIgLib archive from HPWREN and extract its JPEG frames to ``root/<sequence>/``.

    Only regular files named ``<sequence>/<timestamp>_<offset>.jpg`` are written (the time-lapse MP4
    is skipped). Returns the number of frames; a ``.complete`` marker makes later calls no-ops.
    """
    root = Path(root)
    if "/" in sequence or sequence in {"", ".", ".."}:
        raise ValueError(f"invalid FIgLib sequence name {sequence!r}")
    directory = root / sequence
    marker = directory / _COMPLETE_MARKER
    if marker.exists():
        return int(marker.read_text() or 0)
    count = 0
    url = FIGLIB_ARCHIVE_URL.format(sequence=sequence)
    with urllib.request.urlopen(url, timeout=120) as response, tarfile.open(fileobj=response, mode="r|gz") as archive:
        for member in archive:
            parts = member.name.split("/")
            if not member.isfile() or len(parts) != 2 or parts[0] != sequence or not _FRAME_NAME.match(parts[1]):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            directory.mkdir(parents=True, exist_ok=True)
            (directory / parts[1]).write_bytes(handle.read())
            count += 1
    marker.write_text(str(count))
    return count


def download_smokebench(root: PathLike, sequences: Optional[Sequence[str]] = None, *, progress: bool = True) -> Path:
    """Download the SmokeyNet boxes and the FIgLib sequences SmokeBench uses (about 11.4 GB for all 145)."""
    root = Path(root)
    metadata = download_metadata(root)
    wanted = sorted(sequences) if sequences is not None else sorted({key.split("/")[0] for key in load_smokeynet_boxes(metadata)})
    for index, sequence in enumerate(wanted, 1):
        frames = download_sequence(root, sequence)
        if progress:
            print(f"[{index}/{len(wanted)}] {sequence}: {frames} frames", flush=True)
    return root


class FIgLibSmokeBench(Sequence[SmokeSample]):
    """The SmokeBench images: smoke images (with boxes) first, then the sampled smoke-free images.

    Args:
        root: directory holding ``smokeynet_metadata.pkl`` and one folder of JPEG frames per FIgLib
            sequence (the layout of the HPWREN archives; :func:`download_smokebench` creates it).
        sequences: restrict to these FIgLib sequences (default: the 145 annotated ones).
        negatives: number of smoke-free frames to draw (SmokeBench: 1,000); 0 for none.
        negative_seed: seed of the draw.
        download: fetch missing files first (:func:`download_smokebench`).
    """

    def __init__(
        self,
        root: PathLike,
        *,
        sequences: Optional[Sequence[str]] = None,
        negatives: int = NUM_NEGATIVES,
        negative_seed: int = 0,
        download: bool = False,
    ) -> None:
        self.root = Path(root)
        if negatives < 0:
            raise ValueError(f"negatives must be >= 0, got {negatives}")
        if download:
            download_smokebench(self.root, sequences, progress=True)
        metadata = self.root / METADATA_FILENAME
        if not metadata.exists():
            raise FileNotFoundError(
                f"{metadata} not found; call pyhazards.datasets.figlib_smokebench.download_smokebench({str(self.root)!r}) "
                "or pass download=True"
            )
        boxes = load_smokeynet_boxes(metadata)
        annotated = sorted({key.split("/")[0] for key in boxes})
        if sequences is None:
            selected = annotated
        else:
            selected = sorted(set(sequences))
            unknown = [name for name in selected if name not in annotated]
            if unknown:
                raise ValueError(f"sequences without SmokeyNet boxes: {unknown[:5]}")
        chosen = set(selected)
        positives = [
            SmokeSample(key, self.root / f"{key}.jpg", True, tuple(tuple(box) for box in boxes[key]))
            for key in sorted(boxes)
            if key.split("/")[0] in chosen
        ]
        missing = [sample.image_id for sample in positives if not sample.path.exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} annotated frames are missing under {self.root} (e.g. {missing[:3]}); "
                "download the sequences with pyhazards.datasets.figlib_smokebench.download_smokebench"
            )
        pool = sorted(
            f"{sequence}/{path.stem}"
            for sequence in selected
            for path in (self.root / sequence).glob("*_-*.jpg")
            if _FRAME_NAME.match(path.name)
        )
        drawn = sorted(random.Random(negative_seed).sample(pool, min(negatives, len(pool)))) if negatives else []
        self.negative_pool_size = len(pool)
        self.negative_seed = negative_seed
        self.sequences = selected
        self.samples: List[SmokeSample] = positives + [SmokeSample(key, self.root / f"{key}.jpg", False) for key in drawn]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):  # type: ignore[override]
        return self.samples[index]

    def __iter__(self) -> Iterator[SmokeSample]:
        return iter(self.samples)

    @property
    def positives(self) -> List[SmokeSample]:
        return [sample for sample in self.samples if sample.label]

    @property
    def negatives(self) -> List[SmokeSample]:
        return [sample for sample in self.samples if not sample.label]

    @staticmethod
    def load_image(sample: SmokeSample) -> np.ndarray:
        """Decode a frame to an RGB uint8 ``(H, W, 3)`` array (needs Pillow)."""
        try:
            from PIL import Image  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("Reading FIgLib frames needs Pillow: `pip install pyhazards[prompted]`.") from exc
        with Image.open(sample.path) as image:
            return np.array(image.convert("RGB"))

    def describe(self) -> Dict[str, Any]:
        return {
            "dataset": "figlib_smokebench",
            "root": str(self.root),
            "sequences": len(self.sequences),
            "smoke_images": len(self.positives),
            "smoke_free_images": len(self.negatives),
            "smoke_free_pool": self.negative_pool_size,
            "negative_seed": self.negative_seed,
            "boxes": f"{SMOKEYNET_METADATA_URL} (sha256 {SMOKEYNET_METADATA_SHA256})",
            "images": FIGLIB_ARCHIVE_URL,
        }


__all__ = [
    "FIGLIB_ARCHIVE_URL",
    "FIgLibSmokeBench",
    "METADATA_FILENAME",
    "NUM_NEGATIVES",
    "NUM_SEQUENCES",
    "NUM_SMOKE_IMAGES",
    "SMOKEYNET_COMMIT",
    "SMOKEYNET_METADATA_SHA256",
    "SMOKEYNET_METADATA_URL",
    "SmokeSample",
    "download_metadata",
    "download_sequence",
    "download_smokebench",
    "load_smokeynet_boxes",
]
