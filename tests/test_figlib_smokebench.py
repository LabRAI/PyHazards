"""Tests of the FIgLib SmokeBench dataset adapter on small synthetic fixtures (no network)."""

from __future__ import annotations

import hashlib
import io
import pickle
import tarfile
from pathlib import Path

import numpy as np
import pytest

import pyhazards.datasets.figlib_smokebench as figlib
from pyhazards.datasets.figlib_smokebench import FIgLibSmokeBench, SmokeSample, load_smokeynet_boxes

SEQ_A = "20160722_FIRE_mw-e-mobo-c"
SEQ_B = "20170520_FIRE_lp-s-iqeye"


def _write_metadata(root: Path, monkeypatch) -> Path:
    """A metadata.pkl shaped like SmokeyNet's (numpy int32 boxes) and its checksum pinned."""
    metadata = {
        "bbox_labels": {
            f"{SEQ_A}/1469223181_+00060": [[np.int32(1821), np.int32(1027), np.int32(1922), np.int32(1074)]],
            f"{SEQ_A}/1469223241_+00120": [np.array([1807, 1031, 1926, 1074], dtype=np.int32)],
            f"{SEQ_B}/1495300000_+00000": [[10, 20, 30, 40]],
        },
        "night_fires": np.array(["x"]),
    }
    path = root / figlib.METADATA_FILENAME
    path.write_bytes(pickle.dumps(metadata))
    monkeypatch.setattr(figlib, "SMOKEYNET_METADATA_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    return path


def _touch_frames(root: Path, names):
    for name in names:
        path = root / f"{name}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def _fixture(tmp_path, monkeypatch):
    _write_metadata(tmp_path, monkeypatch)
    _touch_frames(
        tmp_path,
        [
            f"{SEQ_A}/1469223181_+00060",
            f"{SEQ_A}/1469223241_+00120",
            f"{SEQ_A}/1469223301_+00180",  # unannotated smoke frame: never used
            f"{SEQ_A}/1469220001_-01200",
            f"{SEQ_A}/1469220061_-01140",
            f"{SEQ_A}/1469220121_-01080",
            f"{SEQ_B}/1495300000_+00000",
            f"{SEQ_B}/1495299940_-00060",
        ],
    )
    (tmp_path / SEQ_A / "notes.txt").write_text("ignored")
    return tmp_path


def test_boxes_load_as_plain_ints(tmp_path, monkeypatch):
    path = _write_metadata(tmp_path, monkeypatch)
    boxes = load_smokeynet_boxes(path)
    assert boxes[f"{SEQ_A}/1469223181_+00060"] == [[1821, 1027, 1922, 1074]]
    assert boxes[f"{SEQ_A}/1469223241_+00120"] == [[1807, 1031, 1926, 1074]]
    assert all(type(v) is int for box in boxes[f"{SEQ_A}/1469223181_+00060"] for v in box)
    monkeypatch.setattr(figlib, "SMOKEYNET_METADATA_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        load_smokeynet_boxes(path)
    assert len(load_smokeynet_boxes(path, verify=False)) == 3


def test_metadata_unpickler_refuses_code(tmp_path):
    class Exploit:
        def __reduce__(self):
            return (print, ("pwned",))

    path = tmp_path / "evil.pkl"
    path.write_bytes(pickle.dumps({"bbox_labels": Exploit()}))
    with pytest.raises(pickle.UnpicklingError, match="refusing"):
        load_smokeynet_boxes(path, verify=False)


def test_dataset_lists_smoke_and_sampled_smoke_free_frames(tmp_path, monkeypatch):
    root = _fixture(tmp_path, monkeypatch)
    data = FIgLibSmokeBench(root, negatives=2, negative_seed=0)
    assert [s.image_id for s in data.positives] == [
        f"{SEQ_A}/1469223181_+00060",
        f"{SEQ_A}/1469223241_+00120",
        f"{SEQ_B}/1495300000_+00000",
    ]
    assert data.positives[0].boxes == ((1821, 1027, 1922, 1074),)
    assert len(data.negatives) == 2 and data.negative_pool_size == 4
    assert all(not s.label and s.boxes == () and s.offset_seconds < 0 for s in data.negatives)
    assert [s.image_id for s in FIgLibSmokeBench(root, negatives=2, negative_seed=0).negatives] == [s.image_id for s in data.negatives]
    assert len(data) == 5 and data[0] is data.positives[0]
    assert data.positives[2].offset_seconds == 0 and data.positives[1].offset_seconds == 120
    everything = FIgLibSmokeBench(root, negatives=100)
    assert len(everything.negatives) == 4  # capped at the pool
    only_b = FIgLibSmokeBench(root, sequences=[SEQ_B])
    assert [s.image_id for s in only_b] == [f"{SEQ_B}/1495300000_+00000", f"{SEQ_B}/1495299940_-00060"]
    assert FIgLibSmokeBench(root, negatives=0).negatives == []
    info = data.describe()
    assert info["smoke_images"] == 3 and info["smoke_free_images"] == 2 and info["sequences"] == 2


def test_dataset_errors(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError, match="download_smokebench"):
        FIgLibSmokeBench(tmp_path)
    root = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="without SmokeyNet boxes"):
        FIgLibSmokeBench(root, sequences=["20990101_FIRE_none"])
    with pytest.raises(ValueError, match="negatives"):
        FIgLibSmokeBench(root, negatives=-1)
    (root / SEQ_B / "1495300000_+00000.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="1 annotated frames are missing"):
        FIgLibSmokeBench(root)


def _archive(path: Path, sequence: str) -> None:
    with tarfile.open(path, "w:gz") as tar:
        def add(name: str, data: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        add(f"{sequence}/1469223181_+00060.jpg", b"jpeg-1")
        add(f"{sequence}/1469220001_-01200.jpg", b"jpeg-2")
        add(f"{sequence}/{sequence}.mp4", b"video")
        add(f"{sequence}/sub/1469223181_+00060.jpg", b"nested")
        add("../1469223181_+00060.jpg", b"escape")
        add("other_sequence/1469223181_+00060.jpg", b"other")


def test_download_streams_only_frames(tmp_path, monkeypatch):
    server = tmp_path / "server"
    server.mkdir()
    _archive(server / f"{SEQ_A}.tgz", SEQ_A)
    monkeypatch.setattr(figlib, "FIGLIB_ARCHIVE_URL", server.as_uri() + "/{sequence}.tgz")
    root = tmp_path / "data"
    assert figlib.download_sequence(root, SEQ_A) == 2
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    assert files == [f"{SEQ_A}/.complete", f"{SEQ_A}/1469220001_-01200.jpg", f"{SEQ_A}/1469223181_+00060.jpg"]
    assert (root / SEQ_A / "1469223181_+00060.jpg").read_bytes() == b"jpeg-1"
    (server / f"{SEQ_A}.tgz").unlink()
    assert figlib.download_sequence(root, SEQ_A) == 2  # marker: no second download
    with pytest.raises(ValueError, match="invalid"):
        figlib.download_sequence(root, "../etc")


def test_download_metadata_checks_the_pin(tmp_path, monkeypatch):
    server = tmp_path / "server"
    server.mkdir()
    (server / "metadata.pkl").write_bytes(pickle.dumps({"bbox_labels": {}}))
    monkeypatch.setattr(figlib, "SMOKEYNET_METADATA_URL", (server / "metadata.pkl").as_uri())
    monkeypatch.setattr(figlib, "SMOKEYNET_METADATA_SHA256", hashlib.sha256((server / "metadata.pkl").read_bytes()).hexdigest())
    path = figlib.download_metadata(tmp_path / "data")
    assert path.name == figlib.METADATA_FILENAME and load_smokeynet_boxes(path) == {}
    monkeypatch.setattr(figlib, "SMOKEYNET_METADATA_SHA256", "1" * 64)
    with pytest.raises(ValueError, match="sha256"):
        figlib.download_metadata(tmp_path / "other")


def test_sample_offsets_and_image_loading(tmp_path):
    sample = SmokeSample("seq/1469220001_-01200", tmp_path / "x.jpg", False)
    assert sample.sequence == "seq" and sample.offset_seconds == -1200
    pil = pytest.importorskip("PIL.Image")
    array = (np.arange(6 * 8 * 3) % 256).reshape(6, 8, 3).astype(np.uint8)
    pil.fromarray(array).save(tmp_path / "x.png")
    loaded = FIgLibSmokeBench.load_image(SmokeSample("seq/1_+00000", tmp_path / "x.png", True, ((0, 0, 1, 1),)))
    assert loaded.dtype == np.uint8 and np.array_equal(loaded, array)
