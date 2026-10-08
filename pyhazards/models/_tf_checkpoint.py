"""Read TensorFlow (V2 "tensor bundle") checkpoints without TensorFlow.

A checkpoint ``<prefix>`` consists of ``<prefix>.index`` and ``<prefix>.data-NNNNN-of-MMMMM``. The index
is an SSTable (LevelDB table format: data blocks of prefix-compressed key/value entries, an index block
and a 48-byte footer) that maps each variable name to a ``BundleEntryProto`` (dtype, shape, shard,
offset, size); the data files hold the raw little-endian tensor bytes. This reader supports the
uncompressed, single-slice layout that ``tf.train.Saver`` writes, which is what the official PhaseNet
checkpoint uses. It was checked against ``tf.train.load_checkpoint`` (TensorFlow 2.11) in
``tests/oracle/test_phasenet_oracle.py``.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Union

import numpy as np

_MAGIC = 0xDB4775248B80FB57
_DTYPES = {1: "<f4", 2: "<f8", 3: "<i4", 4: "u1", 5: "<i2", 6: "i1", 9: "<i8", 10: "?", 19: "<f2"}


def _varint(buffer: bytes, pos: int) -> Tuple[int, int]:
    result, shift = 0, 0
    while True:
        byte = buffer[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _block(data: bytes, offset: int, size: int) -> bytes:
    compression = data[offset + size]
    if compression != 0:
        raise ValueError("Compressed TensorFlow checkpoint index blocks are not supported.")
    return data[offset : offset + size]


def _entries(block: bytes) -> Iterator[Tuple[bytes, bytes]]:
    num_restarts = struct.unpack_from("<I", block, len(block) - 4)[0]
    end = len(block) - 4 - 4 * num_restarts
    pos, key = 0, b""
    while pos < end:
        shared, pos = _varint(block, pos)
        non_shared, pos = _varint(block, pos)
        value_length, pos = _varint(block, pos)
        key = key[:shared] + block[pos : pos + non_shared]
        pos += non_shared
        yield key, block[pos : pos + value_length]
        pos += value_length


def _fields(message: bytes) -> Iterator[Tuple[int, int, Union[int, bytes]]]:
    pos = 0
    while pos < len(message):
        tag, pos = _varint(message, pos)
        number, wire = tag >> 3, tag & 7
        if wire == 0:
            value, pos = _varint(message, pos)
        elif wire == 1:
            value, pos = message[pos : pos + 8], pos + 8
        elif wire == 2:
            length, pos = _varint(message, pos)
            value, pos = message[pos : pos + length], pos + length
        elif wire == 5:
            value, pos = message[pos : pos + 4], pos + 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}.")
        yield number, wire, value


def _bundle_entry(value: bytes) -> Dict[str, object]:
    entry: Dict[str, object] = {"dtype": 0, "shape": [], "shard_id": 0, "offset": 0, "size": 0, "slices": False}
    for number, _, field in _fields(value):
        if number == 1:
            entry["dtype"] = field
        elif number == 2:  # TensorShapeProto: repeated Dim (field 2) with size (field 1)
            dims: List[int] = []
            for dim_number, _, dim in _fields(field):  # type: ignore[arg-type]
                if dim_number == 2:
                    size = 0
                    for size_number, _, size_value in _fields(dim):  # type: ignore[arg-type]
                        if size_number == 1:
                            size = int(size_value)  # type: ignore[arg-type]
                    dims.append(size)
            entry["shape"] = dims
        elif number == 3:
            entry["shard_id"] = field
        elif number == 4:
            entry["offset"] = field
        elif number == 5:
            entry["size"] = field
        elif number == 7:
            entry["slices"] = True
    return entry


def read_tf_checkpoint(prefix: Union[str, Path]) -> Dict[str, np.ndarray]:
    """All numeric variables of a TensorFlow V2 checkpoint, keyed by variable name."""
    prefix = Path(prefix)
    index = (prefix.parent / (prefix.name + ".index")).read_bytes()
    footer = index[-48:]
    if struct.unpack("<Q", footer[-8:])[0] != _MAGIC:
        raise ValueError(f"{prefix}.index is not a TensorFlow checkpoint index.")
    _, pos = _varint(footer, 0)  # metaindex handle (unused)
    _, pos = _varint(footer, pos)
    index_offset, pos = _varint(footer, pos)
    index_size, pos = _varint(footer, pos)

    entries: Dict[str, Dict[str, object]] = {}
    num_shards = 1
    for _, handle in _entries(_block(index, index_offset, index_size)):
        offset, hpos = _varint(handle, 0)
        size, _ = _varint(handle, hpos)
        for key, value in _entries(_block(index, offset, size)):
            if key == b"":  # BundleHeaderProto
                for number, _, field in _fields(value):
                    if number == 1:
                        num_shards = int(field)  # type: ignore[arg-type]
                continue
            entries[key.decode("utf-8")] = _bundle_entry(value)

    shards: Dict[int, bytes] = {}
    tensors: Dict[str, np.ndarray] = {}
    for name, entry in entries.items():
        dtype = _DTYPES.get(int(entry["dtype"]))  # type: ignore[arg-type]
        if dtype is None or entry["slices"]:
            continue  # strings and partitioned variables are not needed here
        shard = int(entry["shard_id"])  # type: ignore[arg-type]
        if shard not in shards:
            shards[shard] = (prefix.parent / f"{prefix.name}.data-{shard:05d}-of-{num_shards:05d}").read_bytes()
        start, size = int(entry["offset"]), int(entry["size"])  # type: ignore[arg-type]
        array = np.frombuffer(shards[shard][start : start + size], dtype=dtype)
        tensors[name] = array.reshape(entry["shape"]).copy()  # type: ignore[arg-type]
    return tensors


__all__ = ["read_tf_checkpoint"]
