from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import h5py
import numpy as np

from dataloader.schema import Sample


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    return value


def _decode_h5_value(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("S", "U", "O"):
            out = []
            for v in value.tolist():
                if isinstance(v, bytes):
                    out.append(v.decode("utf-8"))
                else:
                    out.append(v)
            return out
        return value
    return value


def save_sample_h5(sample: Sample, output_path: str) -> str:
    """Save a loaded sample to HDF5.

    Stores:
    - /x : feature tensor, shape (T, C, H, W)
    - /y : label tensor, shape (T, H, W)
    - /meta/<key> : ndarray metadata when possible
    - /meta_json : fallback JSON metadata string
    """
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out, "w") as f:
        f.create_dataset("x", data=sample.x, compression="gzip")
        f.create_dataset("y", data=sample.y, compression="gzip")

        meta_group = f.create_group("meta")
        meta_json: dict[str, Any] = {}
        for k, v in sample.meta.items():
            if isinstance(v, np.ndarray):
                meta_group.create_dataset(str(k), data=v, compression="gzip")
            elif isinstance(v, list) and all(isinstance(i, str) for i in v):
                dt = h5py.string_dtype(encoding="utf-8")
                meta_group.create_dataset(str(k), data=np.array(v, dtype=dt))
            else:
                meta_json[str(k)] = _to_serializable(v)

        f.create_dataset("meta_json", data=json.dumps(meta_json, ensure_ascii=True))

    return str(out)


def load_sample_h5(input_path: str) -> Sample:
    """Load a Sample object from HDF5 produced by `save_sample_h5`."""
    path = Path(input_path).expanduser().resolve()
    with h5py.File(path, "r") as f:
        x = f["x"][()]
        y = f["y"][()]
        meta: dict[str, Any] = {}

        if "meta" in f:
            for k in f["meta"].keys():
                meta[k] = _decode_h5_value(f["meta"][k][()])

        if "meta_json" in f:
            raw = f["meta_json"][()]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if raw:
                decoded = json.loads(raw)
                for k, v in decoded.items():
                    if k not in meta:
                        meta[k] = v

    return Sample(x=x, y=y, meta=meta)


def to_torch_batch(sample: Sample):
    """Convert Sample arrays to torch tensors: returns (x, y, meta)."""
    import torch

    x = torch.from_numpy(sample.x)
    y = torch.from_numpy(sample.y)
    return x, y, sample.meta
