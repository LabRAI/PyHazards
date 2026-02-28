import numpy as np
import torch

from pyhazards.datasets import load_dataset


def _to_bundle(ds):
    if hasattr(ds, "load") and callable(getattr(ds, "load")):
        return ds.load()
    if hasattr(ds, "_load") and callable(getattr(ds, "_load")):
        return ds._load()
    raise TypeError(
        f"{ds.__class__.__name__} does not expose load() or _load(). "
        "Cannot obtain a DataBundle."
    )


def _as_tensor(a):
    if isinstance(a, torch.Tensor):
        return a
    if isinstance(a, np.ndarray):
        return torch.from_numpy(a)
    # allow python lists
    return torch.tensor(a)


def _split_xy(split):
    """
    Extract x,y from:
      - dict: {"x": ..., "y": ...}
      - object with attributes .x/.y etc
      - tuple/list (x,y)
    """
    if isinstance(split, dict):
        # support common key variants
        for xk in ("x", "X", "inputs", "features", "data"):
            for yk in ("y", "Y", "targets", "labels"):
                if xk in split and yk in split:
                    return split[xk], split[yk]
        raise KeyError(f"Split dict missing x/y keys. Keys={sorted(split.keys())}")

    for x_name in ("x", "X", "inputs", "features", "data"):
        for y_name in ("y", "Y", "targets", "labels"):
            if hasattr(split, x_name) and hasattr(split, y_name):
                return getattr(split, x_name), getattr(split, y_name)

    if isinstance(split, (tuple, list)) and len(split) == 2:
        return split[0], split[1]

    raise TypeError(f"Unsupported split type for extracting (x,y): {type(split)}")


def test_dataset_tabular_shapes():
    ds = load_dataset("wildfire_fpa_fod_tabular", micro=True, seed=1337)
    bundle = _to_bundle(ds)

    assert hasattr(bundle, "splits"), "Expected DataBundle.splits"
    assert isinstance(bundle.splits, dict)
    assert "train" in bundle.splits

    x, y = _split_xy(bundle.splits["train"])
    x = _as_tensor(x).float()
    y = _as_tensor(y).long()

    # Contract: tabular classification
    assert x.ndim == 2  # [N, D]
    assert x.dtype == torch.float32
    assert y.ndim == 1  # [N]
    assert y.dtype in (torch.int64, torch.long)

    assert x.shape[0] == y.shape[0]
    assert x.shape[1] > 0


def test_dataset_weekly_shapes():
    ds = load_dataset("wildfire_fpa_fod_weekly", micro=True, seed=1337, region="US")
    bundle = _to_bundle(ds)

    assert hasattr(bundle, "splits"), "Expected DataBundle.splits"
    assert isinstance(bundle.splits, dict)
    assert "train" in bundle.splits

    x, y = _split_xy(bundle.splits["train"])
    x = _as_tensor(x).float()
    y = _as_tensor(y).float()

    # Contract: weekly sequences
    assert x.ndim == 3  # [N, T, D]
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert y.ndim in (1, 2)  # [N] or [N, O]

    assert x.shape[0] == y.shape[0]
    assert x.shape[1] > 0  # T
    assert x.shape[2] > 0  # D
