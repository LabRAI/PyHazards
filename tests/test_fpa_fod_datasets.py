import torch

from pyhazards.datasets import load_dataset


def test_fpa_fod_tabular_micro_shapes():
    bundle = load_dataset("fpa_fod_tabular", micro=True, task="cause").load()
    train = bundle.get_split("train")

    assert train.inputs.ndim == 2
    assert train.targets.ndim == 1
    assert train.inputs.dtype == torch.float32
    assert train.targets.dtype == torch.long
    assert train.inputs.shape[0] == train.targets.shape[0]
    assert bundle.feature_spec.input_dim == train.inputs.shape[1]


def test_fpa_fod_weekly_micro_shapes():
    bundle = load_dataset(
        "fpa_fod_weekly",
        micro=True,
        lookback_weeks=12,
        features="counts+time",
    ).load()
    train = bundle.get_split("train")

    assert train.inputs.ndim == 3
    assert train.targets.ndim == 2
    assert train.inputs.dtype == torch.float32
    assert train.targets.dtype == torch.float32
    assert train.inputs.shape[0] == train.targets.shape[0]
    assert train.inputs.shape[1] == 12
    assert bundle.feature_spec.input_dim == train.inputs.shape[2]
