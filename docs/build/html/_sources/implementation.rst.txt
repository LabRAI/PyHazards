Implementation Guide
====================

Summary
-------

PyHazards is modular and registry-driven. This guide shows how to add your own datasets and models in line
with the hazard-first architecture.

Datasets
--------

Implement a dataset by subclassing ``Dataset`` and returning a ``DataBundle`` from ``_load()``. Register it so users can load by name.

.. code-block:: python

    import torch
    from pyhazards.datasets import (
        DataBundle, DataSplit, Dataset, FeatureSpec, LabelSpec, register_dataset
    )

    class MyHazard(Dataset):
        name = "my_hazard"

        def _load(self):
            x = torch.randn(1000, 16)
            y = torch.randint(0, 2, (1000,))
            splits = {
                "train": DataSplit(x[:800], y[:800]),
                "val": DataSplit(x[800:900], y[800:900]),
                "test": DataSplit(x[900:], y[900:]),
            }
            return DataBundle(
                splits=splits,
                feature_spec=FeatureSpec(input_dim=16, description="example features"),
                label_spec=LabelSpec(num_targets=2, task_type="classification"),
            )

    register_dataset(MyHazard.name, MyHazard)

Models
------

Use the provided backbones (MLP, CNN patch encoder, temporal encoder) and task heads (classification, regression, segmentation) via ``build_model``. To add a custom model, register a builder:

.. code-block:: python

    import torch.nn as nn
    from pyhazards.models import register_model

    def my_model_builder(task: str, in_dim: int, out_dim: int, **kwargs) -> nn.Module:
        # Simple example: a two-layer MLP for classification/regression
        hidden = kwargs.get("hidden_dim", 128)
        layers = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        return layers

    register_model("my_mlp", my_model_builder, defaults={"hidden_dim": 128})
