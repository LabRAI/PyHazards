Implementation Guide
====================

Use this guide when you want to extend PyHazards itself. It is written for
contributors who are adding datasets, models, smoke tests, or generated docs.

Contributor Workflow
--------------------

Most contributions follow the same pattern:

1. add or update a dataset or model implementation,
2. register it in the appropriate subsystem,
3. validate the change with a targeted smoke test,
4. update the relevant documentation page.

Datasets
--------

Implement a dataset by subclassing ``Dataset`` and returning a ``DataBundle``
from ``_load()``. Register it so users can load it by name.

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

Use the model registry when adding a new architecture. The builder should accept
``task`` plus the required shape arguments and return an ``nn.Module``.

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

Documenting a Model Contribution
--------------------------------

Model contributions should stay aligned with the catalog-backed documentation
workflow:

1. add or update ``pyhazards/model_cards/<model_name>.yaml``,
2. run ``python scripts/render_model_docs.py``,
3. verify the model with ``python scripts/smoke_test_models.py --models <model_name>``,
4. keep any public-facing hazard/model descriptions concise and reproducible.

Validation Checklist
--------------------

Before opening a pull request, run the smallest checks that match your change:

.. code-block:: bash

    python -c "import pyhazards; print(pyhazards.__version__)"
    python scripts/render_model_docs.py --check
    python scripts/verify_table_entries.py

If you changed a model implementation, also run:

.. code-block:: bash

    python scripts/smoke_test_models.py --models <model_name>
    python -m pytest tests/test_model_catalog.py

Contributor Notes
-----------------

- Use the registry APIs instead of hard-coding new entrypoints.
- Keep examples minimal and runnable.
- Put user-facing discovery text in docs pages and model cards, not only in code
  comments.
- For repository-specific contributor rules, also see ``.github/IMPLEMENTATION.md``.
