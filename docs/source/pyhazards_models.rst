Models
===================

Summary
-------

PyHazards provides a lightweight, extensible model architecture with:

- Backbones for common data types: MLP (tabular), CNN patch encoder (raster), temporal encoder (time-series).
- Task heads: classification, regression, segmentation.
- A registry-driven builder so you can construct built-ins by name or register your own.
- Hazard-focused implementations such as ``wildfire_aspp`` and ``hydrographnet``.

Model
-----

We implemented different hazard prediction models for flood, wildfire, earthquake, weather, and more.
Click a module name to open its detail page with summary and usage example.

Wildfire
~~~~~~~~

.. list-table::
   :widths: 15 85
   :header-rows: 1
   :class: dataset-list

   * - Module
     - Description
   * - :doc:`wildfire_aspp <modules/models_wildfire_aspp>`
     - An explainable CNN model with an ASPP mechanism (CNN-ASPP) for next-day wildfire spread prediction using environmental variables from the Next Day Wildfire Spread dataset; compared against RF, SVM, ANN, and a baseline CNN. See `Marjani et al. (2024) <https://ieeexplore.ieee.org/document/10568207>`_.

Flood
~~~~~

.. list-table::
   :widths: 15 85
   :header-rows: 1
   :class: dataset-list

   * - Module
     - Description
   * - :doc:`hydrographnet <modules/models_hydrographnet>`
     - A novel physics-informed GNN framework that integrates the Kolmogorov-Arnold Network (KAN) to enhance interpretability for unstructured mesh-based flood forecasting. See `Taghizadeh et al. (2025) <https://onlinelibrary.wiley.com/doi/10.1111/mice.13484>`_.

Build and register custom model
-------------------------------

Build a built-in model
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyhazards.models import build_model

    model = build_model(
        name="mlp",
        task="classification",
        in_dim=32,
        out_dim=5,
        hidden_dim=256,
        depth=3,
    )

Register a custom model
~~~~~~~~~~~~~~~~~~~~~~~

Create a builder function that returns an ``nn.Module`` and register it with a name. The registry handles defaults and discoverability.

.. code-block:: python

    import torch.nn as nn
    from pyhazards.models import register_model, build_model

    def my_custom_builder(task: str, in_dim: int, out_dim: int, **kwargs) -> nn.Module:
        hidden = kwargs.get("hidden_dim", 128)
        layers = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        return layers

    register_model("my_mlp", my_custom_builder, defaults={"hidden_dim": 128})

    model = build_model(name="my_mlp", task="regression", in_dim=16, out_dim=1)

Design notes
~~~~~~~~~~~~

- Builders receive ``task`` plus any kwargs you pass; use this to switch heads internally if needed.
- ``register_model`` stores optional defaults so you can keep CLI/configs minimal.
- Models are plain PyTorch modules, so you can compose them with the ``Trainer`` or your own loops.

.. toctree::
   :maxdepth: 1
   :hidden:

   modules/models_wildfire_aspp
   modules/models_hydrographnet
