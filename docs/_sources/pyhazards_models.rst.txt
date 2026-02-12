Models
===================

Summary
-------

PyHazards provides a lightweight, extensible model architecture with:

- Backbones for common data types: MLP (tabular), CNN patch encoder (raster), temporal encoder (time-series).
- Task heads: classification, regression, segmentation.
- A registry-driven builder so you can construct built-ins by name or register your own.

Hazard-focused model tables
---------------------------

Wildfire
~~~~~~~~

.. list-table::
   :widths: 45 55
   :header-rows: 1
   :class: dataset-list

   * - Module to call model
     - Description
   * - ``build_model(name="wildfire_aspp", task="segmentation", in_channels=12)``
     - CNN+ASPP wildfire spread segmentation model. See `Application of Explainable Artificial Intelligence in Predicting Wildfire Spread: An ASPP-Enabled CNN Approach <https://ieeexplore.ieee.org/document/10568207>`_.

Flood
~~~~~

.. list-table::
   :widths: 45 55
   :header-rows: 1
   :class: dataset-list

   * - Module to call model
     - Description
   * - ``build_model(name="hydrographnet", task="regression", node_in_dim=2, edge_in_dim=3, out_dim=1)``
     - Physics-informed/interpretable graph model for flood forecasting. See `Interpretable physics-informed graph neural networks for flood forecasting <https://onlinelibrary.wiley.com/doi/10.1111/mice.13484>`_.

Core modules
------------

- ``pyhazards.models.backbones`` — reusable feature extractors.
- ``pyhazards.models.heads`` — task-specific heads.
- ``pyhazards.models.builder`` — ``build_model(name, task, **kwargs)`` helper plus ``default_builder``.
- ``pyhazards.models.registry`` — ``register_model`` / ``available_models``.
- ``pyhazards.models`` — convenience re-exports and default registrations for ``mlp``, ``cnn``, ``temporal``.

Build a built-in model
----------------------

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
-----------------------

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

Mamba-based wildfire model (spatio-temporal)
--------------------------------------------

PyHazards ships a Mamba-style spatio-temporal model for county-day wildfire prediction using ERA5 features. It couples a selective state-space temporal encoder with a lightweight GCN to mix neighboring counties.

- Input: ``(batch, past_days, num_counties, num_features)`` county-day ERA5 tensors.
- Temporal: stacked selective SSM blocks plus a differential branch to highlight day-to-day changes.
- Spatial: two-layer GCN over a provided adjacency (falls back to identity if none is given).
- Output: per-county logits for the next day (apply ``torch.sigmoid`` for probabilities). Optional count head via ``with_count_head=True``.

Toy usage with random ERA5-like tensors:

.. code-block:: python

    import torch
    from pyhazards.datasets import DataBundle, DataSplit, FeatureSpec, LabelSpec
    from pyhazards.engine import Trainer
    from pyhazards.models import build_model

    past_days = 12
    num_counties = 5
    num_features = 6  # e.g., t2m, d2m, u10, v10, tp, ssr
    samples = 64

    # Fake county-day ERA5 cube and binary fire labels
    x = torch.randn(samples, past_days, num_counties, num_features)
    y = torch.randint(0, 2, (samples, num_counties)).float()
    adjacency = torch.eye(num_counties)  # replace with a distance or correlation matrix

    bundle = DataBundle(
        splits={
            "train": DataSplit(x[:48], y[:48]),
            "val": DataSplit(x[48:], y[48:]),
        },
        feature_spec=FeatureSpec(input_dim=num_features, extra={"past_days": past_days, "counties": num_counties}),
        label_spec=LabelSpec(num_targets=num_counties, task_type="classification"),
    )

    model = build_model(
        name="wildfire_mamba",
        task="classification",
        in_dim=num_features,
        num_counties=num_counties,
        past_days=past_days,
        adjacency=adjacency,
    )

    trainer = Trainer(model=model, mixed_precision=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Fit on the toy data; Trainer works because inputs/targets are plain tensors
    trainer.fit(bundle, optimizer=optimizer, loss_fn=loss_fn, max_epochs=2, batch_size=8)

    # Predict probabilities for the next day
    with torch.no_grad():
        logits = model(x[:1])
        probs = torch.sigmoid(logits)
        print(probs.shape)  # (1, num_counties)

    # For more complex batches (dicts with adjacency), wrap tensors in GraphTemporalDataset
    # and pass graph_collate to Trainer.fit/evaluate/predict.

Design notes
------------

- Builders receive ``task`` plus any kwargs you pass; use this to switch heads internally if needed.
- ``register_model`` stores optional defaults so you can keep CLI/configs minimal.
- Models are plain PyTorch modules, so you can compose them with the ``Trainer`` or your own loops.
