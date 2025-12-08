.. raw:: html

   <div style="margin: 30px 0; text-align: center;">
     <img src="_static/logo.png" alt="PyHazards Icon" style="max-width: 260px; height: auto;">
   </div>

.. image:: https://img.shields.io/pypi/v/pyhazards
   :target: https://pypi.org/project/pyhazards
   :alt: PyPI Version

.. image:: https://img.shields.io/github/actions/workflow/status/LabRAI/PyHazard/docs.yml
   :target: https://github.com/LabRAI/PyHazard/actions
   :alt: Build Status

.. image:: https://img.shields.io/badge/license-MIT-green
   :target: https://github.com/LabRAI/PyHazard/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/pypi/dm/pyhazards
   :target: https://pypi.org/project/pyhazards
   :alt: PyPI Downloads

.. image:: https://img.shields.io/github/issues/LabRAI/PyHazard
   :target: https://github.com/LabRAI/PyHazard
   :alt: Issues

.. image:: https://img.shields.io/github/issues-pr/LabRAI/PyHazard
   :target: https://github.com/LabRAI/PyHazard
   :alt: Pull Requests

.. image:: https://img.shields.io/github/stars/LabRAI/PyHazard
   :target: https://github.com/LabRAI/PyHazard
   :alt: Stars

.. image:: https://img.shields.io/github/forks/LabRAI/PyHazard
   :target: https://github.com/LabRAI/PyHazard
   :alt: GitHub forks

.. image:: _static/github.svg
   :target: https://github.com/LabRAI/PyHazards
   :alt: GitHub

----

**PyHazards** is a comprehensive Python framework for AI-powered hazard prediction and risk assessment. Built on PyTorch with a hazard-first design, the library provides a modular and extensible architecture for building, training, and deploying machine learning models to predict and analyze natural hazards and environmental risks.

**PyHazards is designed for:**

- **Hazard-First Architecture**: Unified dataset interface for tabular, temporal, and raster data
- **Simple, Extensible Models**: Ready-to-use MLP/CNN/temporal encoders with task heads
- **Trainer API**: Fit/evaluate/predict with optional mixed precision and multi-GPU (DDP) support
- **Metrics**: Classification, regression, and segmentation metrics out of the box
- **Extensibility**: Registries for datasets, models, transforms, and pipelines

**Quick Start Example:**

Basic Usage Example (toy dataset):

.. code-block:: python

    import torch
    from pyhazards.datasets import DataBundle, DataSplit, Dataset, FeatureSpec, LabelSpec
    from pyhazards.models import build_model
    from pyhazards.engine import Trainer
    from pyhazards.metrics import ClassificationMetrics

    class ToyHazard(Dataset):
        def _load(self):
            x = torch.randn(500, 16)
            y = torch.randint(0, 2, (500,))
            splits = {
                "train": DataSplit(x[:350], y[:350]),
                "val": DataSplit(x[350:425], y[350:425]),
                "test": DataSplit(x[425:], y[425:]),
            }
            return DataBundle(
                splits=splits,
                feature_spec=FeatureSpec(input_dim=16, description="toy features"),
                label_spec=LabelSpec(num_targets=2, task_type="classification"),
            )

    data = ToyHazard().load()
    model = build_model(name="mlp", task="classification", in_dim=16, out_dim=2)
    trainer = Trainer(model=model, metrics=[ClassificationMetrics()], mixed_precision=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    trainer.fit(data, optimizer=optimizer, loss_fn=loss_fn, max_epochs=5)
    results = trainer.evaluate(data, split="test")
    print(results)

Wildfire Mamba (spatio-temporal toy)
------------------------------------

Mamba-style county/day wildfire model with a graph-aware Dataset and collate.

.. code-block:: python

    import torch
    from pyhazards.datasets import DataBundle, DataSplit, FeatureSpec, LabelSpec, GraphTemporalDataset, graph_collate
    from pyhazards.engine import Trainer
    from pyhazards.models import build_model

    past_days = 8
    num_counties = 4
    num_features = 6
    samples = 32

    # Fake county/day ERA5-like tensor and binary fire labels
    x = torch.randn(samples, past_days, num_counties, num_features)
    y = torch.randint(0, 2, (samples, num_counties)).float()
    adjacency = torch.eye(num_counties)  # replace with distance/correlation matrix

    train_ds = GraphTemporalDataset(x[:24], y[:24], adjacency=adjacency)
    val_ds = GraphTemporalDataset(x[24:], y[24:], adjacency=adjacency)

    bundle = DataBundle(
        splits={
            "train": DataSplit(train_ds, None),
            "val": DataSplit(val_ds, None),
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

    trainer.fit(bundle, optimizer=optimizer, loss_fn=loss_fn, max_epochs=2, batch_size=4, collate_fn=graph_collate)

    # Next-day fire probabilities for one window
    with torch.no_grad():
        logits = model(x[:1])
        probs = torch.sigmoid(logits)
        print(probs.shape)  # (1, num_counties)

Core Components
---------------

**Datasets**
   PyHazards provides a unified dataset interface for tabular, temporal, and raster data, returning a ``DataBundle`` with splits and specs.

**Models**
   Extensible model architecture with MLP/CNN/temporal backbones and task heads for classification, regression, and segmentation.
   Easy to implement and register custom models via the model registry.

**Utilities**
   Helper functions for device management, seeding/logging, and metrics calculation.

How to Cite
-----------

If you use PyHazards in your research, please cite:

.. code-block:: bibtex

   @software{pyhazards2025,
     title={PyHazards: A Python Framework for AI-Powered Hazard Prediction},
     author={Cheng, Xueqi},
     year={2025},
     url={https://github.com/LabRAI/PyHazards}
   }


.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   installation
   quick_start

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   pyhazards_datasets
   pyhazards_models
   pyhazards_engine
   pyhazards_metrics
   pyhazards_utils

.. toctree::
   :maxdepth: 2
   :caption: Additional Information
   :hidden:

   implementation
   cite
   references
   team


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
