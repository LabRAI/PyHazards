.. raw:: html

   <div style="margin: 30px 0; text-align: center;">
     <img src="_static/logo.png" alt="PyHazards Icon" style="max-width: 260px; height: auto;">
   </div>

.. image:: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fpypi.org%2Fpypi%2Fpyhazards%2Fjson&query=%24.info.version&prefix=v&label=PyPI
   :target: https://pypi.org/project/pyhazards
   :alt: PyPI Version

.. image:: https://img.shields.io/github/actions/workflow/status/LabRAI/PyHazards/ci.yml?branch=main
   :target: https://github.com/LabRAI/PyHazards/actions/workflows/ci.yml
   :alt: Build Status

.. image:: https://img.shields.io/badge/license-MIT-green
   :target: https://github.com/LabRAI/PyHazards/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/badge/downloads-check%20PyPI-blue
   :target: https://pypi.org/project/pyhazards
   :alt: PyPI Downloads

.. image:: https://img.shields.io/github/issues/LabRAI/PyHazards
   :target: https://github.com/LabRAI/PyHazards
   :alt: Issues

.. image:: https://img.shields.io/github/issues-pr/LabRAI/PyHazards
   :target: https://github.com/LabRAI/PyHazards
   :alt: Pull Requests

.. image:: https://img.shields.io/github/stars/LabRAI/PyHazards
   :target: https://github.com/LabRAI/PyHazards
   :alt: Stars

.. image:: https://img.shields.io/github/forks/LabRAI/PyHazards
   :target: https://github.com/LabRAI/PyHazards
   :alt: GitHub forks

.. image:: _static/github.svg
   :target: https://github.com/LabRAI/PyHazards
   :alt: GitHub

----

Introduction
------------

PyHazards is a Python framework for AI-powered hazard prediction and risk assessment. It provides a hazard-first API for loading data, building models, running end-to-end experiments, and extending with your own modules.

Core Components
---------------

Use this as the minimal path: install the package, load one implemented dataset, build one implemented model, then run one end-to-end validation test.

- **Datasets**: Unified interfaces for tabular, temporal, raster, and graph-style hazard data through ``DataBundle``.
- **Models**: Built-in hazard models plus reusable backbones/heads via a registry-driven model API.
- **Engine**: ``Trainer`` for fit/evaluate/predict workflows with mixed precision and distributed options.
- **Metrics and Utilities**: Classification/regression/segmentation metrics, hardware helpers, and reproducibility tools.

Install
-------

Install from PyPI. If you plan to run on GPU, install a compatible PyTorch build first.

.. code-block:: bash

    pip install pyhazards

Load Data
---------

Use the dataset module entrypoint for consistent dataset inspection/loading across datasets.

.. code-block:: bash

    python -m pyhazards.datasets.era5.inspection --path pyhazards/data/era5_subset --max-vars 10

Load Model
----------

Build one implemented model from the registry (this example uses the wildfire model).

Example using ``wildfire_aspp``:

.. code-block:: python

    from pyhazards.models import build_model

    print("[Step 2/3] Building model...")
    model = build_model(
        name="wildfire_aspp",
        task="segmentation",
        in_channels=12,
    )
    print("[Step 2/3] Model built.")
    print(type(model).__name__)

Full Test
---------

Validation example: load the same ERA5-based hydrograph subset and run one epoch with ``hydrographnet``.
This uses ``load_hydrograph_data`` as the model-specific adapter after dataset inspection.

.. code-block:: python

    import torch
    from pyhazards.data.load_hydrograph_data import load_hydrograph_data
    from pyhazards.datasets import graph_collate
    from pyhazards.engine import Trainer
    from pyhazards.models import build_model

    data = load_hydrograph_data("pyhazards/data/era5_subset", max_nodes=50)

    model = build_model(
        name="hydrographnet",
        task="regression",
        node_in_dim=2,
        edge_in_dim=3,
        out_dim=1,
    )

    trainer = Trainer(model=model, mixed_precision=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    print("[Step 3/3] Running one training epoch...")
    trainer.fit(
        data,
        optimizer=optimizer,
        loss_fn=loss_fn,
        max_epochs=1,
        batch_size=1,
        collate_fn=graph_collate,
    )

    print("[Step 3/3] Evaluating on train split...")
    metrics = trainer.evaluate(
        data,
        split="train",
        batch_size=1,
        collate_fn=graph_collate,
    )
    print(metrics)

Quick Verification (``test.py``)
--------------------------------

Run the built-in GPU smoke test:

.. code-block:: bash

    python test.py

``test.py`` is a validation/smoke test only. It verifies pipeline correctness and integration, not final benchmark performance.

It prints step-by-step status and ends with:

.. code-block:: text

    PASS: end-to-end implementation is working.

Custom Module
-------------

Use this when you want to add your own dataset/model implementation into PyHazards.

To upload and use your own data/model modules:

1. Upload your raw data files to your project path and write a dataset loader that returns a ``DataBundle``.
2. Register your model with ``register_model`` and a builder function that returns an ``nn.Module``.
3. Build with ``build_model(...)`` and train/evaluate through ``Trainer``.

For implementation details, see :doc:`implementation`, :doc:`pyhazards_datasets`, and :doc:`pyhazards_models`.

How to Cite
-----------

If you use PyHazards in your research, please cite:

.. code-block:: bibtex

   @misc{pyhazards2025,
     title        = {PyHazards: An Open-Source Library for AI-Powered Hazard Prediction},
     author       = {Cheng, Xueqi and Xu, Yangshuang and Xu, Runyang and Schneier, Lex and Kodudula, Sharan Kumar Reddy and Hsu, Deyang and Shen, Dacheng and Dong, Yushun},
     year         = {2025},
     howpublished = {\url{https://github.com/LabRAI/PyHazards}},
     note         = {GitHub repository}
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
