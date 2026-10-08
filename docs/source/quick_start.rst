Quick Start
===========

Use this page after :doc:`installation` to run the first end-to-end PyHazards
workflow: verify the package, inspect example data, build a model, and execute
one short training loop.

Step 1: Verify the Package
--------------------------

Confirm that Python can import the package cleanly:

.. code-block:: bash

    python -c "import pyhazards; print(pyhazards.__version__)"

Step 2: Inspect Example Data
----------------------------

Use the ERA5 inspection entrypoint to validate the bundled sample data before
training:

.. code-block:: bash

    python -m pyhazards.datasets.era5.inspection --path pyhazards/data/era5_subset --max-vars 10

Step 3: Build a Model
---------------------

Instantiate ``hydrographnet`` (PhysicsNeMo's MeshGraphKAN, the HydroGraphNet flood model) through the
unified model registry:

.. code-block:: python

    from pyhazards.models import build_model

    model = build_model(name="hydrographnet", task="regression")
    print(type(model).__name__, sum(p.numel() for p in model.parameters()))  # HydroGraphNet 2318722

Step 4: Run a Short Train/Evaluate Loop
---------------------------------------

This example trains ``hydrographnet`` for one epoch on synthetic mesh hydrographs (the layout of the
HydroGraphNet White River data; use ``hydrographnet_white_river`` with a local copy of the release for
real data) to confirm that the dataset, model, and training engine work together.

.. code-block:: python

    import torch
    from pyhazards.datasets import load_dataset
    from pyhazards.datasets.flood import hydrograph_collate
    from pyhazards.engine import Trainer
    from pyhazards.metrics import RegressionMetrics
    from pyhazards.models import build_model

    data = load_dataset("flood_mesh_synthetic", micro=True).load()
    model = build_model(name="hydrographnet", task="regression")

    trainer = Trainer(model=model, metrics=[RegressionMetrics()], mixed_precision=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    trainer.fit(
        data,
        optimizer=optimizer,
        loss_fn=loss_fn,
        max_epochs=1,
        batch_size=1,
        collate_fn=hydrograph_collate,
    )

    metrics = trainer.evaluate(
        data,
        split="train",
        batch_size=1,
        collate_fn=hydrograph_collate,
    )
    print(metrics)

The flood benchmark scores the test hydrographs with autoregressive rollouts:
``python scripts/run_benchmark.py --config pyhazards/configs/flood/hydrographnet_smoke.yaml``.

Step 5: Next Steps
------------------

- Go to :doc:`pyhazards_datasets` to browse supported datasets.
- Go to :doc:`pyhazards_models` to compare built-in models.
- Go to :doc:`implementation` to add your own dataset or model.

Device Notes
------------

PyHazards uses CUDA automatically when available. To force a device:

.. code-block:: bash

    export PYHAZARDS_DEVICE=cuda:0

.. code-block:: python

    from pyhazards.utils import set_device

    set_device("cuda:0")
    set_device("cpu")
