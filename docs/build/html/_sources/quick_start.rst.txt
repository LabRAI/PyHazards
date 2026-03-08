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

Instantiate ``hydrographnet`` through the unified model registry:

.. code-block:: python

    from pyhazards.models import build_model

    model = build_model(
        name="hydrographnet",
        task="regression",
        node_in_dim=2,
        edge_in_dim=3,
        out_dim=1,
    )
    print(type(model).__name__)

Step 4: Run a Short Train/Evaluate Loop
---------------------------------------

This example pairs the ERA5 subset with ``hydrographnet`` to confirm that the
dataset, model, and training engine work together in one workflow.

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

    trainer.fit(
        data,
        optimizer=optimizer,
        loss_fn=loss_fn,
        max_epochs=1,
        batch_size=1,
        collate_fn=graph_collate,
    )

    metrics = trainer.evaluate(
        data,
        split="train",
        batch_size=1,
        collate_fn=graph_collate,
    )
    print(metrics)

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
