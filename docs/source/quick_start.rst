Quick Start
=================
This page is the shortest end-to-end path through PyHazards: verify the install,
inspect one dataset, build one model, and run one short training loop.

Step 1: Verify the Package
--------------------------

Confirm that Python can import the package:

.. code-block:: bash

    python -c "import pyhazards; print(pyhazards.__version__)"

Step 2: Inspect Example Data
----------------------------

Use the ERA5 inspection entrypoint to validate the bundled sample data:

.. code-block:: bash

    python -m pyhazards.datasets.era5.inspection --path pyhazards/data/era5_subset --max-vars 10

Step 3: Build a Model
---------------------

Build ``hydrographnet`` from the model registry:

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

This example uses the ERA5 subset plus ``hydrographnet`` to confirm that the
dataset, model, and training engine work together.

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

Step 5: Choose What to Explore Next
-----------------------------------

- Go to :doc:`pyhazards_datasets` if you want to browse supported datasets.
- Go to :doc:`pyhazards_models` if you want to compare built-in models.
- Go to :doc:`implementation` if you want to add your own dataset or model.

Device Notes
------------

PyHazards uses CUDA automatically when available. To force a device:

.. code-block:: bash

    export PYHAZARDS_DEVICE=cuda:0

.. code-block:: python

    from pyhazards.utils import set_device

    set_device("cuda:0")
    set_device("cpu")
