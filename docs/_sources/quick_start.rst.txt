Quick Start
=================
This guide will help you get started with PyHazards quickly using the hazard-first API.

Basic Usage
-----------

Use the following minimal workflow to get started quickly: load one dataset, build one model, then run a short end-to-end test.

Load Dataset (ERA5 example)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
This example uses the canonical ERA5 dataset module entrypoint to inspect/validate local ERA5 files.
This keeps dataset usage aligned with other dataset modules in PyHazards.

.. code-block:: bash

    python -m pyhazards.datasets.era5.inspection --path pyhazards/data/era5_subset --max-vars 10

Load Model (HydroGraphNet example)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This example builds the implemented ``hydrographnet`` model for graph-based flood regression.

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

Full Test (ERA5 + HydroGraphNet)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This short script runs one training epoch and then evaluates on the available split to verify the end-to-end pipeline.
It uses ``load_hydrograph_data`` as the model-specific ERA5-to-graph adapter for HydroGraphNet.

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

GPU Support
-----------

PyHazards automatically detects CUDA availability. To explicitly set the device:

**Using Environment Variable:**

.. code-block:: bash

    export PYHAZARDS_DEVICE=cuda:0

**Using Python API:**

.. code-block:: python

    from pyhazards.utils import set_device

    # Set to use CUDA device 0
    set_device("cuda:0")

    # Or use CPU
    set_device("cpu")

Next Steps
----------

For more detailed documentation, please refer to:

- :doc:`pyhazards_datasets` - Dataset interface and registration
- :doc:`pyhazards_utils` - Utility functions and helpers
- :doc:`implementation` - Guide for implementing custom datasets and models
