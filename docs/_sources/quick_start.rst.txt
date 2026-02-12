Quick Start
=================
This guide will help you get started with PyHazards quickly using the hazard-first API.

Basic Usage
-----------

How to load data (real ERA5 subset for flood modeling)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyhazards.data.load_hydrograph_data import load_hydrograph_data

    # Uses bundled NetCDF files under pyhazards/data/era5_subset
    data = load_hydrograph_data(
        era5_path="pyhazards/data/era5_subset",
        max_nodes=50,
    )

    print(data.feature_spec)
    print(data.label_spec)
    print(data.splits.keys())  # dict_keys(["train"])

How to load models
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyhazards.models import build_model

    # Wildfire model (ASPP-enabled CNN)
    wildfire_model = build_model(
        name="wildfire_aspp",
        task="segmentation",
        in_channels=12,
    )

    # Flood model (HydroGraphNet)
    flood_model = build_model(
        name="hydrographnet",
        task="regression",
        node_in_dim=2,
        edge_in_dim=3,
        out_dim=1,
    )

    print(type(wildfire_model).__name__)
    print(type(flood_model).__name__)

End-to-end example (real data + implemented flood model)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
        max_epochs=20,
        batch_size=1,
        collate_fn=graph_collate,
    )

    # This DataBundle currently has only the "train" split
    train_metrics = trainer.evaluate(
        data,
        split="train",
        batch_size=1,
        collate_fn=graph_collate,
    )
    print(train_metrics)

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
