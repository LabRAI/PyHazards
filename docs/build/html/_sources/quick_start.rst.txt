Quick Start
=================
This guide will help you get started with PyHazard quickly.

Basic Usage
-----------

Loading a Dataset
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyhazard.datasets import Cora

    # Load the Cora dataset with PyTorch Geometric API
    dataset = Cora(api_type='pyg')

    print(f"Number of nodes: {dataset.num_nodes}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")

Using Graph Neural Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pyhazard
    from pyhazard.datasets import Cora
    from pyhazard.models.nn import GCN
    from pyhazard.utils import get_device

    # Load dataset
    dataset = Cora(api_type='pyg')

    # Initialize model
    device = get_device()
    model = GCN(
        input_dim=dataset.num_features,
        hidden_dim=64,
        output_dim=dataset.num_classes
    ).to(device)

    # Your training code here
    # ...

GPU Support
-----------

PyHazard automatically detects CUDA availability. To explicitly set the device:

**Using Environment Variable:**

.. code-block:: bash

    export PYHAZARD_DEVICE=cuda:0

**Using Python API:**

.. code-block:: python

    from pyhazard.utils import set_device

    # Set to use CUDA device 0
    set_device("cuda:0")

    # Or use CPU
    set_device("cpu")

Available Datasets
------------------

PyHazard includes several built-in graph datasets:

- **Cora**: Citation network (2708 nodes, 1433 features, 7 classes)
- **CiteSeer**: Citation network (3327 nodes, 3703 features, 6 classes)
- **PubMed**: Citation network (19717 nodes, 500 features, 3 classes)
- **Computers**: Amazon co-purchase network
- **Photo**: Amazon co-purchase network
- **CoauthorCS**: Co-authorship network (Computer Science)
- **CoauthorPhysics**: Co-authorship network (Physics)

Example with different datasets:

.. code-block:: python

    from pyhazard.datasets import CiteSeer, PubMed, Computers

    # Load CiteSeer
    citeseer = CiteSeer(api_type='pyg')

    # Load PubMed
    pubmed = PubMed(api_type='pyg')

    # Load Computers
    computers = Computers(api_type='pyg')

API Types
---------

PyHazard supports both DGL and PyTorch Geometric APIs:

.. code-block:: python

    from pyhazard.datasets import Cora

    # PyTorch Geometric format
    dataset_pyg = Cora(api_type='pyg')

    # DGL format
    dataset_dgl = Cora(api_type='dgl')

Next Steps
----------

For more detailed documentation, please refer to:

- :doc:`pyhazard_datasets` - Complete dataset API reference
- :doc:`pyhazard_utils` - Utility functions and helpers
- :doc:`implementation` - Guide for implementing custom models
