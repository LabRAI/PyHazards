hydrographnet
=============

Description
-----------

``hydrographnet`` is a physics-informed graph neural network for flood forecasting with
interpretable KAN-style components, residual message passing, and delta-state decoding.

The model summary follows Taghizadeh et al. (2025):
`Interpretable physics-informed graph neural networks for flood forecasting <https://onlinelibrary.wiley.com/doi/10.1111/mice.13484>`_.

In PyHazards, this module is built from the model registry and is typically used with the
ERA5-based hydrograph adapter ``load_hydrograph_data`` for end-to-end validation.

Example of how to use it
------------------------

.. code-block:: python

   import torch
   from pyhazards.data.load_hydrograph_data import load_hydrograph_data
   from pyhazards.datasets import graph_collate
   from pyhazards.models import build_model

   data = load_hydrograph_data("pyhazards/data/era5_subset", max_nodes=50)
   sample = data.splits["train"].inputs[0]
   batch = graph_collate([sample])

   model = build_model(
       name="hydrographnet",
       task="regression",
       node_in_dim=2,
       edge_in_dim=3,
       out_dim=1,
   )

   y = model(batch)
   print(y.shape)
