hydrographnet
=============

Description
-----------

``hydrographnet`` is a physics-inspired graph neural network for flood forecasting with:

- KAN-style node encoding
- residual message-passing processor blocks
- residual delta-state decoding (depth/volume style update)
- optional autoregressive rollout via ``predict_steps``

Example usage
-------------

.. code-block:: python

   import torch
   from pyhazards.models import build_model

   model = build_model(
       name="hydrographnet",
       task="regression",
       node_in_dim=2,
       edge_in_dim=3,
       out_dim=1,
   )

   batch = {
       "x": torch.randn(1, 3, 6, 2),
       "adj": torch.eye(6).unsqueeze(0),
       "coords": torch.randn(6, 2),
       "predict_steps": 2,
   }
   y = model(batch)
   print(y.shape)  # (1, 2, 6, 1)
