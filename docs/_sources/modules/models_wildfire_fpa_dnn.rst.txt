wildfire_fpa_dnn
================

Description
-----------

``wildfire_fpa_dnn`` is the paper-facing DNN classification component from the FPA-FOD wildfire
framework. In PyHazards it is implemented as a configurable multilayer perceptron for incident-level
cause or size-class prediction.

This corresponds to the classification stage described in:
`Prediction of the cause and size of wildfire using artificial intelligence <https://www.sciencedirect.com/science/article/pii/S2949926723000033>`_.

Example of how to use it
------------------------

.. code-block:: python

   import torch
   from pyhazards.models import build_model

   model = build_model(
       name="wildfire_fpa_dnn",
       task="classification",
       in_dim=8,
       out_dim=5,
       hidden_dim=64,
       depth=2,
   )

   x = torch.randn(4, 8)
   logits = model(x)
   print(logits.shape)  # (4, 5)

Notes
-----

- ``wildfire_fpa_dnn`` is the primary PyHazards registry name for this classifier.
