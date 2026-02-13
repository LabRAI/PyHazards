wildfire_aspp
=============

Description
-----------

``wildfire_aspp`` is an explainable CNN segmentation model with an ASPP mechanism for next-day
wildfire spread prediction.

The model summary follows Marjani et al. (2024):
`Application of Explainable Artificial Intelligence in Predicting Wildfire Spread: An ASPP-Enabled CNN Approach <https://ieeexplore.ieee.org/document/10568207>`_.

In PyHazards, this module is built from the model registry and can be used directly for wildfire
segmentation experiments.

Example of how to use it
------------------------

.. code-block:: python

   import torch
   from pyhazards.models import build_model

   model = build_model(
       name="wildfire_aspp",
       task="segmentation",
       in_channels=12,
   )

   x = torch.randn(2, 12, 64, 64)
   logits = model(x)
   print(logits.shape)  # (2, 1, 64, 64)
