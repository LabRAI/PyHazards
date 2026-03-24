FireCastNet
===========

Description
-----------

``firecastnet`` is a lightweight PyHazards port of the FireCastNet model family.

This module keeps the main benchmark-relevant ideas needed for integration:

- compact wildfire-risk raster encoder
- dense decoding head
- forecasting-oriented wildfire output map

Paper / source
--------------

- `FireCastNet paper <https://doi.org/10.1038/s41598-025-30645-7>`_

Paper parity note
-----------------

This PyHazards implementation is intentionally **not** a full reproduction of
the original FireCastNet seasonal graph pipeline. Instead, it is a clean
benchmark-facing neural port that preserves the forecasting-oriented wildfire
modeling role needed for PyHazards integration.

It does not claim architecture or preprocessing parity with the original release.

Example of how to use it
------------------------

.. code-block:: python

   import torch
   from pyhazards.models import build_model

   model = build_model(
       name="firecastnet",
       task="segmentation",
       in_channels=12,
       out_channels=1,
   )

   x = torch.randn(2, 12, 32, 32)
   logits = model(x)
   print(logits.shape)
