TS-SatFire
==========

Description
-----------

``ts_satfire`` is a lightweight PyHazards port inspired by the TS-SatFire
multi-temporal wildfire prediction benchmark family.

This module keeps the benchmark-relevant ideas we need for integration:

- multi-temporal satellite image sequences
- auxiliary environmental channels
- spatio-temporal raster encoding
- dense wildfire progression prediction

Paper / source
--------------

- `TS-SatFire paper <https://www.nature.com/articles/s41597-025-06271-3>`_
- `TS-SatFire official repository <https://github.com/zhaoyutim/TS-SatFire>`_

Paper parity note
-----------------

This PyHazards implementation is intentionally **not** the entire official
TS-SatFire processing and benchmark stack. Instead, it is a clean spatio-temporal
port that preserves the prediction-task modeling role needed for benchmark integration.

It does not claim exact architecture, dataset, or training parity with the original release.

Example of how to use it
------------------------

.. code-block:: python

   import torch
   from pyhazards.models import build_model

   model = build_model(
       name="ts_satfire",
       task="segmentation",
       history=5,
       in_channels=8,
       out_channels=1,
   )

   x = torch.randn(2, 5, 8, 32, 32)
   logits = model(x)
   print(logits.shape)
