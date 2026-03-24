Prithvi BurnScars
=================

Description
-----------

``prithvi_burnscars`` is a lightweight PyHazards downstream segmentation model
inspired by the official Prithvi BurnScars release.

This module keeps the benchmark-relevant ideas from the model card:

- Prithvi-style EO temporal backbone
- single-timestamp or arbitrary-timestamp fine-tuning support
- burn-scar-style segmentation head
- U-Net-like skip fusion for dense output

Paper / source
--------------

- `Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications <https://huggingface.co/papers/2412.02732>`_
- `Prithvi-EO-2.0-300M-BurnScars model card <https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars>`_

Paper parity note
-----------------

This PyHazards implementation is intentionally **not** the official released
checkpoint. Instead, it is a benchmark-facing downstream port that preserves the
main architectural story of the official BurnScars release:

- EO foundation-style encoder
- downstream burn-scar segmentation objective
- dense decoder with skip fusion

It is suitable for PyHazards integration and smoke testing, while remaining
transparent about not being a weight-identical reproduction.

Example of how to use it
------------------------

.. code-block:: python

   import torch
   from pyhazards.models import build_model

   model = build_model(
       name="prithvi_burnscars",
       task="segmentation",
       image_size=32,
       in_channels=6,
       out_dim=1,
   )

   x = torch.randn(2, 1, 6, 32, 32)
   logits = model(x)
   print(logits.shape)
