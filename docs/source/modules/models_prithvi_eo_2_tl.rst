Prithvi-EO-2.0-TL
=================

Description
-----------

``prithvi_eo_2_tl`` is a lightweight PyHazards port inspired by the
Prithvi-EO-2.0 transfer-learning model family.

This module keeps the main ideas highlighted in the official paper/model card:

- multi-temporal EO input sequences
- temporal embeddings
- location embeddings
- transformer-style EO backbone
- segmentation-ready downstream head

Paper / source
--------------

- `Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications <https://huggingface.co/papers/2412.02732>`_
- `Prithvi-EO-2.0-300M-TL model card <https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL>`_

Paper parity note
-----------------

This PyHazards implementation is intentionally **not** the full official
pretrained foundation model with released checkpoints. Instead, it is a clean
PyTorch port that preserves the benchmark-relevant architectural ideas needed
for PyHazards integration:

- sequence-based EO input handling
- temporal and location conditioning
- transformer encoder over patch tokens
- downstream segmentation decoding

It does not claim checkpoint parity with the official IBM-NASA release.

Example of how to use it
------------------------

.. code-block:: python

   import torch
   from pyhazards.models import build_model

   model = build_model(
       name="prithvi_eo_2_tl",
       task="segmentation",
       image_size=32,
       in_channels=6,
       out_dim=1,
   )

   x = torch.randn(2, 4, 6, 32, 32)
   logits = model(x)
   print(logits.shape)
