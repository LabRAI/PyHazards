ASUFM
=====

Description
-----------

``asufm`` is a self-contained PyHazards port of the ASUFM wildfire model family:
an Attention Swin U-Net with focal modulation for wildfire spread prediction.

This module follows the official ASUFM configuration pattern with:

- ``image_size=64``
- ``patch_size=4``
- ``in_channels=6``
- ``embed_dim=96``
- ``depths=(2, 2, 2, 2)``
- ``num_heads=(3, 6, 12, 24)``
- focal modulation in the encoder
- attention-gated skip connections in the decoder

Paper / source
--------------

- `Wildfire Spread Prediction in North America Using Satellite Imagery and Vision Transformer <https://doi.org/10.1109/CAI59869.2024.00278>`_
- `Official repository <https://github.com/bronteee/fire-asufm>`_

Paper parity note
-----------------

This PyHazards implementation preserves the main architectural ideas from the
official repository while staying dependency-free inside the main library.
It intentionally replaces the original ``timm``/``einops``-based components
with a native PyTorch implementation of:

- patch embedding
- hierarchical Swin-style window attention
- focal modulation in encoder blocks
- U-Net-style decoder with spatially gated skip connections

Example of how to use it
------------------------

.. code-block:: python

   import torch
   from pyhazards.models import build_model

   model = build_model(
       name="asufm",
       task="segmentation",
       image_size=64,
       in_channels=6,
       out_dim=1,
   )

   x = torch.randn(2, 6, 64, 64)
   logits = model(x)
   print(logits.shape)
