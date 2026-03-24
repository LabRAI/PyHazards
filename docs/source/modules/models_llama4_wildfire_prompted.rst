Llama 4 Wildfire Prompted
=========================

Description
-----------

``llama4_wildfire_prompted`` is a benchmark-facing prompt-conditioned multimodal port
inspired by Meta Llama 4.

This implementation keeps the integration-relevant structure for a generic wildfire
vision-language baseline:

- raster wildfire/environment input
- prompt-token conditioning
- visual-token and prompt-token fusion
- dense wildfire-risk decoding

Paper / source
--------------

- `Meta Llama organization <https://github.com/meta-llama>`_
- `Llama site <https://llama.meta.com/>`_

Paper parity note
-----------------

This PyHazards implementation is intentionally not a checkpoint-level port of
Llama 4. Instead, it is a compact prompt-conditioned wildfire segmentation
baseline that preserves the benchmark-relevant multimodal reasoning pattern.

Example of how to use it
------------------------

.. code-block:: python

   import torch
   from pyhazards.models import build_model

   model = build_model(
       name="llama4_wildfire_prompted",
       task="segmentation",
       in_channels=6,
       out_dim=1,
   )

   x = torch.randn(2, 6, 32, 32)
   logits = model(x)
   print(logits.shape)
