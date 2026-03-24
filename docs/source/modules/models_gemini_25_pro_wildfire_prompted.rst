Gemini 2.5 Pro Wildfire Prompted
================================

Description
-----------

``gemini_25_pro_wildfire_prompted`` is a benchmark-facing prompt-conditioned VLM port
inspired by Gemini 2.5 Pro.

This implementation keeps the integration-relevant structure for a generic wildfire
vision-language baseline:

- raster wildfire/environment input
- prompt-token conditioning
- visual-token and prompt-token fusion
- dense wildfire-risk decoding

Paper / source
--------------

- `Gemini models documentation <https://ai.google.dev/gemini-api/docs/models/gemini-v2>`_

Paper parity note
-----------------

This PyHazards implementation is intentionally not a checkpoint-level port of
Gemini 2.5 Pro. Instead, it is a compact prompt-conditioned wildfire segmentation
baseline that preserves the benchmark-relevant VLM pattern.

Example of how to use it
------------------------

.. code-block:: python

   import torch
   from pyhazards.models import build_model

   model = build_model(
       name="gemini_25_pro_wildfire_prompted",
       task="segmentation",
       in_channels=6,
       out_dim=1,
   )

   x = torch.randn(2, 6, 32, 32)
   logits = model(x)
   print(logits.shape)
