InternVL3 Wildfire Prompted
===========================

Description
-----------

``internvl3_wildfire_prompted`` is a benchmark-facing prompt-conditioned VLM port
inspired by InternVL3.

This implementation keeps the integration-relevant structure for a generic wildfire
vision-language baseline:

- raster wildfire/environment input
- prompt-token conditioning
- visual-token and prompt-token fusion
- dense wildfire-risk decoding

Paper / source
--------------

- `InternVL repository <https://github.com/OpenGVLab/InternVL>`_

Paper parity note
-----------------

This PyHazards implementation is intentionally not a checkpoint-level port of
InternVL3. Instead, it is a compact prompt-conditioned wildfire segmentation
baseline that preserves the benchmark-relevant VLM pattern.

Example of how to use it
------------------------

.. code-block:: python

   import torch
   from pyhazards.models import build_model

   model = build_model(
       name="internvl3_wildfire_prompted",
       task="segmentation",
       in_channels=6,
       out_dim=1,
   )

   x = torch.randn(2, 6, 32, 32)
   logits = model(x)
   print(logits.shape)
