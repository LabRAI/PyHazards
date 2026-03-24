Qwen2.5-VL Wildfire Prompted
============================

Description
-----------

``qwen25_vl_wildfire_prompted`` is a benchmark-facing prompt-conditioned VLM port
inspired by Qwen2.5-VL.

This implementation keeps the integration-relevant structure for a generic wildfire
vision-language baseline:

- raster wildfire/environment input
- prompt-token conditioning
- visual-token and prompt-token fusion
- dense wildfire-risk decoding

Paper / source
--------------

- `QwenLM/Qwen2.5-VL GitHub repository <https://github.com/QwenLM/Qwen2.5-VL>`_
- `Qwen2.5-VL Technical Report <https://arxiv.org/abs/2502.13923>`_

Paper parity note
-----------------

This PyHazards implementation is intentionally not a full parameter-port of the
released Qwen2.5-VL checkpoints. Instead, it is a compact prompt-conditioned
wildfire segmentation baseline that preserves the benchmark-relevant VLM pattern:

- prompt-conditioned visual reasoning
- image-token and prompt-token fusion
- dense downstream wildfire prediction head

Example of how to use it
------------------------

.. code-block:: python

   import torch
   from pyhazards.models import build_model

   model = build_model(
       name="qwen25_vl_wildfire_prompted",
       task="segmentation",
       in_channels=6,
       out_dim=1,
   )

   x = torch.randn(2, 6, 32, 32)
   logits = model(x)
   print(logits.shape)
