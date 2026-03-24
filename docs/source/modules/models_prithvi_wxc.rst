Prithvi-WxC
===========

Description
-----------

``prithvi_wxc`` is a lightweight PyHazards port inspired by the
Prithvi-WxC weather-climate foundation-model family.

This module keeps the main ideas highlighted in the official paper/model card:

- multi-step weather input sequences
- lead-time conditioning
- variable-summary conditioning
- transformer-style weather backbone
- dense downstream head for wildfire-style grid prediction

Paper / source
--------------

- `Prithvi WxC: Foundation Model for Weather and Climate <https://huggingface.co/papers/2409.13598>`_
- `Prithvi-WxC model card <https://huggingface.co/Prithvi-WxC/prithvi.wxc.2300m.v1>`_

Paper parity note
-----------------

This PyHazards implementation is intentionally **not** the full official
pretrained Prithvi-WxC checkpoint stack. Instead, it is a clean PyTorch port
that preserves the benchmark-relevant ideas we need for integration:

- multi-variable weather sequence handling
- lead-time-aware conditioning
- transformer encoder over weather patch tokens
- dense wildfire-risk decoding

It does not claim checkpoint parity with the official NASA/IBM release.

Example of how to use it
------------------------

.. code-block:: python

   import torch
   from pyhazards.models import build_model

   model = build_model(
       name="prithvi_wxc",
       task="segmentation",
       image_size=32,
       in_channels=8,
       out_dim=1,
   )

   x = torch.randn(2, 5, 8, 32, 32)
   lead_time = torch.linspace(6.0, 30.0, 5).repeat(2, 1)
   logits = model({"x": x, "lead_time_hours": lead_time})
   print(logits.shape)
