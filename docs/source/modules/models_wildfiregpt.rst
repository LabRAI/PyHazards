WildfireGPT
===========

Description
-----------

``wildfiregpt`` is a benchmark-facing PyHazards port inspired by the
WildfireGPT multi-agent retrieval-augmented generation system.

This module preserves the main ideas emphasized by the official paper/repository:

- user profile conditioning
- planning / analyst style system-role tokens
- retrieved knowledge conditioning
- decision-support style fusion before producing a wildfire risk map

Paper / source
--------------

- `MARSHA: multi-agent RAG system for hazard adaptation <https://www.nature.com/articles/s44168-025-00254-1>`_
- `WildfireGPT repository <https://github.com/project-araia/WildfireGPT>`_

Paper parity note
-----------------

This PyHazards implementation is intentionally **not** the original Streamlit +
OpenAI Assistant API system. Instead, it is a benchmark-friendly neural port
that preserves the architectural roles needed for PyHazards integration:

- user-profile representation
- retrieved-context representation
- multi-agent style orchestration tokens
- downstream wildfire risk decoding

It is suitable for smoke testing and benchmark integration, while remaining
transparent about not reproducing the external hosted LLM stack.

Example of how to use it
------------------------

.. code-block:: python

   import torch
   from pyhazards.models import build_model

   model = build_model(
       name="wildfiregpt",
       task="segmentation",
       in_channels=12,
       out_dim=1,
   )

   x = torch.randn(2, 12, 32, 32)
   logits = model(x)
   print(logits.shape)
