FireMM-IR
=========

Description
-----------

``firemm_ir`` is a benchmark-facing PyHazards port inspired by the FireMM-IR
multi-modal large language model for remote-sensing forest fire monitoring.

This module preserves the main ideas emphasized by the paper:

- dual-modality optical + infrared fusion
- class-aware memory
- instruction-conditioned segmentation reasoning
- dense wildfire-scene decoding

Paper / source
--------------

- `FireMM-IR: An Infrared-Enhanced Multi-Modal Large Language Model for Comprehensive Scene Understanding in Remote Sensing Forest Fire Monitoring <https://www.mdpi.com/1424-8220/26/2/390>`_
- `PubMed entry <https://pubmed.ncbi.nlm.nih.gov/41600187/>`_

Paper parity note
-----------------

This PyHazards implementation is intentionally **not** the original full MLLM
stack with text generation, external instruction tuning, and dataset-specific
serving pipeline. Instead, it is a benchmark-friendly neural port that
preserves the architectural roles needed for PyHazards integration:

- optical / infrared dual encoder
- class-aware memory enhancement
- instruction-conditioned feature fusion
- dense segmentation head

It is suitable for smoke testing and benchmark integration, while remaining
transparent about not reproducing the original external MLLM runtime.

Example of how to use it
------------------------

.. code-block:: python

   import torch
   from pyhazards.models import build_model

   model = build_model(
       name="firemm_ir",
       task="segmentation",
       in_channels=6,
       out_dim=1,
   )

   x = torch.randn(2, 6, 32, 32)
   logits = model(x)
   print(logits.shape)
