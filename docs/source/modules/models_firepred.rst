FirePred
========

Description
-----------

``firepred`` is a PyHazards port inspired by the FirePred wildfire spread model.

This implementation keeps the benchmark-relevant structure of the published method:

- multi-temporal wildfire raster input
- separate recent, aggregated, and snapshot branches
- fused CNN decoding for next-step wildfire spread prediction

Paper / source
--------------

- `FirePred GitHub repository <https://github.com/Seyed-Ali-Ahmadi/FirePred>`_
- Paper title used by the official repository: ``FirePred: A hybrid multi-temporal convolutional neural network model for wildfire spread prediction``

Paper parity note
-----------------

This PyHazards implementation is intentionally a lightweight benchmark-facing port.
It preserves the multi-temporal hybrid-CNN pattern while avoiding notebook-only or
project-specific training code from the original release.

Example of how to use it
------------------------

.. code-block:: python

   import torch
   from pyhazards.models import build_model

   model = build_model(
       name="firepred",
       task="segmentation",
       history=5,
       in_channels=8,
       out_channels=1,
   )

   x = torch.randn(2, 5, 8, 32, 32)
   logits = model(x)
   print(logits.shape)
