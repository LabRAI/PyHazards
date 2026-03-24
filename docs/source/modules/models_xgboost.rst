XGBoost
=======

Description
-----------

``xgboost`` is the canonical PyHazards promotion of the wildfire benchmark Track-O baseline.

This module is kept in ``pyhazards.models`` so the main branch can treat the benchmark baseline as a first-class model implementation.

It is primarily intended for benchmark integration, smoke testing, and registry-based construction through ``build_model(...)``.

Paper / source
--------------

- Promoted from the wildfire benchmark Track-O model family in PyHazards.
- Source implementation lineage: ``pyhazards.pipelines.wildfire_benchmark.models.xgboost_track_o``.

Paper parity note
-----------------

This PyHazards implementation is intentionally benchmark-facing. It preserves the modeling role of the Track-O baseline while making the model available from the main ``pyhazards.models`` layer.

Example of how to use it
------------------------

.. code-block:: python

   import torch
   from pyhazards.models import build_model

   model = build_model(
       name="xgboost",
       task="classification",
   )

   if "classification" == "classification":
       x = torch.randn(4, 16)
   else:
       x = torch.randn(2, 1, 32, 32)
   out = model(x)
   print(type(out))
