ForeFire Adapter
================

Description
-----------

``forefire`` is a lightweight PyHazards raster adapter inspired by the
front-propagation behavior of the ForeFire wildfire spread simulator.

This module is designed as a benchmark-facing canonical model that keeps the
main local spread mechanism simple and reproducible inside the PyHazards
library:

- ``in_channels=12``
- ``out_channels=1``
- ``diffusion_steps=2`` by default
- repeated neighborhood spread updates
- explicit fuel and wind modulation

Paper / source
--------------

- `ForeFire: open source code for wildland fire spread models <https://doi.org/10.14195/978-989-26-0884-6_29>`_
- `ForeFire repository <https://github.com/forefireAPI/forefire>`_

Paper parity note
-----------------

This PyHazards implementation is intentionally **not** the full ForeFire
simulation system. Instead, it provides a compact raster adapter that captures
the main deterministic spread intuition needed for registry integration and
smoke testing in the main library.

The canonical PyHazards version keeps:

- raster input/output contract
- repeated local front spread updates
- fuel-conditioned spread
- wind-conditioned spread

It does not attempt to reproduce the full propagation solver, landscape
representation, or operational simulation stack of the original ForeFire
system.

Example of how to use it
------------------------

.. code-block:: python

   import torch
   from pyhazards.models import build_model

   model = build_model(
       name="forefire",
       task="segmentation",
       in_channels=12,
       diffusion_steps=2,
   )

   x = torch.randn(2, 12, 32, 32)
   spread = model(x)
   print(spread.shape)
