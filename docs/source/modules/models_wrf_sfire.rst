WRF-SFIRE Adapter
=================

Description
-----------

``wrf_sfire`` is a lightweight PyHazards raster adapter inspired by the
transport-and-diffusion behavior of the WRF-SFIRE wildfire spread system.

This module is designed as a benchmark-facing canonical model that keeps the
main spread intuition simple inside the PyHazards library:

- ``in_channels=12``
- ``out_channels=1``
- ``diffusion_steps=3`` by default
- local transport via a fixed spread kernel
- terrain and moisture modulation during repeated spread steps

Paper / source
--------------

- `Coupled atmosphere-wildland fire modeling with WRF 3.3 and SFIRE 2011 <https://doi.org/10.5194/gmd-4-591-2011>`_
- `WRF-SFIRE repository <https://github.com/openwfm/WRF-SFIRE>`_

Paper parity note
-----------------

This PyHazards implementation is intentionally **not** the full WRF-SFIRE
coupled simulator. Instead, it provides a compact raster adapter that preserves
the main local-spread intuition needed for benchmark integration and smoke
testing inside the main library.

The canonical PyHazards version keeps:

- raster input/output contract
- repeated local diffusion
- terrain-aware spread scaling
- moisture damping

It does not attempt to reproduce the full atmospheric coupling, mesh handling,
or solver stack of the original WRF-SFIRE system.

Example of how to use it
------------------------

.. code-block:: python

   import torch
   from pyhazards.models import build_model

   model = build_model(
       name="wrf_sfire",
       task="segmentation",
       in_channels=12,
       diffusion_steps=3,
   )

   x = torch.randn(2, 12, 32, 32)
   spread = model(x)
   print(spread.shape)
