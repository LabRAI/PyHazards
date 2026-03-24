VIIRS 375 m Active Fire
=======================

Description
-----------

``viirs_375m_active_fire`` is a PyHazards operational-detection baseline inspired by
NASA's VIIRS 375 m active-fire algorithm and its FIRMS-facing use in practice.

This implementation keeps the benchmark-relevant structure of the published method:

- satellite active-fire detection framing rather than generic segmentation
- contextual thermal anomaly estimation
- split-window style evidence between mid-IR and longwave channels
- lightweight learnable calibration head so the method can run under the PyHazards benchmark contract

Paper / source
--------------

- `NASA Earthdata VIIRS I-Band 375 m Active Fire page <https://www.earthdata.nasa.gov/data/instruments/viirs/viirs-i-band-375-m-active-fire-data>`_
- `Schroeder et al. (2014) <https://doi.org/10.1016/j.rse.2013.12.008>`_

Paper parity note
-----------------

This PyHazards implementation is intentionally a benchmark-facing surrogate rather than a byte-for-byte
reproduction of the NASA operational code path. It preserves the operational-detection intuition of
contextual thermal anomaly plus spectral evidence, while adding a compact learnable calibration head so
that smoke runs can generate standard training artifacts.

Example of how to use it
------------------------

.. code-block:: python

   import torch
   from pyhazards.models import build_model

   model = build_model(
       name="viirs_375m_active_fire",
       task="segmentation",
       in_channels=5,
       out_dim=1,
   )

   x = torch.randn(2, 5, 32, 32)
   logits = model(x)
   print(logits.shape)
