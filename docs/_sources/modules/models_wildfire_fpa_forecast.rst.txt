wildfire_fpa_forecast
=====================

Description
-----------

``wildfire_fpa_forecast`` is the forecasting component of the FPA-FOD wildfire framework. It combines:

- an LSTM temporal encoder over weekly sequences, and
- an autoencoder-derived latent summary of the same sequence,

before predicting next-week wildfire counts.

This is PyHazards' modular implementation of the forecasting stage described in:
`Prediction of the cause and size of wildfire using artificial intelligence <https://www.sciencedirect.com/science/article/pii/S2949926723000033>`_.

Example of how to use it
------------------------

.. code-block:: python

   import torch
   from pyhazards.models import build_model

   model = build_model(
       name="wildfire_fpa_forecast",
       task="forecasting",
       input_dim=7,
       output_dim=5,
       lookback=12,
       latent_dim=16,
   )

   x = torch.randn(2, 12, 7)
   preds = model(x)
   print(preds.shape)  # (2, 5)

Notes
-----

- ``forward_with_reconstruction(...)`` returns both the forecast and the sequence reconstruction.
- PyHazards also keeps standalone temporal and reconstruction components available for lower-level
  experimentation, but the paper-facing entrypoint is ``wildfire_fpa_forecast``.
