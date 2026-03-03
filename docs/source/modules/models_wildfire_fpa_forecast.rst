:orphan:

FPA-FOD Internal Forecast
=========================

Description
-----------

``wildfire_fpa_forecast`` is the internal forecasting stage used beneath the public ``wildfire_fpa``
framework entrypoint. It combines:

- an LSTM temporal encoder over weekly sequences, and
- an autoencoder-derived latent summary of the same sequence,

before predicting next-week wildfire counts.

This is PyHazards' modular implementation of the forecasting stage described in:
`Developing risk assessment framework for wildfire in the United States - A deep learning approach to safety and sustainability <https://www.sciencedirect.com/science/article/pii/S2949926723000033>`_.

Example of how to use it
------------------------

.. code-block:: python

   import torch
   from pyhazards.models.wildfire_fpa_forecast import WildfireFPAForecast

   model = WildfireFPAForecast(
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
  experimentation, but the paper-facing entrypoint is ``wildfire_fpa`` with
  ``task="forecasting"``.
