DNN-LSTM-AutoEncoder
====================

Description
-----------

``wildfire_fpa`` is the paper-facing PyHazards entrypoint for the two-stage wildfire risk assessment
framework described in:
`Developing risk assessment framework for wildfire in the United States - A deep learning approach to safety and sustainability <https://www.sciencedirect.com/science/article/pii/S2949926723000033>`_.

The framework comprises two sequential model stages:

- a DNN stage for wildfire cause and size prediction from incident-level tabular features, and
- an LSTM + autoencoder stage for imminent wildfire forecasting from weekly sequences.

PyHazards keeps those stages modular internally, but exposes them under one public model name.

Modular
-------

``wildfire_fpa``

Example of how to use it
------------------------

Classification stage
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from pyhazards.models import build_model

   model = build_model(
       name="wildfire_fpa",
       task="classification",
       in_dim=8,
       out_dim=5,
       hidden_dim=64,
       depth=2,
   )

   x = torch.randn(4, 8)
   logits = model(x)
   print(logits.shape)  # (4, 5)

Forecasting stage
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from pyhazards.models import build_model

   model = build_model(
       name="wildfire_fpa",
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

- Use ``task="classification"`` to build the DNN cause/size stage.
- Use ``task="forecasting"`` or ``task="regression"`` to build the LSTM + autoencoder stage.
- Lower-level internal modules remain available in ``pyhazards.models.wildfire_fpa_dnn``,
  ``pyhazards.models.wildfire_fpa_forecast``, ``pyhazards.models.wildfire_fpa_lstm``, and
  ``pyhazards.models.wildfire_fpa_autoencoder``.
