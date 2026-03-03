:orphan:

FPA-FOD Internal LSTM
=====================

Description
-----------

``wildfire_fpa_lstm`` is the standalone LSTM component used beneath the paper-facing forecast model
family. It consumes a rolling lookback window and predicts next-week wildfire counts.

Example of how to use it
------------------------

.. code-block:: python

   import torch
   from pyhazards.models import build_model

   model = build_model(
       name="wildfire_fpa_lstm",
       task="forecasting",
       input_dim=7,
       output_dim=5,
       lookback=12,
   )

   x = torch.randn(2, 12, 7)
   preds = model(x)
   print(preds.shape)  # (2, 5)
