:orphan:

FPA-FOD Internal LSTM
=====================

Description
-----------

``wildfire_fpa_lstm`` is the standalone LSTM component used beneath the public ``wildfire_fpa``
framework. It consumes a rolling lookback window and predicts next-week wildfire counts.

Example of how to use it
------------------------

.. code-block:: python

   import torch
   from pyhazards.models.wildfire_fpa_lstm import WildfireFPALSTM

   model = WildfireFPALSTM(
       input_dim=7,
       output_dim=5,
       lookback=12,
   )

   x = torch.randn(2, 12, 7)
   preds = model(x)
   print(preds.shape)  # (2, 5)
