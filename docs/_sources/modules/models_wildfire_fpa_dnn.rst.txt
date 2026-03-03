:orphan:

FPA-FOD Internal DNN
====================

Description
-----------

``wildfire_fpa_dnn`` is the internal DNN stage used beneath the public ``wildfire_fpa`` framework
entrypoint. In PyHazards it is implemented as a configurable multilayer perceptron for incident-level
cause or size-class prediction.

This corresponds to the classification stage described in:
`Developing risk assessment framework for wildfire in the United States - A deep learning approach to safety and sustainability <https://www.sciencedirect.com/science/article/pii/S2949926723000033>`_.

Example of how to use it
------------------------

.. code-block:: python

   import torch
   from pyhazards.models.wildfire_fpa_dnn import WildfireFPADNN

   model = WildfireFPADNN(
       in_dim=8,
       out_dim=5,
       hidden_dim=64,
       depth=2,
   )

   x = torch.randn(4, 8)
   logits = model(x)
   print(logits.shape)  # (4, 5)

Notes
-----

- The public paper-facing registry name is ``wildfire_fpa`` with ``task="classification"``.
