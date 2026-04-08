DLWP
========

Description
-----------

``dlwp`` (Deep Learning Weather Prediction) is a U-Net architecture for 
global weather forecasting on cubed-sphere grids. It predicts future atmospheric 
states from current observations using a data-driven approach.

Modular
-------

``dlwp``

Example of how to use it
------------------------

.. code-block:: python
    from pyhazards.models import build_model
    import torch
 
    # Build model
    model = build_model(
        name="dlwp",
        task="regression",
        in_channels=7,      # 7 atmospheric variables
        num_faces=6,        # Cubed-sphere grid
        face_size=64,       # 64×64 per face
    )
 
    # Single-step forecast
    current_state = torch.randn(1, 7, 6, 64, 64)  # Current atmospheric state
    future_state = model(current_state)            # State 6 hours later
    print(future_state.shape)  # torch.Size([1, 7, 6, 64, 64])
