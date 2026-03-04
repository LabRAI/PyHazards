WaveCastNet
===========

WaveCastNet is a deep learning model for earthquake wavefield forecasting using 
sequence-to-sequence learning with Convolutional Long Expressive Memory (ConvLEM) cells.

Overview
--------

WaveCastNet predicts the future evolution of seismic wavefields for earthquake early 
warning systems. It uses a sequence-to-sequence architecture with ConvLEM cells to:

- Process past wavefield observations (e.g., 60 timesteps of 3-component particle velocity)
- Compress temporal sequences into a latent representation
- Generate future wavefield predictions (e.g., next 60 timesteps)

The model operates on dense spatial grids and is designed for real-time forecasting 
of ground motions without requiring explicit magnitude or epicenter estimation.

Example of how to use it
------------------------

.. code-block:: python

    from pyhazards.models import build_model
    import torch

    # Build model
    model = build_model(
        name="wavecastnet",
        task="regression",
        in_channels=3,          # X, Y, Z velocity components
        height=344,             # Spatial grid height
        width=224,              # Spatial grid width
        temporal_in=60,         # Input sequence length (timesteps)
        temporal_out=60,        # Output sequence length (timesteps)
    )

    # Forward pass
    x = torch.randn(4, 3, 60, 344, 224)  # Past wavefields
    y = model(x)                          # Future wavefields
    print(y.shape)  # (4, 3, 60, 344, 224)