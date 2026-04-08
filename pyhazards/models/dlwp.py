def dlwp_builder(
    task: str,
    in_channels: int,
    num_faces: int,
    face_size: int,
    **kwargs,
) -> nn.Module:
    """
    Builder function for DLWP (Deep Learning Weather Prediction) model.
    
    Args:
        task: Task type (must be 'regression')
        in_channels: Number of input atmospheric variables (e.g., 7)
        num_faces: Number of cubed-sphere faces (typically 6)
        face_size: Resolution per face (e.g., 64 for 64×64 grid)
        **kwargs: Additional hyperparameters:
            - out_channels: Number of output variables (default: same as in_channels)
            - num_levels: U-Net depth (default: 4)
            - base_channels: Starting channel count (default: 64)
            - kernel_size: Conv2d kernel size (default: 3)
            - activation: Activation function 'relu' or 'leaky_relu' (default: 'relu')
            - dropout: Dropout rate (default: 0.1)
    
    Returns:
        DLWP model instance
    
    Raises:
        ValueError: If task is not 'regression'
    
    Example:
        >>> model = build_model(
        ...     name="dlwp",
        ...     task="regression",
        ...     in_channels=7,
        ...     num_faces=6,
        ...     face_size=64,
        ... )
    """
    # Validate task
    if task.lower() != "regression":
        raise ValueError(
            f"DLWP only supports regression tasks for weather prediction, "
            f"got task='{task}'"
        )
    
    # Extract kwargs with defaults
    out_channels = kwargs.get("out_channels", in_channels)
    num_levels = kwargs.get("num_levels", 4)
    base_channels = kwargs.get("base_channels", 64)
    kernel_size = kwargs.get("kernel_size", 3)
    activation = kwargs.get("activation", "relu")
    dropout = kwargs.get("dropout", 0.1)
    
    # Build and return model
    return DLWP(
        in_channels=in_channels,
        out_channels=out_channels,
        num_faces=num_faces,
        face_size=face_size,
        num_levels=num_levels,
        base_channels=base_channels,
        kernel_size=kernel_size,
        activation=activation,
        dropout=dropout,
    )