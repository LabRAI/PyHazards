class DLWP(nn.Module):
    """
    Deep Learning Weather Prediction (DLWP) model using U-Net on cubed-sphere grid.
    
    Based on the papers:
    - "Sub-Seasonal Forecasting With a Large Ensemble of Deep-Learning Weather 
       Prediction Models" (Weyn et al., 2021)
    - "Improving Data-Driven Global Weather Prediction Using Deep Convolutional 
       Neural Networks on a Cubed Sphere" (Weyn et al., 2020)
    
    The DLWP model predicts future atmospheric state u(t+Δt) from current state u(t)
    where Δt = 6 hours. It uses a U-Net architecture adapted for cubed-sphere grids
    to capture multi-scale atmospheric processes.
    
    Args:
        in_channels: Number of input atmospheric variables (e.g., 7)
        out_channels: Number of output variables (default: same as input)
        num_faces: Number of cubed-sphere faces (typically 6)
        face_size: Grid resolution per face (e.g., 64 for 64×64)
        num_levels: U-Net encoder/decoder depth (default: 4)
        base_channels: Starting channel count, doubles each level (default: 64)
        kernel_size: Conv2d kernel size (default: 3)
        activation: Activation function - 'relu' or 'leaky_relu' (default: 'relu')
        dropout: Dropout rate (default: 0.1)
    
    Input:
        x: Atmospheric state tensor (B, C, F, H, W)
           - B: batch size
           - C: channels (atmospheric variables)
           - F: faces (6 for cubed-sphere)
           - H, W: spatial dimensions per face (e.g., 64×64)
    
    Output:
        y: Predicted future state (B, C_out, F, H, W)
           - Same spatial dimensions as input
           - C_out channels (usually same as C)
    
    Example:
        >>> model = DLWP(
        ...     in_channels=7,
        ...     num_faces=6,
        ...     face_size=64,
        ... )
        >>> x = torch.randn(4, 7, 6, 64, 64)  # Current atmospheric state
        >>> y = model(x)                       # Predicted state 6hr later
        >>> y.shape
        torch.Size([4, 7, 6, 64, 64])
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for weather prediction.
        
        Args:
            x: Current atmospheric state (B, C, F, H, W)
        
        Returns:
            Predicted future atmospheric state (B, C_out, F, H, W)
        
        Raises:
            ValueError: If input shape doesn't match expected dimensions
        """
        # Validate input shape
        if x.ndim != 5:
            raise ValueError(
                f"Expected 5D input (B, C, F, H, W), got {x.ndim}D tensor "
                f"with shape {tuple(x.shape)}"
            )
        
        B, C, F, H, W = x.shape
        
        if C != self.in_channels:
            raise ValueError(
                f"Expected in_channels={self.in_channels}, got {C}"
            )
        if F != self.num_faces:
            raise ValueError(
                f"Expected num_faces={self.num_faces}, got {F}"
            )
        if H != self.face_size or W != self.face_size:
            raise ValueError(
                f"Expected face_size={self.face_size}×{self.face_size}, "
                f"got {H}×{W}"
            )
        
        # [Forward implementation here]
        # Returns: (B, out_channels, F, H, W)

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