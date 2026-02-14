from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.init as init

def _normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """
    Row-normalize an adjacency matrix with self-loops.
    
    Args:
        adj: Adjacency matrix (N, N) or (B, N, N)
    
    Returns:
        Normalized adjacency with self-loops
    """
    if adj.dim() == 2:
        adj = adj.unsqueeze(0)
    
    # Add self-loops
    eye = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype)
    adj = adj.float() + eye.unsqueeze(0)
    
    # Row normalization: D^-1 * A
    return adj / adj.sum(-1, keepdim=True).clamp(min=1e-6)

class GraphConvLEMCell(nn.Module):
    """
    Graph-adapted ConvLEM cell for county-level temporal predictions.
    
    Args:
        in_channels: Input feature dimension
        out_channels: Hidden/memory state dimension
        num_counties: Number of graph nodes (counties)
        dt: Time step parameter for memory integration (default: 1.0)
        activation: Activation function - 'tanh' or 'relu' (default: 'tanh')
        use_reset_gate: Whether to use reset gate variant (default: False)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_counties: int,
        dt: float = 1.0,
        activation: str = 'tanh',
        use_reset_gate: bool = False,
    ):
        super().__init__()
        
        # Activation function
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}. Use 'tanh' or 'relu'.")
        
        self.dt = dt
        self.use_reset_gate = use_reset_gate
        self.num_counties = num_counties
        self.out_channels = out_channels
        
        # Input transformations
        if use_reset_gate:
            self.transform_x = nn.Linear(in_channels, 5 * out_channels)
            self.transform_y = nn.Linear(out_channels, 4 * out_channels)
        else:
            self.transform_x = nn.Linear(in_channels, 4 * out_channels)
            self.transform_y = nn.Linear(out_channels, 3 * out_channels)
        
        # Memory transformation
        self.transform_z = nn.Linear(out_channels, out_channels)
        
        # Learnable Hadamard product weights
        self.W_z1 = nn.Parameter(torch.Tensor(out_channels, num_counties))
        self.W_z2 = nn.Parameter(torch.Tensor(out_channels, num_counties))
        if use_reset_gate:
            self.W_z4 = nn.Parameter(torch.Tensor(out_channels, num_counties))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        for param in self.parameters():
            if len(param.shape) > 1:
                init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features (B, N, in_channels)
            y: Hidden state (B, N, out_channels)
            z: Memory state (B, N, out_channels)
            adj: Optional adjacency matrix (B, N, N) or (N, N)
        
        Returns:
            Tuple of (y_new, z_new)
        """
        B, N, _ = x.shape
        
        # Apply graph convolution if adjacency provided
        if adj is not None:
            if adj.dim() == 2:
                adj = adj.unsqueeze(0).expand(B, -1, -1)
            x = torch.matmul(adj, x)
            y = torch.matmul(adj, y)
            z = torch.matmul(adj, z)
        
        # Transform inputs
        transformed_x = self.transform_x(x)
        transformed_y = self.transform_y(y)
        
        if self.use_reset_gate:
            # With reset gate variant
            i_dt1, i_dt2, g_dx2, i_z, i_y = torch.chunk(transformed_x, chunks=5, dim=-1)
            h_dt1, h_dt2, h_y, g_dy2 = torch.chunk(transformed_y, chunks=4, dim=-1)
            
            ms_dt = self.dt * torch.sigmoid(i_dt2 + h_dt2 + self.W_z2.t().unsqueeze(0) * z)
            z = (1.0 - ms_dt) * z + ms_dt * self.activation(i_y + h_y)
            
            gate2 = self.dt * torch.sigmoid(g_dx2 + g_dy2 + self.W_z4.t().unsqueeze(0) * z)
            transformed_z = gate2 * self.transform_z(z)
            
            ms_dt_bar = self.dt * torch.sigmoid(i_dt1 + h_dt1 + self.W_z1.t().unsqueeze(0) * z)
            y = (1.0 - ms_dt_bar) * y + ms_dt_bar * self.activation(transformed_z + i_z)
        else:
            # Without reset gate
            i_dt1, i_dt2, i_z, i_y = torch.chunk(transformed_x, chunks=4, dim=-1)
            h_dt1, h_dt2, h_y = torch.chunk(transformed_y, chunks=3, dim=-1)
            
            ms_dt = self.dt * torch.sigmoid(i_dt2 + h_dt2 + self.W_z2.t().unsqueeze(0) * z)
            z = (1.0 - ms_dt) * z + ms_dt * self.activation(i_y + h_y)
            
            transformed_z = self.transform_z(z)
            
            ms_dt_bar = self.dt * torch.sigmoid(i_dt1 + h_dt1 + self.W_z1.t().unsqueeze(0) * z)
            y = (1.0 - ms_dt_bar) * y + ms_dt_bar * self.activation(transformed_z + i_z)
        
        return y, z

class ConvLEMWildfire(nn.Module):

    """
    ConvLEM-based wildfire prediction model (minimal version).
    """
    
    def __init__(
        self,
        in_dim: int,
        num_counties: int,
        past_days: int,
        hidden_dim: int = 144,
        num_layers: int = 2,
        dt: float = 1.0,
        activation: str = 'tanh',
        use_reset_gate: bool = False,
        dropout: float = 0.1,
        adjacency: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.num_counties = num_counties
        self.past_days = past_days
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Placeholder: just a simple linear layer for now
        self.placeholder = nn.Linear(in_dim, 1)
    
    def forward(self, x: torch.Tensor, adjacency: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Minimal forward pass.
        
        Args:
            x: (batch, past_days, num_counties, in_dim)
            adjacency: Optional adjacency matrix
            
        Returns:
            logits: (batch, num_counties)
        """
        B, T, N, F = x.shape
        
        # Validate shapes
        if T != self.past_days:
            raise ValueError(f"Expected past_days={self.past_days}, got {T}")
        if N != self.num_counties:
            raise ValueError(f"Expected num_counties={self.num_counties}, got {N}")
        if F != self.in_dim:
            raise ValueError(f"Expected in_dim={self.in_dim}, got {F}")
        
        # Placeholder forward: just use last timestep
        last_step = x[:, -1, :, :]  # (B, N, F)
        logits = self.placeholder(last_step).squeeze(-1)  # (B, N)
        
        return logits

		
"""
Builder function for ConvLEM wildfire prediction model.
This builder follows PyHazards conventions:
- Takes task + required params + **kwargs
- Validates task type
- Returns instantiated model with merged defaults

    Args:
        task: Task type (must be 'classification')
        in_dim: Number of input features per county-day
        num_counties: Number of counties (graph nodes)
        past_days: Number of historical days in input sequence
        **kwargs: Additional hyperparameters:
            - hidden_dim: Hidden state dimension (default: 144)
            - num_layers: Number of encoder/decoder layer pairs (default: 2)
            - dt: Time step parameter for LEM mechanism (default: 1.0)
            - activation: Activation function 'tanh' or 'relu' (default: 'tanh')
            - use_reset_gate: Use reset gate variant (default: False)
            - dropout: Dropout rate (default: 0.1)
            - adjacency: Optional adjacency matrix (N, N) (default: None)
    
    Returns:
        Instantiated ConvLEMWildfire model
    
    Raises:
        ValueError: If task is not 'classification'
    """
		    
def convlem_wildfire_builder(
    task: str,
    in_dim: int,
    num_counties: int,
    past_days: int,
    **kwargs,
) -> ConvLEMWildfire:
    """Builder function for ConvLEM wildfire model."""
    
    if task.lower() not in {"classification", "binary_classification"}:
        raise ValueError(
            f"ConvLEM wildfire model is classification-only, got task='{task}'"
        )
    
    return ConvLEMWildfire(
        in_dim=in_dim,
        num_counties=num_counties,
        past_days=past_days,
        hidden_dim=kwargs.get("hidden_dim", 144),
        num_layers=kwargs.get("num_layers", 2),
        dt=kwargs.get("dt", 1.0),
        activation=kwargs.get("activation", "tanh"),
        use_reset_gate=kwargs.get("use_reset_gate", False),
        dropout=kwargs.get("dropout", 0.1),
        adjacency=kwargs.get("adjacency"),
    )