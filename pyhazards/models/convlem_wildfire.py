from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


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
        - in_channels: Input feature dimension
        - out_channels: Hidden/memory state dimension
        - num_counties: Number of graph nodes (counties)
        - dt: Time step parameter for memory integration (default: 1.0)
        - activation: Activation function - 'tanh' or 'relu' (default: 'tanh')
        - use_reset_gate: Whether to use reset gate (ConvLEMCell_1 variant) (default: False)
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
            - x: Input features (B, N, in_channels)
            - y: Hidden state (B, N, out_channels)
            - z: Memory state (B, N, out_channels)
            - adj: Optional adjacency matrix (B, N, N) or (N, N)
        
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
    ConvLEM-based wildfire prediction model for county-level data.
    
    input embedding -> encoder -> decoder -> output projection
    
    Args:
        - in_dim: Number of input features per county per day
        - num_counties: Number of counties (graph nodes)
        - past_days: Number of historical days to process
        - hidden_dim: Hidden state dimension (default: 144)
        - num_layers: Number of ConvLEM encoder/decoder layer pairs (default: 2)
        - dt: Time step parameter for LEM mechanism (default: 1.0)
        - activation: Activation function - 'tanh' or 'relu' (default: 'tanh')
        - use_reset_gate: Whether to use reset gate variant (default: False)
        - dropout: Dropout rate (default: 0.1)
        - adjacency: Optional fixed adjacency matrix (N, N)
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
        
        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            GraphConvLEMCell(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                num_counties=num_counties,
                dt=dt,
                activation=activation,
                use_reset_gate=use_reset_gate,
            )
            for _ in range(num_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            GraphConvLEMCell(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                num_counties=num_counties,
                dt=dt,
                activation=activation,
                use_reset_gate=use_reset_gate,
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Register adjacency matrix as buffer
        self.register_buffer("_adjacency", None)
        if adjacency is not None:
            self.set_adjacency(adjacency)
    
    def set_adjacency(self, adj: torch.Tensor) -> None:
        """Set or override the spatial adjacency matrix."""
        adj = _normalize_adjacency(adj.detach())
        self._adjacency = adj
    
    def _get_adjacency(self, batch_size: int) -> torch.Tensor:
        """Get normalized adjacency matrix, defaulting to identity if not set."""
        if self._adjacency is None:
            # Use identity (no spatial interaction) as fallback
            eye = torch.eye(
                self.num_counties,
                device=next(self.parameters()).device
            )
            adj = _normalize_adjacency(eye)
        else:
            adj = self._adjacency
        
        # Expand to batch dimension if needed
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        if adj.size(0) == 1 and batch_size > 1:
            adj = adj.expand(batch_size, -1, -1)
        
        return adj
    
    def forward(
        self,
        x,  # Can be dict or tensor
        adjacency: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Either:
               - Input tensor (batch, past_days, num_counties, in_dim), OR
               - Dict with keys {"x": tensor, "adj": tensor} from graph_collate
            adjacency: Optional adjacency override (N, N) or (B, N, N)
        
        Returns:
            logits: Binary classification logits (batch, num_counties)
        """
        # Handle dictionary input from graph_collate
        if isinstance(x, dict):
            adjacency = x.get("adj", adjacency)
            x = x["x"]
        
        B, T, N, F = x.shape
        
        # Validate input dimensions
        if T != self.past_days:
            raise ValueError(f"Expected past_days={self.past_days}, got {T}")
        if N != self.num_counties:
            raise ValueError(f"Expected num_counties={self.num_counties}, got {N}")
        if F != self.in_dim:
            raise ValueError(f"Expected in_dim={self.in_dim}, got {F}")
        
        # Get adjacency matrix
        adj = (
            _normalize_adjacency(adjacency)
            if adjacency is not None
            else self._get_adjacency(B)
        )
        
        # Embed input features
        x = x.view(B * T, N, F)
        x = self.input_embed(x)
        x = x.view(B, T, N, self.hidden_dim)
        
        # Initialize encoder states
        device = x.device
        encoder_h = [
            torch.zeros(B, N, self.hidden_dim, device=device)
            for _ in range(self.num_layers)
        ]
        encoder_z = [
            torch.zeros(B, N, self.hidden_dim, device=device)
            for _ in range(self.num_layers)
        ]
        
        # Encoder: Process temporal sequence
        for t in range(T):
            x_t = x[:, t, :, :]  # (B, N, hidden_dim)
            
            for i, layer in enumerate(self.encoder_layers):
                if i == 0:
                    h_in = x_t
                else:
                    h_in = encoder_h[i - 1]
                
                encoder_h[i], encoder_z[i] = layer(
                    h_in,
                    encoder_h[i],
                    encoder_z[i],
                    adj=adj,
                )
        
        # Initialize decoder states from encoder
        decoder_h = [h.clone() for h in encoder_h]
        decoder_z = [z.clone() for z in encoder_z]
        
        # Decoder: Single-step prediction
        encoder_vector = encoder_h[-1]  # (B, N, hidden_dim)
        
        for i, layer in enumerate(self.decoder_layers):
            if i == 0:
                h_in = encoder_vector
            else:
                h_in = decoder_h[i - 1]
            
            decoder_h[i], decoder_z[i] = layer(
                h_in,
                decoder_h[i],
                decoder_z[i],
                adj=adj,
            )
        
        # Output projection
        output = decoder_h[-1]  # (B, N, hidden_dim)
        output = self.dropout(output)
        logits = self.output_proj(output).squeeze(-1)  # (B, N)
        
        return logits


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


__all__ = ["ConvLEMWildfire", "GraphConvLEMCell", "convlem_wildfire_builder"]