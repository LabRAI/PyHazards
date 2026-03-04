from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.init as init


class ConvLEMCell(nn.Module):
    """
    Convolutional Long Expressive Memory (ConvLEM) cell.
    
    Args:
        in_channels: Input feature channels
        out_channels: Hidden/memory state channels
        kernel_size: Convolution kernel size (default: 3)
        dt: Time step parameter for memory integration (default: 1.0)
        activation: Activation function - 'tanh' or 'relu' (default: 'tanh')
        use_reset_gate: Whether to use reset gate variant (default: False)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
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
        self.out_channels = out_channels
        
        padding = (kernel_size - 1) // 2
        
        # Input transformations (convx in paper)
        if use_reset_gate:
            # ConvLEMCell_1: with reset gate (5 chunks)
            self.conv_x = nn.Conv2d(in_channels, 5 * out_channels, kernel_size, padding=padding)
            self.conv_h = nn.Conv2d(out_channels, 4 * out_channels, kernel_size, padding=padding)
        else:
            # ConvLEMCell: without reset gate (4 chunks)
            self.conv_x = nn.Conv2d(in_channels, 4 * out_channels, kernel_size, padding=padding)
            self.conv_h = nn.Conv2d(out_channels, 3 * out_channels, kernel_size, padding=padding)
        
        # Memory transformation (convc in paper)
        self.conv_c = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        
        # Learnable weights for Hadamard products with memory state
        # These are spatial: (out_channels, 1, 1) to broadcast across H, W
        self.W_c1 = nn.Parameter(torch.Tensor(out_channels, 1, 1))
        self.W_c2 = nn.Parameter(torch.Tensor(out_channels, 1, 1))
        if use_reset_gate:
            self.W_c4 = nn.Parameter(torch.Tensor(out_channels, 1, 1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters following WaveCastNet conventions."""
        for name, param in self.named_parameters():
            if 'W_c' in name:
                # Initialize Hadamard weights
                nn.init.constant_(param, 0)
            elif len(param.shape) > 1:
                # Initialize conv weights
                init.xavier_uniform_(param)
            else:
                # Initialize biases
                nn.init.constant_(param, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of ConvLEMCell.
        
        Args:
            x: Input features (B, C_in, H, W)
            h: Hidden state from previous timestep (B, C_out, H, W)
            c: Memory state from previous timestep (B, C_out, H, W)
        
        Returns:
            Tuple of (h_new, c_new) - updated hidden and memory states
        """
        # Transform inputs through convolutions
        conv_x_out = self.conv_x(x)  # (B, K*C_out, H, W)
        conv_h_out = self.conv_h(h)  # (B, L*C_out, H, W)
        
        if self.use_reset_gate:
            # ConvLEMCell_1 with reset gate
            i_dt1, i_dt2, g_dx2, i_c, i_h = torch.chunk(conv_x_out, chunks=5, dim=1)
            h_dt1, h_dt2, h_h, g_dh2 = torch.chunk(conv_h_out, chunks=4, dim=1)
            
            # Memory update gate
            ms_dt = self.dt * torch.sigmoid(i_dt2 + h_dt2 + self.W_c2 * c)
            
            # Update memory state
            c = (1.0 - ms_dt) * c + ms_dt * self.activation(i_h + h_h)
            
            # Reset gate
            gate2 = self.dt * torch.sigmoid(g_dx2 + g_dh2 + self.W_c4 * c)
            
            # Transform memory with reset gate
            conv_c_out = gate2 * self.conv_c(c)
            
            # Hidden state update gate
            ms_dt_bar = self.dt * torch.sigmoid(i_dt1 + h_dt1 + self.W_c1 * c)
            
            # Update hidden state
            h = (1.0 - ms_dt_bar) * h + ms_dt_bar * self.activation(conv_c_out + i_c)
        else:
            # ConvLEMCell without reset gate
            i_dt1, i_dt2, i_c, i_h = torch.chunk(conv_x_out, chunks=4, dim=1)
            h_dt1, h_dt2, h_h = torch.chunk(conv_h_out, chunks=3, dim=1)
            
            # Memory update gate
            ms_dt = self.dt * torch.sigmoid(i_dt2 + h_dt2 + self.W_c2 * c)
            
            # Update memory state
            c = (1.0 - ms_dt) * c + ms_dt * self.activation(i_h + h_h)
            
            # Transform memory (no reset gate)
            conv_c_out = self.conv_c(c)
            
            # Hidden state update gate
            ms_dt_bar = self.dt * torch.sigmoid(i_dt1 + h_dt1 + self.W_c1 * c)
            
            # Update hidden state
            h = (1.0 - ms_dt_bar) * h + ms_dt_bar * self.activation(conv_c_out + i_c)
        
        return h, c

class WaveCastNet(nn.Module):
    """
    WaveCastNet: Sequence-to-sequence wavefield forecasting model.
    
    Args:
        in_channels: Number of input channels (3 for X, Y, Z velocity)
        height: Spatial grid height
        width: Spatial grid width
        temporal_in: Input sequence length (number of timesteps)
        temporal_out: Output sequence length (number of timesteps to predict)
        hidden_dim: Hidden state dimension (default: 144)
        num_layers: Number of ConvLEM encoder/decoder pairs (default: 2)
        kernel_size: Conv2d kernel size (default: 3)
        dt: Time step parameter for LEM mechanism (default: 1.0)
        activation: Activation function - 'tanh' or 'relu' (default: 'tanh')
        dropout: Dropout rate (default: 0.1)
    
    Input:
        x: Wavefield tensor (B, C, T_in, H, W)
    
    Output:
        y: Predicted wavefield (B, C, T_out, H, W)
    """
    
    def __init__(
        self,
        in_channels: int,
        height: int,
        width: int,
        temporal_in: int,
        temporal_out: int,
        hidden_dim: int = 144,
        num_layers: int = 2,
        kernel_size: int = 3,
        dt: float = 1.0,
        activation: str = 'tanh',
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.temporal_in = temporal_in
        self.temporal_out = temporal_out
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        padding = (kernel_size - 1) // 2
        
        # Input embedding: project input channels to hidden dimension
        self.input_embed = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size, padding=padding),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )
        
        # Encoder layers: process input temporal sequence
        self.encoder_layers = nn.ModuleList([
            ConvLEMCell(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                dt=dt,
                activation=activation,
                use_reset_gate=False,
            )
            for _ in range(num_layers)
        ])
        
        # Decoder layers: generate output temporal sequence
        self.decoder_layers = nn.ModuleList([
            ConvLEMCell(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                dt=dt,
                activation=activation,
                use_reset_gate=False,
            )
            for _ in range(num_layers)
        ])
        
        # Output projection: reconstruct wavefield from hidden state
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim // 2, in_channels, kernel_size, padding=padding),
        )
        
        self.dropout = nn.Dropout2d(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for wavefield forecasting.
        
        Args:
            x: Input wavefield (B, C, T_in, H, W)
        
        Returns:
            Predicted future wavefield (B, C, T_out, H, W)
        
        Raises:
            ValueError: If input shape doesn't match expected dimensions
        """
        # Validate input shape
        if x.ndim != 5:
            raise ValueError(
                f"Expected 5D input (B, C, T, H, W), got {x.ndim}D tensor with shape {tuple(x.shape)}"
            )
        
        B, C, T_in, H, W = x.shape
        
        if C != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {C}")
        if T_in != self.temporal_in:
            raise ValueError(f"Expected temporal_in={self.temporal_in}, got {T_in}")
        if H != self.height:
            raise ValueError(f"Expected height={self.height}, got {H}")
        if W != self.width:
            raise ValueError(f"Expected width={self.width}, got {W}")
        
        device = x.device
        
        # ===== ENCODER =====
        # Initialize encoder states for each layer
        encoder_h = [
            torch.zeros(B, self.hidden_dim, H, W, device=device)
            for _ in range(self.num_layers)
        ]
        encoder_c = [
            torch.zeros(B, self.hidden_dim, H, W, device=device)
            for _ in range(self.num_layers)
        ]
        
        # Process input sequence through encoder
        for t in range(T_in):
            x_t = x[:, :, t, :, :]  # (B, C, H, W)
            
            # Embed input
            x_embed = self.input_embed(x_t)  # (B, hidden_dim, H, W)
            
            # Pass through encoder layers
            for i, layer in enumerate(self.encoder_layers):
                if i == 0:
                    h_in = x_embed
                else:
                    h_in = encoder_h[i - 1]
                
                encoder_h[i], encoder_c[i] = layer(
                    h_in,
                    encoder_h[i],
                    encoder_c[i],
                )
        
        # ===== DECODER =====
        # Initialize decoder states from final encoder states
        decoder_h = [h.clone() for h in encoder_h]
        decoder_c = [c.clone() for c in encoder_c]
        
        # Generate output sequence
        outputs = []
        
        for t in range(self.temporal_out):
            # Use final encoder state as input to decoder
            # (In paper, this could be random noise or last encoder output)
            decoder_input = encoder_h[-1] if t == 0 else decoder_h[-1]
            
            # Pass through decoder layers
            for i, layer in enumerate(self.decoder_layers):
                if i == 0:
                    h_in = decoder_input
                else:
                    h_in = decoder_h[i - 1]
                
                decoder_h[i], decoder_c[i] = layer(
                    h_in,
                    decoder_h[i],
                    decoder_c[i],
                )
            
            # Project final decoder state to output
            output_t = decoder_h[-1]  # (B, hidden_dim, H, W)
            output_t = self.dropout(output_t)
            output_t = self.output_proj(output_t)  # (B, C, H, W)
            
            outputs.append(output_t)
        
        # Stack outputs along temporal dimension
        output = torch.stack(outputs, dim=2)  # (B, C, T_out, H, W)
        
        return output

__all__ = ["ConvLEMCell", "WaveCastNet"]