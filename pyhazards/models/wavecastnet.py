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

__all__ = ["ConvLEMCell"]