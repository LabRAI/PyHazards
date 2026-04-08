from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.init as init


class DoubleConv(nn.Module):
    """
    Double convolution block used in U-Net encoder and decoder.
        
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size (default: 3)
        activation: Activation function - 'relu' or 'leaky_relu' (default: 'relu')
        dropout: Dropout rate (default: 0.0)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: str = 'relu',
        dropout: float = 0.0,
    ):
        super().__init__()
        
        padding = (kernel_size - 1) // 2
        
        # Choose activation
        if activation == 'relu':
            act = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            act = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            act,
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            act,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downsampling block for U-Net encoder.
        
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        activation: Activation function
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: str = 'relu',
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel_size, activation, dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upsampling block for U-Net decoder with skip connection.
        
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        activation: Activation function
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: str = 'relu',
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Upsample
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # After concatenation with skip connection: in_channels // 2 + in_channels // 2 = in_channels
        self.conv = DoubleConv(in_channels, out_channels, kernel_size, activation, dropout)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connection.
        
        Args:
            x: Input from previous decoder layer
            skip: Skip connection from encoder
        
        Returns:
            Upsampled and concatenated features
        """
        x = self.up(x)
        
        # Handle potential size mismatch due to pooling
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        
        if diff_h > 0 or diff_w > 0:
            x = nn.functional.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                                      diff_h // 2, diff_h - diff_h // 2])
        
        # Concatenate skip connection
        x = torch.cat([skip, x], dim=1)
        
        return self.conv(x)


class DLWP(nn.Module):
    """
    Deep Learning Weather Prediction (DLWP) model using U-Net architecture.
    
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
        x: Atmospheric state (B, C, F, H, W)
    
    Output:
        y: Predicted future state (B, C_out, F, H, W)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_faces: int = 6,
        face_size: int = 64,
        num_levels: int = 4,
        base_channels: int = 64,
        kernel_size: int = 3,
        activation: str = 'relu',
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.num_faces = num_faces
        self.face_size = face_size
        self.num_levels = num_levels
        self.base_channels = base_channels
        
        # Input projection
        self.inc = DoubleConv(in_channels, base_channels, kernel_size, activation, dropout)
        
        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        for i in range(num_levels):
            out_ch = base_channels * (2 ** (i + 1))
            self.down_blocks.append(Down(in_ch, out_ch, kernel_size, activation, dropout))
            in_ch = out_ch
        
        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        for i in range(num_levels):
            out_ch = base_channels * (2 ** (num_levels - i - 1))
            self.up_blocks.append(Up(in_ch, out_ch, kernel_size, activation, dropout))
            in_ch = out_ch
        
        # Output projection
        self.outc = nn.Conv2d(base_channels, self.out_channels, kernel_size=1)
    
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
        
        # Reshape: (B, C, F, H, W) → (B*F, C, H, W)
        # Process all faces together as separate samples
        x = x.view(B * F, C, H, W)
        
        # ===== ENCODER =====
        # Input projection
        x = self.inc(x)  # (B*F, base_channels, H, W)
        
        # Store skip connections
        skips = [x]
        
        # Downsample
        for down in self.down_blocks:
            x = down(x)
            skips.append(x)
        
        # Remove last skip (bottleneck doesn't need skip to itself)
        skips.pop()
        
        # ===== DECODER =====
        # Upsample with skip connections
        for up in self.up_blocks:
            skip = skips.pop()
            x = up(x, skip)
        
        # ===== OUTPUT =====
        x = self.outc(x)  # (B*F, out_channels, H, W)
        
        # Reshape back: (B*F, C_out, H, W) → (B, C_out, F, H, W)
        x = x.view(B, self.out_channels, F, H, W)
        
        return x


def dlwp_builder(
    task: str,
    in_channels: int,
    num_faces: int,
    face_size: int,
    **kwargs,
) -> DLWP:
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
    """
    # Validate task
    if task.lower() != "regression":
        raise ValueError(
            f"DLWP only supports regression tasks for weather prediction, "
            f"got task='{task}'"
        )
    
    return DLWP(
        in_channels=in_channels,
        out_channels=kwargs.get("out_channels", in_channels),
        num_faces=num_faces,
        face_size=face_size,
        num_levels=kwargs.get("num_levels", 4),
        base_channels=kwargs.get("base_channels", 64),
        kernel_size=kwargs.get("kernel_size", 3),
        activation=kwargs.get("activation", "relu"),
        dropout=kwargs.get("dropout", 0.1),
    )

class WeatherMetrics:
    """
    Metrics for weather prediction evaluation:
        RMSE
        BIAS
        ACC
    """
    
    @staticmethod
    def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute Root Mean Square Error.
        
        Args:
            pred: Predicted atmospheric state (B, C, F, H, W)
            target: Ground truth atmospheric state (B, C, F, H, W)
        
        Returns:
            RMSE value (lower is better)
        """
        mse = torch.mean((pred - target) ** 2)
        rmse = torch.sqrt(mse)
        return rmse.item()
    
    @staticmethod
    def bias(pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute mean bias.
        
        Args:
            pred: Predicted atmospheric state (B, C, F, H, W)
            target: Ground truth atmospheric state (B, C, F, H, W)
        
        Returns:
            Bias value (0 is perfect, positive/negative indicates direction)
        """
        bias = torch.mean(pred - target)
        # Bias = mean(pred - target)
        # Positive bias = systematic overprediction
        # Negative bias = systematic underprediction

        return bias.item()
    
    @staticmethod
    def acc(pred: torch.Tensor, target: torch.Tensor, climatology: Optional[torch.Tensor] = None) -> float:
        """
        Compute Anomaly Correlation Coefficient (ACC).
        
        Args:
            pred: Predicted atmospheric state (B, C, F, H, W)
            target: Ground truth atmospheric state (B, C, F, H, W)
            climatology: Long-term mean state (C, F, H, W), optional
        
        Returns:
            ACC value in [-1, 1], where 1 is perfect correlation
        """
        # Compute anomalies (deviations from climatology)
        if climatology is None:
            # Use target mean as approximation
            climatology = target.mean(dim=0, keepdim=True)
        else:
            climatology = climatology.unsqueeze(0)  # Add batch dim
        
        pred_anom = pred - climatology
        target_anom = target - climatology
        
        # Compute correlation
        numerator = (pred_anom * target_anom).sum()
        denominator = torch.sqrt((pred_anom ** 2).sum() * (target_anom ** 2).sum())
        
        # ACC = Σ(pred_anom * target_anom) / sqrt(Σ(pred_anom²) * Σ(target_anom²))

        acc = numerator / denominator.clamp(min=1e-8)
        
        return acc.item()
    
    @staticmethod
    def compute_all(
        pred: torch.Tensor,
        target: torch.Tensor,
        climatology: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Compute all metrics at once.
        
        Args:
            pred: Predicted atmospheric state (B, C, F, H, W)
            target: Ground truth atmospheric state (B, C, F, H, W)
            climatology: Long-term mean state (C, F, H, W), optional
        
        Returns:
            Dictionary with all metric values
        """
        return {
            "RMSE": WeatherMetrics.rmse(pred, target),
            "Bias": WeatherMetrics.bias(pred, target),
            "ACC": WeatherMetrics.acc(pred, target, climatology),
        }

__all__ = ["DLWP", "dlwp_builder", "DoubleConv", "Down", "Up"]