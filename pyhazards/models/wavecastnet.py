from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init


class ConvLEMCell(nn.Module):
    """
    Convolutional Long Expressive Memory (ConvLEM) cell used by WaveCastNet.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dt: float = 1.0,
        activation: str = "tanh",
        use_reset_gate: bool = False,
    ):
        super().__init__()

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        else:
            raise ValueError(
                "Unsupported activation: {activation}. Use 'tanh' or 'relu'.".format(
                    activation=activation
                )
            )

        self.dt = float(dt)
        self.use_reset_gate = bool(use_reset_gate)
        self.out_channels = int(out_channels)

        padding = (kernel_size - 1) // 2
        if self.use_reset_gate:
            self.conv_x = nn.Conv2d(
                in_channels,
                5 * out_channels,
                kernel_size,
                padding=padding,
            )
            self.conv_h = nn.Conv2d(
                out_channels,
                4 * out_channels,
                kernel_size,
                padding=padding,
            )
        else:
            self.conv_x = nn.Conv2d(
                in_channels,
                4 * out_channels,
                kernel_size,
                padding=padding,
            )
            self.conv_h = nn.Conv2d(
                out_channels,
                3 * out_channels,
                kernel_size,
                padding=padding,
            )

        self.conv_c = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.W_c1 = nn.Parameter(torch.empty(out_channels, 1, 1))
        self.W_c2 = nn.Parameter(torch.empty(out_channels, 1, 1))
        if self.use_reset_gate:
            self.W_c4 = nn.Parameter(torch.empty(out_channels, 1, 1))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if "W_c" in name:
                nn.init.constant_(param, 0.0)
            elif param.ndim > 1:
                init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or h.ndim != 4 or c.ndim != 4:
            raise ValueError("ConvLEMCell expects x, h, c shaped (B, C, H, W).")

        conv_x_out = self.conv_x(x)
        conv_h_out = self.conv_h(h)

        if self.use_reset_gate:
            i_dt1, i_dt2, g_dx2, i_c, i_h = torch.chunk(conv_x_out, chunks=5, dim=1)
            h_dt1, h_dt2, h_h, g_dh2 = torch.chunk(conv_h_out, chunks=4, dim=1)

            ms_dt = self.dt * torch.sigmoid(i_dt2 + h_dt2 + self.W_c2 * c)
            c = (1.0 - ms_dt) * c + ms_dt * self.activation(i_h + h_h)

            gate2 = self.dt * torch.sigmoid(g_dx2 + g_dh2 + self.W_c4 * c)
            conv_c_out = gate2 * self.conv_c(c)

            ms_dt_bar = self.dt * torch.sigmoid(i_dt1 + h_dt1 + self.W_c1 * c)
            h = (1.0 - ms_dt_bar) * h + ms_dt_bar * self.activation(conv_c_out + i_c)
        else:
            i_dt1, i_dt2, i_c, i_h = torch.chunk(conv_x_out, chunks=4, dim=1)
            h_dt1, h_dt2, h_h = torch.chunk(conv_h_out, chunks=3, dim=1)

            ms_dt = self.dt * torch.sigmoid(i_dt2 + h_dt2 + self.W_c2 * c)
            c = (1.0 - ms_dt) * c + ms_dt * self.activation(i_h + h_h)

            conv_c_out = self.conv_c(c)
            ms_dt_bar = self.dt * torch.sigmoid(i_dt1 + h_dt1 + self.W_c1 * c)
            h = (1.0 - ms_dt_bar) * h + ms_dt_bar * self.activation(conv_c_out + i_c)

        return h, c


class WaveCastNet(nn.Module):
    """
    Sequence-to-sequence wavefield forecasting model based on ConvLEM cells.

    Input shape: (B, C, T_in, H, W)
    Output shape: (B, C, T_out, H, W)
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
        activation: str = "tanh",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.height = int(height)
        self.width = int(width)
        self.temporal_in = int(temporal_in)
        self.temporal_out = int(temporal_out)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)

        padding = (kernel_size - 1) // 2
        proj_dim = max(1, self.hidden_dim // 2)

        self.input_embed = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_dim, kernel_size, padding=padding),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        self.encoder_layers = nn.ModuleList(
            [
                ConvLEMCell(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    kernel_size=kernel_size,
                    dt=dt,
                    activation=activation,
                    use_reset_gate=False,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                ConvLEMCell(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    kernel_size=kernel_size,
                    dt=dt,
                    activation=activation,
                    use_reset_gate=False,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.output_proj = nn.Sequential(
            nn.Conv2d(self.hidden_dim, proj_dim, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(proj_dim, self.in_channels, kernel_size, padding=padding),
        )
        self.dropout = nn.Dropout2d(dropout)

    def _init_states(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        hidden = [
            x.new_zeros(x.size(0), self.hidden_dim, self.height, self.width)
            for _ in range(self.num_layers)
        ]
        memory = [
            x.new_zeros(x.size(0), self.hidden_dim, self.height, self.width)
            for _ in range(self.num_layers)
        ]
        return hidden, memory

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                "WaveCastNet expects x shaped (B, C, T, H, W), got {shape}".format(
                    shape=tuple(x.shape)
                )
            )

        batch_size, channels, temporal_in, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(
                "Expected in_channels={expected}, got {actual}".format(
                    expected=self.in_channels,
                    actual=channels,
                )
            )
        if temporal_in != self.temporal_in:
            raise ValueError(
                "Expected temporal_in={expected}, got {actual}".format(
                    expected=self.temporal_in,
                    actual=temporal_in,
                )
            )
        if height != self.height or width != self.width:
            raise ValueError(
                "Expected spatial size ({h}, {w}), got ({actual_h}, {actual_w})".format(
                    h=self.height,
                    w=self.width,
                    actual_h=height,
                    actual_w=width,
                )
            )

        encoder_h, encoder_c = self._init_states(x)
        for t in range(self.temporal_in):
            encoded = self.input_embed(x[:, :, t, :, :])
            for i, layer in enumerate(self.encoder_layers):
                layer_input = encoded if i == 0 else encoder_h[i - 1]
                encoder_h[i], encoder_c[i] = layer(layer_input, encoder_h[i], encoder_c[i])

        decoder_h = [state.clone() for state in encoder_h]
        decoder_c = [state.clone() for state in encoder_c]

        outputs = []
        for t in range(self.temporal_out):
            decoder_input = encoder_h[-1] if t == 0 else decoder_h[-1]
            for i, layer in enumerate(self.decoder_layers):
                layer_input = decoder_input if i == 0 else decoder_h[i - 1]
                decoder_h[i], decoder_c[i] = layer(layer_input, decoder_h[i], decoder_c[i])
            output_t = self.output_proj(self.dropout(decoder_h[-1]))
            outputs.append(output_t)

        if len(outputs) != self.temporal_out:
            raise RuntimeError(
                "Decoder generated {actual} steps, expected {expected}".format(
                    actual=len(outputs),
                    expected=self.temporal_out,
                )
            )
        return torch.stack(outputs, dim=2)


def wavecastnet_builder(
    task: str,
    in_channels: int,
    height: int,
    width: int,
    temporal_in: int,
    temporal_out: int,
    **kwargs,
) -> WaveCastNet:
    if task.lower() != "regression":
        raise ValueError("WaveCastNet only supports regression tasks.")

    return WaveCastNet(
        in_channels=in_channels,
        height=height,
        width=width,
        temporal_in=temporal_in,
        temporal_out=temporal_out,
        hidden_dim=kwargs.get("hidden_dim", 144),
        num_layers=kwargs.get("num_layers", 2),
        kernel_size=kwargs.get("kernel_size", 3),
        dt=kwargs.get("dt", 1.0),
        activation=kwargs.get("activation", "tanh"),
        dropout=kwargs.get("dropout", 0.1),
    )


class WaveCastNetLoss(nn.Module):
    """
    Huber loss used in the WaveCastNet paper.
    """

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = float(delta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        abs_diff = diff.abs()
        quadratic = 0.5 * diff.square()
        linear = self.delta * abs_diff - 0.5 * self.delta**2
        return torch.where(abs_diff <= self.delta, quadratic, linear).mean()


class WavefieldMetrics:
    """
    ACC and RFNE metrics reported in the WaveCastNet paper.
    """

    @staticmethod
    def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
        pred_flat = pred.reshape(pred.size(0), -1)
        target_flat = target.reshape(target.size(0), -1)
        numerator = (pred_flat * target_flat).sum(dim=1)
        pred_norm = pred_flat.square().sum(dim=1).sqrt()
        target_norm = target_flat.square().sum(dim=1).sqrt()
        acc = numerator / (pred_norm * target_norm).clamp(min=1e-8)
        return float(acc.mean().detach().cpu())

    @staticmethod
    def rfne(pred: torch.Tensor, target: torch.Tensor) -> float:
        error_norm = (pred - target).reshape(pred.size(0), -1).square().sum(dim=1).sqrt()
        target_norm = target.reshape(target.size(0), -1).square().sum(dim=1).sqrt()
        rfne = error_norm / target_norm.clamp(min=1e-8)
        return float(rfne.mean().detach().cpu())

    @staticmethod
    def compute_all(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        return {
            "ACC": WavefieldMetrics.accuracy(pred, target),
            "RFNE": WavefieldMetrics.rfne(pred, target),
        }


__all__ = [
    "ConvLEMCell",
    "WaveCastNet",
    "WaveCastNetLoss",
    "WavefieldMetrics",
    "wavecastnet_builder",
]
