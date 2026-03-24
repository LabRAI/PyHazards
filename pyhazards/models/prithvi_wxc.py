from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeatherSequencePatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        return self.proj(x.permute(0, 2, 1, 3, 4))


class PrithviWxCBackbone(nn.Module):
    """Weather-climate backbone inspired by Prithvi-WxC."""

    def __init__(
        self,
        image_size: int = 32,
        in_channels: int = 8,
        patch_size: int = 4,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        lead_time_dim: int = 1,
        variable_summary_dim: int = 8,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size={image_size} must be divisible by patch_size={patch_size}")

        self.image_size = int(image_size)
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.lead_time_dim = int(lead_time_dim)
        self.variable_summary_dim = int(variable_summary_dim)
        self.grid_size = self.image_size // self.patch_size

        self.patch_embed = WeatherSequencePatchEmbed(
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
        )
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.grid_size * self.grid_size, self.embed_dim)
        )
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)

        self.lead_time_proj = nn.Linear(self.lead_time_dim, self.embed_dim)
        self.variable_proj = nn.Linear(self.variable_summary_dim, self.embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=int(num_heads),
            dim_feedforward=int(self.embed_dim * mlp_ratio),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(depth))
        self.norm = nn.LayerNorm(self.embed_dim)

    def _unpack_inputs(
        self,
        inputs: torch.Tensor | Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if isinstance(inputs, dict):
            x = inputs.get("x")
            lead_time = inputs.get("lead_time_hours")
            variable_summary = inputs.get("variable_summary")
        else:
            x = inputs
            lead_time = None
            variable_summary = None

        if not isinstance(x, torch.Tensor):
            raise ValueError("PrithviWxCBackbone expects a tensor input or a dict containing key 'x'.")
        if x.ndim != 5:
            raise ValueError(
                "PrithviWxCBackbone expects input shape (B, T, C, H, W), "
                f"got {tuple(x.shape)}."
            )
        if x.size(2) != self.in_channels:
            raise ValueError(
                f"PrithviWxCBackbone expected in_channels={self.in_channels}, got {x.size(2)}."
            )
        if x.size(-1) != self.image_size or x.size(-2) != self.image_size:
            raise ValueError(
                f"PrithviWxCBackbone expected spatial size {self.image_size}x{self.image_size}, "
                f"got {tuple(x.shape[-2:])}."
            )
        return x, lead_time, variable_summary

    def _build_lead_time(
        self,
        batch: int,
        timesteps: int,
        device: torch.device,
        lead_time: torch.Tensor | None,
    ) -> torch.Tensor:
        if lead_time is None:
            base = torch.linspace(0.0, 1.0, timesteps, device=device).view(1, timesteps, 1)
            return base.expand(batch, -1, -1)
        if lead_time.ndim == 1:
            lead_time = lead_time.view(batch, 1, 1).expand(-1, timesteps, -1)
        elif lead_time.ndim == 2:
            lead_time = lead_time.unsqueeze(-1)
        if lead_time.ndim != 3:
            raise ValueError(
                f"lead_time_hours must have shape (B,), (B,T), or (B,T,D), got {tuple(lead_time.shape)}"
            )
        if lead_time.size(0) != batch or lead_time.size(1) != timesteps:
            raise ValueError(
                "lead_time_hours batch/timestep mismatch: "
                f"expected ({batch},{timesteps}), got ({lead_time.size(0)},{lead_time.size(1)})"
            )
        if lead_time.size(-1) == self.lead_time_dim:
            return lead_time.to(device=device, dtype=torch.float32)
        if lead_time.size(-1) > self.lead_time_dim:
            return lead_time[..., : self.lead_time_dim].to(device=device, dtype=torch.float32)
        pad = torch.zeros(batch, timesteps, self.lead_time_dim - lead_time.size(-1), device=device)
        return torch.cat([lead_time.to(device=device, dtype=torch.float32), pad], dim=-1)

    def _build_variable_summary(
        self,
        x: torch.Tensor,
        variable_summary: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, timesteps, channels, _, _ = x.shape
        if variable_summary is None:
            summary = x.mean(dim=(-1, -2))
        else:
            summary = variable_summary
            if summary.ndim == 2:
                summary = summary.unsqueeze(1).expand(-1, timesteps, -1)
            if summary.ndim != 3:
                raise ValueError(
                    "variable_summary must have shape (B,D) or (B,T,D), "
                    f"got {tuple(summary.shape)}"
                )
            if summary.size(0) != batch or summary.size(1) != timesteps:
                raise ValueError(
                    "variable_summary batch/timestep mismatch: "
                    f"expected ({batch},{timesteps}), got ({summary.size(0)},{summary.size(1)})"
                )
        if summary.size(-1) > self.variable_summary_dim:
            summary = summary[..., : self.variable_summary_dim]
        elif summary.size(-1) < self.variable_summary_dim:
            pad = torch.zeros(
                batch,
                timesteps,
                self.variable_summary_dim - summary.size(-1),
                device=x.device,
                dtype=torch.float32,
            )
            summary = torch.cat([summary.to(device=x.device, dtype=torch.float32), pad], dim=-1)
        else:
            summary = summary.to(device=x.device, dtype=torch.float32)
        return summary

    def forward(self, inputs: torch.Tensor | Dict[str, Any]) -> torch.Tensor:
        x, lead_time, variable_summary = self._unpack_inputs(inputs)
        batch, timesteps, _, _, _ = x.shape

        feat = self.patch_embed(x)
        _, _, _, h_tokens, w_tokens = feat.shape
        tokens = feat.permute(0, 2, 3, 4, 1).reshape(batch, timesteps * h_tokens * w_tokens, self.embed_dim)

        spatial_pos = self.spatial_pos_embed.unsqueeze(1).expand(-1, timesteps, -1, -1)
        spatial_pos = spatial_pos.reshape(1, timesteps * h_tokens * w_tokens, self.embed_dim)
        tokens = tokens + spatial_pos

        lead = self._build_lead_time(batch, timesteps, x.device, lead_time)
        lead_tokens = self.lead_time_proj(lead).unsqueeze(2).expand(-1, -1, h_tokens * w_tokens, -1)
        tokens = tokens + lead_tokens.reshape(batch, timesteps * h_tokens * w_tokens, self.embed_dim)

        var_summary = self._build_variable_summary(x, variable_summary)
        variable_tokens = self.variable_proj(var_summary).unsqueeze(2).expand(-1, -1, h_tokens * w_tokens, -1)
        tokens = tokens + variable_tokens.reshape(batch, timesteps * h_tokens * w_tokens, self.embed_dim)

        encoded = self.norm(self.encoder(tokens))
        encoded = encoded.reshape(batch, timesteps, h_tokens, w_tokens, self.embed_dim).mean(dim=1)
        return encoded.permute(0, 3, 1, 2).contiguous()


class PrithviWxC(nn.Module):
    """Dense wildfire-risk head on top of a Prithvi-WxC-style weather backbone."""

    def __init__(
        self,
        image_size: int = 32,
        in_channels: int = 8,
        out_dim: int = 1,
        patch_size: int = 4,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        lead_time_dim: int = 1,
        variable_summary_dim: int = 8,
        decoder_channels: int = 64,
    ):
        super().__init__()
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")
        self.backbone = PrithviWxCBackbone(
            image_size=image_size,
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            lead_time_dim=lead_time_dim,
            variable_summary_dim=variable_summary_dim,
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, decoder_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Conv2d(decoder_channels, out_dim, kernel_size=1)

    def forward(self, inputs: torch.Tensor | Dict[str, Any]) -> torch.Tensor:
        x = inputs["x"] if isinstance(inputs, dict) else inputs
        features = self.backbone(inputs)
        logits = self.head(self.decoder(features))
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def prithvi_wxc_builder(
    task: str,
    image_size: int = 32,
    in_channels: int = 8,
    out_dim: int = 1,
    patch_size: int = 4,
    embed_dim: int = 128,
    depth: int = 4,
    num_heads: int = 4,
    mlp_ratio: float = 4.0,
    dropout: float = 0.1,
    lead_time_dim: int = 1,
    variable_summary_dim: int = 8,
    decoder_channels: int = 64,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() != "segmentation":
        raise ValueError(f"prithvi_wxc is segmentation-only, got task={task!r}.")
    return PrithviWxC(
        image_size=image_size,
        in_channels=in_channels,
        out_dim=out_dim,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        lead_time_dim=lead_time_dim,
        variable_summary_dim=variable_summary_dim,
        decoder_channels=decoder_channels,
    )


__all__ = ["PrithviWxCBackbone", "PrithviWxC", "prithvi_wxc_builder"]
