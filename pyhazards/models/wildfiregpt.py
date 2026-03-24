from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class WildfireGPTReasoner(nn.Module):
    """Retrieval-conditioned wildfire risk model inspired by the WildfireGPT multi-agent RAG system."""

    def __init__(
        self,
        in_channels: int = 12,
        out_dim: int = 1,
        base_channels: int = 32,
        hidden_dim: int = 64,
        profile_dim: int = 8,
        retrieved_dim: int = 16,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")

        self.in_channels = int(in_channels)
        self.profile_dim = int(profile_dim)
        self.retrieved_dim = int(retrieved_dim)
        self.hidden_dim = int(hidden_dim)

        self.raster_encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.profile_proj = nn.Linear(self.profile_dim, hidden_dim)
        self.retrieved_proj = nn.Linear(self.retrieved_dim, hidden_dim)
        self.raster_proj = nn.Linear(hidden_dim, hidden_dim)

        # Learned system-role tokens: user-profile, planner, analyst.
        self.agent_tokens = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Conv2d(hidden_dim // 2, out_dim, kernel_size=1)

    def _unpack_inputs(self, inputs: torch.Tensor | Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if isinstance(inputs, dict):
            x = inputs.get("x")
            profile = inputs.get("user_profile")
            retrieved = inputs.get("retrieved_context")
        else:
            x = inputs
            profile = None
            retrieved = None

        if not isinstance(x, torch.Tensor):
            raise ValueError("WildfireGPTReasoner expects a tensor input or a dict containing key 'x'.")
        if x.ndim != 4:
            raise ValueError(
                "WildfireGPTReasoner expects input shape (B, C, H, W), "
                f"got {tuple(x.shape)}."
            )
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"WildfireGPTReasoner expected in_channels={self.in_channels}, got {x.size(1)}."
            )
        return x, profile, retrieved

    def _coerce_profile(self, profile: torch.Tensor | None, batch: int, device: torch.device) -> torch.Tensor:
        if profile is None:
            return torch.zeros(batch, self.profile_dim, device=device)
        if profile.ndim != 2 or profile.size(0) != batch:
            raise ValueError(f"user_profile must have shape (B,D), got {tuple(profile.shape)}")
        if profile.size(1) == self.profile_dim:
            return profile.to(device=device, dtype=torch.float32)
        if profile.size(1) > self.profile_dim:
            return profile[:, : self.profile_dim].to(device=device, dtype=torch.float32)
        pad = torch.zeros(batch, self.profile_dim - profile.size(1), device=device)
        return torch.cat([profile.to(device=device, dtype=torch.float32), pad], dim=1)

    def _coerce_retrieved(self, retrieved: torch.Tensor | None, batch: int, device: torch.device) -> torch.Tensor:
        if retrieved is None:
            return torch.zeros(batch, self.retrieved_dim, device=device)
        if retrieved.ndim != 2 or retrieved.size(0) != batch:
            raise ValueError(f"retrieved_context must have shape (B,D), got {tuple(retrieved.shape)}")
        if retrieved.size(1) == self.retrieved_dim:
            return retrieved.to(device=device, dtype=torch.float32)
        if retrieved.size(1) > self.retrieved_dim:
            return retrieved[:, : self.retrieved_dim].to(device=device, dtype=torch.float32)
        pad = torch.zeros(batch, self.retrieved_dim - retrieved.size(1), device=device)
        return torch.cat([retrieved.to(device=device, dtype=torch.float32), pad], dim=1)

    def forward(self, inputs: torch.Tensor | Dict[str, Any]) -> torch.Tensor:
        x, profile, retrieved = self._unpack_inputs(inputs)
        batch = x.size(0)
        device = x.device

        feature_map = self.raster_encoder(x)
        pooled_raster = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)

        profile_token = self.profile_proj(self._coerce_profile(profile, batch, device)).unsqueeze(1)
        retrieved_token = self.retrieved_proj(self._coerce_retrieved(retrieved, batch, device)).unsqueeze(1)
        raster_token = self.raster_proj(pooled_raster).unsqueeze(1)
        agent_tokens = self.agent_tokens.unsqueeze(0).expand(batch, -1, -1)

        tokens = torch.cat([agent_tokens, profile_token, retrieved_token, raster_token], dim=1)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        fused_tokens = attn_out + self.ffn(attn_out)
        fused_global = fused_tokens.mean(dim=1)

        fused_map = fused_global.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, feature_map.size(-2), feature_map.size(-1))
        decoded = self.decoder(torch.cat([feature_map, fused_map], dim=1))
        return self.head(decoded)



def wildfiregpt_builder(
    task: str,
    in_channels: int = 12,
    out_dim: int = 1,
    base_channels: int = 32,
    hidden_dim: int = 64,
    profile_dim: int = 8,
    retrieved_dim: int = 16,
    num_heads: int = 4,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() != "segmentation":
        raise ValueError(f"wildfiregpt is segmentation-only in PyHazards, got task={task!r}.")
    return WildfireGPTReasoner(
        in_channels=in_channels,
        out_dim=out_dim,
        base_channels=base_channels,
        hidden_dim=hidden_dim,
        profile_dim=profile_dim,
        retrieved_dim=retrieved_dim,
        num_heads=num_heads,
        dropout=dropout,
    )


__all__ = ["WildfireGPTReasoner", "wildfiregpt_builder"]
