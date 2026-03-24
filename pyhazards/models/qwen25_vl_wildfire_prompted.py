from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class Qwen25VLWildfirePrompted(nn.Module):
    """Prompt-conditioned wildfire segmentation model inspired by Qwen2.5-VL."""

    def __init__(
        self,
        in_channels: int = 6,
        out_dim: int = 1,
        hidden_dim: int = 64,
        prompt_dim: int = 24,
        num_prompt_tokens: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if prompt_dim <= 0:
            raise ValueError(f"prompt_dim must be positive, got {prompt_dim}")
        if num_prompt_tokens <= 0:
            raise ValueError(f"num_prompt_tokens must be positive, got {num_prompt_tokens}")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")

        self.in_channels = int(in_channels)
        self.prompt_dim = int(prompt_dim)
        self.num_prompt_tokens = int(num_prompt_tokens)
        self.hidden_dim = int(hidden_dim)

        self.visual_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.prompt_proj = nn.Linear(self.prompt_dim, hidden_dim)
        self.prompt_bank = nn.Parameter(torch.randn(self.num_prompt_tokens, self.prompt_dim) * 0.02)
        self.image_summary = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
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
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Conv2d(hidden_dim // 2, out_dim, kernel_size=1)

    def _unpack_inputs(self, inputs: torch.Tensor | Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(inputs, dict):
            x = inputs.get("x")
            prompt = inputs.get("prompt_context")
        else:
            x = inputs
            prompt = None

        if not isinstance(x, torch.Tensor):
            raise ValueError("Qwen25VLWildfirePrompted expects a tensor input or a dict containing key 'x'.")
        if x.ndim != 4:
            raise ValueError(
                "Qwen25VLWildfirePrompted expects input shape (B, C, H, W), "
                f"got {tuple(x.shape)}."
            )
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Qwen25VLWildfirePrompted expected in_channels={self.in_channels}, got {x.size(1)}."
            )
        return x, prompt

    def _coerce_prompt(self, prompt: torch.Tensor | None, batch: int, device: torch.device) -> torch.Tensor:
        learned = self.prompt_bank.unsqueeze(0).expand(batch, -1, -1)
        if prompt is None:
            return self.prompt_proj(learned)
        if prompt.ndim == 2:
            if prompt.size(0) != batch:
                raise ValueError(f"prompt_context must have shape (B,D) or (B,T,D), got {tuple(prompt.shape)}")
            prompt = prompt.unsqueeze(1).expand(-1, self.num_prompt_tokens, -1)
        elif prompt.ndim == 3:
            if prompt.size(0) != batch:
                raise ValueError(f"prompt_context must have shape (B,D) or (B,T,D), got {tuple(prompt.shape)}")
            if prompt.size(1) != self.num_prompt_tokens:
                if prompt.size(1) > self.num_prompt_tokens:
                    prompt = prompt[:, : self.num_prompt_tokens]
                else:
                    pad = torch.zeros(batch, self.num_prompt_tokens - prompt.size(1), prompt.size(2), device=prompt.device)
                    prompt = torch.cat([prompt, pad], dim=1)
        else:
            raise ValueError(f"prompt_context must have rank 2 or 3, got {tuple(prompt.shape)}")

        prompt = prompt.to(device=device, dtype=torch.float32)
        if prompt.size(-1) > self.prompt_dim:
            prompt = prompt[..., : self.prompt_dim]
        elif prompt.size(-1) < self.prompt_dim:
            pad = torch.zeros(batch, self.num_prompt_tokens, self.prompt_dim - prompt.size(-1), device=device)
            prompt = torch.cat([prompt, pad], dim=-1)
        return self.prompt_proj(prompt + learned.to(device=device, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor | Dict[str, Any]) -> torch.Tensor:
        x, prompt = self._unpack_inputs(inputs)
        batch = x.size(0)
        device = x.device

        feature_map = self.visual_encoder(x)
        visual_tokens = feature_map.flatten(2).transpose(1, 2)
        pooled = torch.mean(visual_tokens, dim=1, keepdim=True)
        pooled = self.image_summary(pooled)

        prompt_tokens = self._coerce_prompt(prompt, batch, device)
        query_tokens = torch.cat([prompt_tokens, pooled], dim=1)
        attn_out, _ = self.cross_attn(query_tokens, visual_tokens, visual_tokens, need_weights=False)
        fused_tokens = attn_out + self.ffn(attn_out)
        global_token = fused_tokens.mean(dim=1)

        context_map = global_token.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, feature_map.size(-2), feature_map.size(-1))
        decoded = self.decoder(torch.cat([feature_map, context_map], dim=1))
        return self.head(decoded)


def qwen25_vl_wildfire_prompted_builder(
    task: str,
    in_channels: int = 6,
    out_dim: int = 1,
    hidden_dim: int = 64,
    prompt_dim: int = 24,
    num_prompt_tokens: int = 4,
    num_heads: int = 4,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() != "segmentation":
        raise ValueError(
            f"qwen25_vl_wildfire_prompted is segmentation-only in PyHazards, got task={task!r}."
        )
    return Qwen25VLWildfirePrompted(
        in_channels=in_channels,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        prompt_dim=prompt_dim,
        num_prompt_tokens=num_prompt_tokens,
        num_heads=num_heads,
        dropout=dropout,
    )


__all__ = ["Qwen25VLWildfirePrompted", "qwen25_vl_wildfire_prompted_builder"]
