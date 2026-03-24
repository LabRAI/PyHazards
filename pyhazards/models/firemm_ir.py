from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassAwareMemory(nn.Module):
    """Small memory bank inspired by FireMM-IR's class-aware memory module."""

    def __init__(self, hidden_dim: int, num_memory_slots: int = 3):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_memory_slots = int(num_memory_slots)
        self.memory = nn.Parameter(torch.randn(self.num_memory_slots, self.hidden_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        scores = torch.matmul(tokens, self.memory.t()) / max(1.0, self.hidden_dim ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        retrieved = torch.matmul(weights, self.memory).transpose(1, 2).reshape(batch, channels, height, width)
        return x + retrieved


class FireMMIR(nn.Module):
    """Dual-modality wildfire scene model inspired by FireMM-IR."""

    def __init__(
        self,
        in_channels: int = 6,
        out_dim: int = 1,
        hidden_dim: int = 64,
        instruction_dim: int = 16,
        num_memory_slots: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if in_channels < 2 or in_channels % 2 != 0:
            raise ValueError(f"in_channels must be an even number >= 2, got {in_channels}")
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")

        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.instruction_dim = int(instruction_dim)
        self.optical_channels = self.in_channels // 2
        self.infrared_channels = self.in_channels - self.optical_channels

        self.optical_encoder = ModalityEncoder(self.optical_channels, hidden_dim)
        self.infrared_encoder = ModalityEncoder(self.infrared_channels, hidden_dim)
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.memory = ClassAwareMemory(hidden_dim=hidden_dim, num_memory_slots=num_memory_slots)

        self.instruction_proj = nn.Linear(self.instruction_dim, hidden_dim)
        self.segmentation_token = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.token_attn = nn.MultiheadAttention(
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

    def _unpack_inputs(self, inputs: torch.Tensor | Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(inputs, dict):
            x = inputs.get("x")
            instruction = inputs.get("instruction_context")
        else:
            x = inputs
            instruction = None

        if not isinstance(x, torch.Tensor):
            raise ValueError("FireMMIR expects a tensor input or a dict containing key 'x'.")
        if x.ndim != 4:
            raise ValueError(f"FireMMIR expects input shape (B, C, H, W), got {tuple(x.shape)}")
        if x.size(1) != self.in_channels:
            raise ValueError(f"FireMMIR expected in_channels={self.in_channels}, got {x.size(1)}")
        return x, instruction

    def _coerce_instruction(self, instruction: torch.Tensor | None, batch: int, device: torch.device) -> torch.Tensor:
        if instruction is None:
            return torch.zeros(batch, self.instruction_dim, device=device)
        if instruction.ndim != 2 or instruction.size(0) != batch:
            raise ValueError(f"instruction_context must have shape (B,D), got {tuple(instruction.shape)}")
        if instruction.size(1) == self.instruction_dim:
            return instruction.to(device=device, dtype=torch.float32)
        if instruction.size(1) > self.instruction_dim:
            return instruction[:, : self.instruction_dim].to(device=device, dtype=torch.float32)
        pad = torch.zeros(batch, self.instruction_dim - instruction.size(1), device=device)
        return torch.cat([instruction.to(device=device, dtype=torch.float32), pad], dim=1)

    def forward(self, inputs: torch.Tensor | Dict[str, Any]) -> torch.Tensor:
        x, instruction = self._unpack_inputs(inputs)
        batch = x.size(0)
        device = x.device

        optical = x[:, : self.optical_channels]
        infrared = x[:, self.optical_channels :]
        optical_feat = self.optical_encoder(optical)
        infrared_feat = self.infrared_encoder(infrared)
        gate = self.fusion_gate(torch.cat([optical_feat, infrared_feat], dim=1))
        fused = optical_feat + gate * infrared_feat
        fused = self.memory(fused)

        visual_tokens = fused.flatten(2).transpose(1, 2)
        instruction_token = self.instruction_proj(self._coerce_instruction(instruction, batch, device)).unsqueeze(1)
        seg_token = self.segmentation_token.unsqueeze(0).expand(batch, -1, -1)
        query_tokens = torch.cat([seg_token, instruction_token], dim=1)
        attn_out, _ = self.token_attn(query_tokens, visual_tokens, visual_tokens, need_weights=False)
        query_tokens = attn_out + self.ffn(attn_out)
        global_token = query_tokens.mean(dim=1)

        context_map = global_token.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, fused.size(-2), fused.size(-1))
        decoded = self.decoder(torch.cat([fused, context_map], dim=1))
        return self.head(decoded)


def firemm_ir_builder(
    task: str,
    in_channels: int = 6,
    out_dim: int = 1,
    hidden_dim: int = 64,
    instruction_dim: int = 16,
    num_memory_slots: int = 3,
    num_heads: int = 4,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() != "segmentation":
        raise ValueError(f"firemm_ir is segmentation-only in PyHazards, got task={task!r}.")
    return FireMMIR(
        in_channels=in_channels,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        instruction_dim=instruction_dim,
        num_memory_slots=num_memory_slots,
        num_heads=num_heads,
        dropout=dropout,
    )


__all__ = ["FireMMIR", "firemm_ir_builder"]
