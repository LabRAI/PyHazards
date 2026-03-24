from __future__ import annotations

import torch.nn as nn

from .qwen25_vl_wildfire_prompted import Qwen25VLWildfirePrompted


class Gemini25ProWildfirePrompted(Qwen25VLWildfirePrompted):
    """Benchmark-facing wildfire VLM baseline inspired by Gemini 2.5 Pro."""


def gemini_25_pro_wildfire_prompted_builder(
    task: str,
    in_channels: int = 6,
    out_dim: int = 1,
    hidden_dim: int = 96,
    prompt_dim: int = 32,
    num_prompt_tokens: int = 6,
    num_heads: int = 8,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() != "segmentation":
        raise ValueError(
            f"gemini_25_pro_wildfire_prompted is segmentation-only in PyHazards, got task={task!r}."
        )
    return Gemini25ProWildfirePrompted(
        in_channels=in_channels,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        prompt_dim=prompt_dim,
        num_prompt_tokens=num_prompt_tokens,
        num_heads=num_heads,
        dropout=dropout,
    )


__all__ = ["Gemini25ProWildfirePrompted", "gemini_25_pro_wildfire_prompted_builder"]
