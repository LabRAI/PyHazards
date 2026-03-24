from __future__ import annotations

import torch.nn as nn

from .qwen25_vl_wildfire_prompted import Qwen25VLWildfirePrompted


class InternVL3WildfirePrompted(Qwen25VLWildfirePrompted):
    """Benchmark-facing wildfire VLM baseline inspired by InternVL3."""


def internvl3_wildfire_prompted_builder(
    task: str,
    in_channels: int = 6,
    out_dim: int = 1,
    hidden_dim: int = 96,
    prompt_dim: int = 32,
    num_prompt_tokens: int = 5,
    num_heads: int = 6,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() != "segmentation":
        raise ValueError(
            f"internvl3_wildfire_prompted is segmentation-only in PyHazards, got task={task!r}."
        )
    return InternVL3WildfirePrompted(
        in_channels=in_channels,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        prompt_dim=prompt_dim,
        num_prompt_tokens=num_prompt_tokens,
        num_heads=num_heads,
        dropout=dropout,
    )


__all__ = ["InternVL3WildfirePrompted", "internvl3_wildfire_prompted_builder"]
