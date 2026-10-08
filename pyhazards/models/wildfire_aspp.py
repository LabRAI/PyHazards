from __future__ import annotations

import torch
import torch.nn as nn

from .cnn_aspp import WildfireCNNASPP, cnn_aspp_builder


class WildfireASPP(WildfireCNNASPP):
    """Public name of the CNN-ASPP wildfire spread model (Marjani et al., IEEE GRSL 2024)."""


def wildfire_aspp_builder(task: str, **kwargs) -> nn.Module:
    """Build CNN-ASPP; input shape ``(batch, channels, height, width)``, raises ValueError otherwise."""
    return cnn_aspp_builder(task=task, **kwargs)


class TverskyLoss(nn.Module):
    """Tversky loss ``1 - TP / (TP + alpha * FP + beta * FN)`` for binary segmentation.

    ``alpha`` weights false positives and ``beta`` false negatives (Salehi et al., 2017).
    Marjani et al. (2024) write the same index with the roles named the other way round and
    train CNN-ASPP with 0.7 on false negatives and 0.3 on false positives, i.e.
    ``TverskyLoss(alpha=0.3, beta=0.7)`` here.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1e-6,
        from_logits: bool = True,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)
        self.from_logits = bool(from_logits)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits) if self.from_logits else logits
        probs = probs.reshape(probs.size(0), -1)
        targets = targets.float().reshape(targets.size(0), -1)

        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1 - targets)).sum(dim=1)
        fn = ((1 - probs) * targets).sum(dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1.0 - tversky).mean()


__all__ = ["TverskyLoss", "WildfireASPP", "wildfire_aspp_builder"]
