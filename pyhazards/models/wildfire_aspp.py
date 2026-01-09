from __future__ import annotations

import torch
import torch.nn as nn

from .cnn_aspp import WildfireCNNASPP, cnn_aspp_builder


class WildfireASPP(WildfireCNNASPP):
    """
    Backward-compatible name for the CNN + ASPP wildfire model.
    """


def wildfire_aspp_builder(*args, **kwargs) -> nn.Module:
    return cnn_aspp_builder(*args, **kwargs)


class TverskyLoss(nn.Module):
    """
    Tversky loss for binary segmentation.
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
        if self.from_logits:
            probs = torch.sigmoid(logits)
        else:
            probs = logits

        targets = targets.float()

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1 - targets)).sum(dim=1)
        fn = ((1 - probs) * targets).sum(dim=1)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        loss = 1.0 - tversky
        return loss.mean()
