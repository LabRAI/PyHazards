"""CNN-ASPP (Marjani et al., IEEE GRSL 2024), rebuilt from the paper description."""

from __future__ import annotations

import pytest
import torch

from pyhazards.models import build_model
from pyhazards.models.wildfire_aspp import TverskyLoss


def test_cnn_aspp_matches_paper_architecture():
    model = build_model("wildfire_aspp", task="segmentation", in_channels=12)
    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    assert [c.out_channels for c in convs] == [64, 128, 32, 32, 32, 32, 32, 32, 1]
    assert [c.dilation[0] for c in model.aspp.modules() if isinstance(c, torch.nn.Conv2d)] == [1, 3, 6, 12]
    assert sum(p.numel() for p in model.parameters()) == 274_657


def test_cnn_aspp_keeps_resolution_and_alias():
    model = build_model("wildfire_cnn_aspp", task="segmentation", in_channels=12).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 12, 64, 64)).shape == (2, 1, 64, 64)
    with pytest.raises(ValueError):
        model(torch.randn(2, 11, 64, 64))


def test_tversky_weights_false_positives_with_alpha():
    targets = torch.zeros(1, 1, 2, 2)
    targets[0, 0, 0, 0] = 1
    logits = torch.full((1, 1, 2, 2), -50.0)
    logits[0, 0, 0, 1] = 50.0  # one false positive, one false negative, no true positive
    fp_heavy = TverskyLoss(alpha=0.9, beta=0.1, smooth=0.0)
    assert torch.isclose(fp_heavy(logits, targets), torch.tensor(1.0))
    logits[0, 0, 0, 0] = 50.0  # now one TP and one FP
    loss = TverskyLoss(alpha=0.3, beta=0.7, smooth=0.0)(logits, targets)
    assert torch.isclose(loss, torch.tensor(1 - 1 / (1 + 0.3)), atol=1e-6)
