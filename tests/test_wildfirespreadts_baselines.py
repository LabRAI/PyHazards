"""WildfireSpreadTS baselines: configurations, shapes and input validation.

Numerical equivalence with the reference implementations lives in tests/oracle.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyhazards.model_catalog import REPO_ROOT, load_model_cards
from pyhazards.models import WILDFIRESPREADTS_BASELINES, build_model

# WildfireSpreadTS paper Table 5 (all features, 40 channels); the U-Net count is smp 0.3.2's.
EXPECTED_PARAMS = {
    "logistic_regression": 361,
    "resnet18_unet": 14_444_241,
    "convlstm": 240_449,
    "utae": 1_099_011,
}


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("baseline", WILDFIRESPREADTS_BASELINES)
def test_wildfirespreadts_parameter_counts(baseline):
    model = build_model("wildfirespreadts", task="segmentation", baseline=baseline, in_channels=40, history=1)
    assert _n_params(model) == EXPECTED_PARAMS[baseline]


@pytest.mark.parametrize("baseline", WILDFIRESPREADTS_BASELINES)
def test_wildfirespreadts_forward_shapes(baseline):
    model = build_model("wildfirespreadts", task="segmentation", baseline=baseline, in_channels=6, history=3).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 6, 32, 32))
    assert out.shape == (2, 1, 32, 32)


def test_standalone_builders_match_presets():
    for name, kwargs in {
        "logistic_regression": {},
        "resnet18_unet": {},
        "convlstm": {},
        "utae": {},
    }.items():
        model = build_model(name, task="segmentation", in_channels=40, **kwargs)
        assert _n_params(model) == EXPECTED_PARAMS[name], name


def test_multiday_input_is_flattened_time_major():
    model = build_model("logistic_regression", task="segmentation", in_channels=4, history=2)
    x = torch.randn(1, 2, 4, 8, 8)
    torch.testing.assert_close(model(x), model.conv(torch.cat([x[:, 0], x[:, 1]], dim=1)))


def test_utae_default_positions_are_time_indices():
    model = build_model("utae", task="segmentation", in_channels=3).eval()
    x = torch.randn(2, 4, 3, 16, 16)
    positions = torch.arange(4, dtype=torch.float32).expand(2, -1)
    with torch.no_grad():
        torch.testing.assert_close(model(x), model(x, batch_positions=positions))


def test_utae_handles_padded_dates():
    model = build_model("utae", task="segmentation", in_channels=3).eval()
    x = torch.randn(2, 4, 3, 16, 16)
    x[0, :2] = 0
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "name, shape",
    [
        ("convlstm", (2, 3, 16, 16)),
        ("utae", (2, 3, 16, 16)),
        ("utae", (2, 4, 3, 12, 12)),
        ("resnet18_unet", (2, 3, 48, 48)),
        ("logistic_regression", (2, 16, 16)),
    ],
)
def test_bad_input_shapes_raise(name, shape):
    model = build_model(name, task="segmentation", in_channels=3)
    with pytest.raises(ValueError):
        model(torch.randn(*shape))


def test_unknown_baseline_raises():
    with pytest.raises(ValueError):
        build_model("wildfirespreadts", task="segmentation", baseline="persistence")


def test_reproduction_metadata_points_at_existing_oracle_tests():
    for card in load_model_cards():
        if card.reproduction is None:
            continue
        assert (REPO_ROOT / card.reproduction.oracle_test).exists(), card.model_name


def test_oracle_manifest_pins_full_commits():
    import yaml

    manifest = yaml.safe_load((Path(__file__).parent / "oracle" / "repos.yaml").read_text())
    for name, spec in manifest["repos"].items():
        assert len(spec["commit"]) == 40, name
    for name, spec in manifest["assets"].items():
        assert len(spec["sha256"]) == 64, name
