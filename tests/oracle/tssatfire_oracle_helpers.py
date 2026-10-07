"""Shared helpers for the MONAI / TS-SatFire oracle tests (unet3d, unetr, swin_unetr, ts_satfire).

References: ``monai==1.3.2`` (the version in TS-SatFire's environment.yml; requirements in
``requirements-tssatfire.txt``) and TS-SatFire's ``spatial_models`` at the commit pinned in
repos.yaml (no LICENSE, so it is only imported here as an oracle).
"""

from __future__ import annotations

import warnings
from types import ModuleType

import torch

from oracle_utils import import_from, oracle_package, oracle_repo

REQUIREMENTS = "requirements-tssatfire.txt"
TS_SATFIRE_CHANNELS = (64, 128, 256, 512, 1024)
# 27 TS-SatFire bands with the 17-class land cover one-hot encoded (FireDataset.preprocess).
PRED_CHANNELS = 43


def monai_nets() -> ModuleType:
    oracle_package("monai", "1.3.2", REQUIREMENTS)
    oracle_package("einops", "0.8.0", REQUIREMENTS)
    from monai.networks import nets

    return nets


def ts_satfire_module(module: str) -> ModuleType:
    """Import ``spatial_models.<module>`` from the pinned TS-SatFire checkout."""
    monai_nets()  # TS-SatFire's files import MONAI blocks
    if module.startswith("swinunetr"):
        oracle_package("torchinfo", "1.8.0", REQUIREMENTS)  # imported by swinunetr.py
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return import_from(oracle_repo("TS-SatFire"), f"spatial_models.{module}")


def n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def assert_same_state(reference: torch.nn.Module, port: torch.nn.Module) -> None:
    """Identical parameter/buffer names in the same order, and identical values."""
    ref_state, port_state = reference.state_dict(), port.state_dict()
    assert list(ref_state) == list(port_state)
    for key, value in ref_state.items():
        assert torch.equal(value, port_state[key]), key


def assert_same_names_and_shapes(reference: torch.nn.Module, port: torch.nn.Module) -> None:
    ref_state, port_state = reference.state_dict(), port.state_dict()
    assert list(ref_state) == list(port_state)
    for key, value in ref_state.items():
        assert value.shape == port_state[key].shape, key


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def randomise(reference: torch.nn.Module, port: torch.nn.Module, seed: int) -> None:
    """Give both models the same non-trivial weights, BatchNorm statistics and PReLU slopes."""
    generator = torch.Generator().manual_seed(seed)
    state = reference.state_dict()
    for key, value in state.items():
        if not value.is_floating_point():
            continue
        if key.endswith("running_var"):
            value.copy_(torch.rand(value.shape, generator=generator) + 0.5)
        elif key.endswith("running_mean"):
            value.copy_(0.1 * torch.randn(value.shape, generator=generator))
        else:
            value.add_(0.05 * torch.randn(value.shape, generator=generator))
    reference.load_state_dict(state, strict=True)
    port.load_state_dict(state, strict=True)


def compare_eval_and_train(reference, port, x_ref, x_port, reduce_time: bool = False, seed: int = 0) -> None:
    """Outputs equal in eval mode and in train mode (same seed before each call), then equal states.

    With ``reduce_time`` the port takes PyHazards' ``(B, T, C, H, W)`` layout and averages over
    time, as TS-SatFire's prediction script does with ``outputs.mean(2)``.
    """

    def ref_out():
        out = reference(x_ref)
        return out.mean(2) if reduce_time else out

    reference.eval()
    port.eval()
    with torch.no_grad():
        assert_close(port(x_port), ref_out())
    reference.train()
    port.train()
    torch.manual_seed(seed)
    expected = ref_out()
    torch.manual_seed(seed)
    assert_close(port(x_port), expected)
    assert_same_state(reference, port)  # BatchNorm running statistics were updated identically
