"""Attention U-Net checked against MONAI 1.3.2 and TS-SatFire's modified copy.

References: ``monai==1.3.2`` (pinned in requirements.txt; the version in TS-SatFire's
environment.yml) and ``spatial_models/attentionunet.py`` from zhaoyutim/TS-SatFire (pinned in
repos.yaml; no LICENSE, so it is only imported here as an oracle). Each check builds both models
from the same seed, requires identical parameter names and initial values, then compares outputs
in eval mode (with randomised BatchNorm statistics) and train mode.

Spatial sizes are reduced for CPU speed (the network is fully convolutional; TS-SatFire uses
256x256 tiles): 32x32 with T=2 for the full-width 3D prediction model, 64x64 for the 2D model.
"""

from __future__ import annotations

import torch

from oracle_utils import import_from, oracle_package, oracle_repo
from pyhazards.models import build_model
from pyhazards.models.attention_unet import AttentionUnet, TemporalAttentionUnet

TS_SATFIRE_CHANNELS = (64, 128, 256, 512, 1024)
SMALL_CHANNELS = (8, 16, 32, 64, 128)


def _monai():
    return oracle_package("monai", "1.3.2")


def _monai_attention_unet():
    _monai()
    from monai.networks.nets import AttentionUnet as MonaiAttentionUnet

    return MonaiAttentionUnet


def _ts_satfire_attention_unet():
    _monai()  # the TS-SatFire file imports MONAI blocks
    module = import_from(oracle_repo("TS-SatFire"), "spatial_models.attentionunet")
    return module.AttentionUnet


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _assert_same_state(reference: torch.nn.Module, port: torch.nn.Module) -> None:
    ref_state, port_state = reference.state_dict(), port.state_dict()
    assert list(ref_state) == list(port_state)
    for key, value in ref_state.items():
        assert torch.equal(value, port_state[key]), key


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def _randomise(reference: torch.nn.Module, port: torch.nn.Module, seed: int) -> None:
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


def test_ts_satfire_prediction_model_matches_reference():
    TSAttentionUnet = _ts_satfire_attention_unet()
    # run_spatial_temp_model_pred.py: 43 input channels after the land-cover one-hot, 2 classes.
    torch.manual_seed(0)
    reference = TSAttentionUnet(
        spatial_dims=3, in_channels=43, out_channels=2, channels=TS_SATFIRE_CHANNELS, strides=(1, 2, 2)
    )
    torch.manual_seed(0)
    port = build_model("attention_unet", task="segmentation", in_channels=43)
    assert isinstance(port, TemporalAttentionUnet)
    _assert_same_state(reference, port)
    assert _n_params(port) == _n_params(reference) == 94_555_506

    _randomise(reference, port, seed=1)
    torch.manual_seed(2)
    x = torch.randn(2, 2, 43, 32, 32)  # (batch, time, channels, H, W)
    x_ref = x.transpose(1, 2).contiguous()  # TS-SatFire feeds (batch, channels, time, H, W)
    reference.eval()
    port.eval()
    with torch.no_grad():
        _assert_close(port(x), reference(x_ref).mean(2))
        port.time_reduction = "none"  # per-date maps, as in the AF/BA scripts
        _assert_close(port(x), reference(x_ref))
        port.time_reduction = "mean"

    reference.train()
    port.train()
    _assert_close(port(x), reference(x_ref).mean(2))
    _assert_same_state(reference, port)  # BatchNorm running statistics were updated identically


def test_ts_satfire_shared_strides_equal_monai_per_level_strides():
    TSAttentionUnet = _ts_satfire_attention_unet()
    MonaiAttentionUnet = _monai_attention_unet()
    torch.manual_seed(0)
    ts_model = TSAttentionUnet(
        spatial_dims=3, in_channels=5, out_channels=2, channels=SMALL_CHANNELS, strides=(1, 2, 2)
    )
    torch.manual_seed(0)
    monai_model = MonaiAttentionUnet(
        spatial_dims=3, in_channels=5, out_channels=2, channels=SMALL_CHANNELS, strides=((1, 2, 2),) * 4
    )
    torch.manual_seed(0)
    port_shared = AttentionUnet(
        spatial_dims=3, in_channels=5, out_channels=2, channels=SMALL_CHANNELS, strides=(1, 2, 2), stride_mode="shared"
    )
    torch.manual_seed(0)
    port_per_level = AttentionUnet(
        spatial_dims=3, in_channels=5, out_channels=2, channels=SMALL_CHANNELS, strides=((1, 2, 2),) * 4
    )
    for model in (monai_model, port_shared, port_per_level):
        _assert_same_state(ts_model, model)

    torch.manual_seed(1)
    x = torch.randn(2, 5, 3, 32, 32)
    for model in (ts_model, monai_model, port_shared, port_per_level):
        model.eval()
    with torch.no_grad():
        expected = ts_model(x)
        assert expected.shape == (2, 2, 3, 32, 32)  # time is never down-sampled
        _assert_close(monai_model(x), expected)
        _assert_close(port_shared(x), expected)
        _assert_close(port_per_level(x), expected)


def test_ts_satfire_2d_config_matches_monai():
    MonaiAttentionUnet = _monai_attention_unet()
    # run_spatial_model.py: stock MONAI, 2D, strides (2, 2, 2, 2); FireDataset default n_channel=8.
    torch.manual_seed(0)
    reference = MonaiAttentionUnet(
        spatial_dims=2, in_channels=8, out_channels=2, channels=TS_SATFIRE_CHANNELS, strides=(2, 2, 2, 2)
    )
    torch.manual_seed(0)
    port = build_model("attention_unet", task="segmentation", in_channels=8, spatial_dims=2)
    assert type(port) is AttentionUnet
    _assert_same_state(reference, port)
    assert _n_params(port) == 31_743_282

    _randomise(reference, port, seed=3)
    torch.manual_seed(4)
    x = torch.randn(2, 8, 64, 64)
    reference.eval()
    port.eval()
    with torch.no_grad():
        _assert_close(port(x), reference(x))
    reference.train()
    port.train()
    _assert_close(port(x), reference(x))


def test_non_default_configs_match_monai():
    """Per-level non-uniform strides, larger kernels and dropout, in 1D, 2D and 3D."""
    MonaiAttentionUnet = _monai_attention_unet()
    configs = [
        dict(spatial_dims=1, in_channels=3, out_channels=1, channels=(4, 8, 16), strides=(2, 2), kernel_size=5),
        dict(
            spatial_dims=2,
            in_channels=3,
            out_channels=3,
            channels=(4, 8, 16, 32),
            strides=((2, 1), 2, (1, 2)),
            kernel_size=5,
            up_kernel_size=3,
            dropout=0.2,
        ),
        dict(
            spatial_dims=3,
            in_channels=2,
            out_channels=1,
            channels=(4, 8, 16),
            strides=((1, 2, 2), 2),
            kernel_size=(1, 3, 3),
            up_kernel_size=(3, 3, 3),
            dropout=0.1,
        ),
    ]
    inputs = [(2, 3, 16), (2, 3, 16, 16), (2, 2, 4, 16, 16)]
    for config, shape in zip(configs, inputs):
        torch.manual_seed(0)
        reference = MonaiAttentionUnet(**config)
        torch.manual_seed(0)
        port = AttentionUnet(**config)
        _assert_same_state(reference, port)
        _randomise(reference, port, seed=5)

        x = torch.randn(*shape)
        reference.eval()
        port.eval()
        with torch.no_grad():
            _assert_close(port(x), reference(x))
        reference.train()
        port.train()
        torch.manual_seed(6)
        expected = reference(x)
        torch.manual_seed(6)
        _assert_close(port(x), expected)  # same dropout masks: same module order


def test_ts_satfire_copy_ignores_kernel_size():
    """The TS-SatFire copy predates MONAI 1.3.1 and builds 3x3 blocks whatever ``kernel_size`` is.

    The port follows MONAI 1.3.2 and honours ``kernel_size``; at the default 3 used by TS-SatFire
    both are identical (checked above).
    """
    TSAttentionUnet = _ts_satfire_attention_unet()
    MonaiAttentionUnet = _monai_attention_unet()
    config = dict(spatial_dims=2, in_channels=3, out_channels=1, channels=(4, 8, 16), strides=2, kernel_size=5)
    ts_model = TSAttentionUnet(**config)
    monai_model = MonaiAttentionUnet(**{**config, "strides": (2, 2)})
    port = AttentionUnet(**{**config, "stride_mode": "shared"})
    key = "model.0.conv.0.conv.weight"
    assert ts_model.state_dict()[key].shape[-1] == 3
    assert monai_model.state_dict()[key].shape[-1] == 5
    assert port.state_dict()[key].shape[-1] == 5
