"""WildfireSpreadTS baselines checked against their reference implementations.

References (pinned in repos.yaml): VSainteuf/utae-paps for U-TAE and ConvLSTM (vendored unchanged
by WildfireSpreadTS), segmentation_models_pytorch 0.3.2 for the ResNet-18 U-Net, and the
WildfireSpreadTS sources for the logistic-regression baseline. Each check builds both models from
the same seed, requires identical parameter names and initial values, and compares outputs.
"""

from __future__ import annotations

import re

import torch

from oracle_utils import import_from, oracle_asset, oracle_package, oracle_repo
from pyhazards.models import build_model
from pyhazards.models.utae import UTAE

WSTS_UTAE_CONFIG = dict(
    encoder_widths=[64, 64, 64, 128],
    decoder_widths=[32, 32, 64, 128],
    out_conv=[32, 1],
    str_conv_k=4,
    str_conv_s=2,
    str_conv_p=1,
    agg_mode="att_group",
    encoder_norm="group",
    n_head=16,
    d_model=256,
    d_k=4,
    encoder=False,
    return_maps=False,
    pad_value=0,
    padding_mode="reflect",
)


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _assert_same_state(reference: torch.nn.Module, port: torch.nn.Module) -> None:
    ref_state, port_state = reference.state_dict(), port.state_dict()
    assert list(ref_state) == list(port_state)
    for key, value in ref_state.items():
        assert torch.equal(value, port_state[key]), key


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def _utae_paps(module: str):
    return import_from(oracle_repo("utae-paps"), f"src.backbones.{module}")


def _strip_imports(text: str) -> list[str]:
    return [line.rstrip() for line in text.splitlines() if line.strip() and not re.match(r"\s*(from|import) ", line)]


def test_wildfirespreadts_vendors_utae_paps_unchanged():
    wsts = oracle_repo("WildfireSpreadTS") / "src" / "models" / "utae_paps_models"
    paps = oracle_repo("utae-paps") / "src" / "backbones"
    for name in ["utae.py", "ltae.py", "convlstm.py", "positional_encoding.py"]:
        assert _strip_imports((wsts / name).read_text()) == _strip_imports((paps / name).read_text()), name


def test_logistic_regression_matches_wildfirespreadts_definition():
    source = (oracle_repo("WildfireSpreadTS") / "src" / "models" / "LogisticRegression.py").read_text()
    assert re.search(
        r"nn\.Conv2d\(\s*in_channels=n_channels,\s*out_channels=1,\s*kernel_size=3,\s*padding=1\s*\)", source
    )
    port = build_model("wildfirespreadts", task="segmentation", baseline="logistic_regression", in_channels=40, history=1)
    torch.manual_seed(0)
    reference = torch.nn.Conv2d(in_channels=40, out_channels=1, kernel_size=3, padding=1)
    port.conv.load_state_dict(reference.state_dict())
    assert _n_params(port) == 361
    x = torch.randn(2, 1, 40, 24, 24)
    _assert_close(port(x), reference(x[:, 0]))


def test_convlstm_matches_reference():
    convlstm = _utae_paps("convlstm")
    torch.manual_seed(0)
    reference = convlstm.ConvLSTM_Seg(num_classes=1, input_size=(32, 32), input_dim=40, hidden_dim=64, kernel_size=(3, 3))
    torch.manual_seed(0)
    port = build_model("wildfirespreadts", task="segmentation", baseline="convlstm", in_channels=40)
    _assert_same_state(reference, port)
    assert _n_params(port) == 240_449

    x = torch.randn(2, 5, 40, 32, 32)
    x[1, 0] = 0  # an all-zero frame takes the reference pad-mask branch
    with torch.no_grad():
        _assert_close(port(x), reference(x))


def test_utae_wildfirespreadts_config_matches_reference():
    utae = _utae_paps("utae")
    torch.manual_seed(0)
    reference = utae.UTAE(input_dim=40, **WSTS_UTAE_CONFIG)
    torch.manual_seed(0)
    port = build_model("wildfirespreadts", task="segmentation", baseline="utae", in_channels=40)
    _assert_same_state(reference, port)
    assert _n_params(port) == 1_099_011

    torch.manual_seed(1)
    x = torch.randn(2, 5, 40, 32, 32)
    doy = torch.tensor([[150.0, 151, 152, 153, 154], [200, 201, 202, 203, 204]])
    reference.eval()
    port.eval()
    with torch.no_grad():
        ref_out, ref_att = reference(x, batch_positions=doy, return_att=True)
        out, att = port(x, batch_positions=doy, return_att=True)
        _assert_close(out, ref_out)
        _assert_close(att, ref_att)

        padded = x.clone()
        padded[1, :2] = 0  # two leading padded dates in the second sample
        _assert_close(port(padded, batch_positions=doy), reference(padded, batch_positions=doy))

    reference.train()
    port.train()
    torch.manual_seed(2)
    ref_out = reference(x, batch_positions=doy)
    torch.manual_seed(2)
    out = port(x, batch_positions=doy)
    _assert_close(out, ref_out)


def test_utae_loads_official_pastis_weights():
    utae = _utae_paps("utae")
    checkpoint = oracle_asset("utae_pastis_weights") / "UATE_zenodo" / "Fold_1" / "model.pth.tar"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    # PASTIS semantic segmentation configuration from the released conf.json (N_params 1,087,260).
    port = UTAE(input_dim=10, out_conv=[32, 20])
    port.load_state_dict(state, strict=True)
    reference = utae.UTAE(input_dim=10, out_conv=[32, 20])
    reference.load_state_dict(state, strict=True)
    assert _n_params(port) == 1_087_260

    torch.manual_seed(3)
    x = torch.randn(2, 6, 10, 64, 64)
    days = torch.tensor([[10.0, 25, 40, 70, 100, 130], [5, 20, 50, 80, 110, 160]])
    reference.eval()
    port.eval()
    with torch.no_grad():
        _assert_close(port(x, batch_positions=days), reference(x, batch_positions=days))


def _smp():
    return oracle_package("segmentation_models_pytorch", "0.3.2")


def test_resnet18_unet_matches_smp():
    smp = _smp()
    torch.manual_seed(0)
    reference = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=40, classes=1)
    torch.manual_seed(0)
    port = build_model("wildfirespreadts", task="segmentation", baseline="resnet18_unet", in_channels=40, history=1)
    _assert_same_state(reference, port)
    assert _n_params(port) == 14_444_241

    torch.manual_seed(1)
    x = torch.randn(2, 40, 64, 64)
    reference.eval()
    port.eval()
    with torch.no_grad():
        _assert_close(port(x), reference(x))
        _assert_close(port(x[:, None]), reference(x))
    reference.train()
    port.train()
    _assert_close(port(x), reference(x))


def test_resnet18_unet_multiday_flattening_matches_smp():
    smp = _smp()
    reference = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=200, classes=1).eval()
    port = build_model("wildfirespreadts", task="segmentation", baseline="resnet18_unet", in_channels=40, history=5).eval()
    port.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(1, 5, 40, 32, 32)
    with torch.no_grad():
        _assert_close(port(x), reference(x.flatten(1, 2)))


def test_resnet18_unet_imagenet_stem_matches_smp():
    smp = _smp()
    reference = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=40, classes=1)
    port = build_model("resnet18_unet", task="segmentation", in_channels=40, encoder_weights="imagenet")
    ref_encoder, port_encoder = reference.encoder.state_dict(), port.encoder.state_dict()
    assert list(ref_encoder) == list(port_encoder)
    for key, value in ref_encoder.items():
        assert torch.equal(value, port_encoder[key]), key
