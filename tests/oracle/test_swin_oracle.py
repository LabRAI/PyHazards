"""Swin-Unet and ASUFM checked against their official implementations.

References (pinned in repos.yaml): HuCaoFighting/Swin-Unet (no license; used only here, never
vendored) with the Swin-T ImageNet checkpoint of microsoft/Swin-Transformer, and bronteee/fire-asufm
(Apache-2.0). Both need timm 0.4.12, einops, ml_collections and yacs (requirements-swin.txt), which
conflict with the segmentation_models_pytorch pin of requirements.txt, so the Oracle workflow runs
this file in its own job.

Each check builds both models from the same seed, requires identical parameter names and initial
values, and compares outputs in eval mode and in train mode (stochastic depth with a fixed seed).
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from oracle_utils import import_from, missing, oracle_asset, oracle_repo
from pyhazards.models import build_model
from pyhazards.models.swin_unet import SWIN_TINY_IMAGENET_SHA256, SWIN_TINY_IMAGENET_URL, SwinUnet

SWIN_UNET_YAML = "configs/swin_tiny_patch4_window7_224_lite.yaml"
SWIN_TINY_CHECKPOINT = "swin_tiny_patch4_window7_224.pth"
REQUIREMENTS = Path(__file__).parent / "requirements-swin.txt"


@pytest.fixture(autouse=True)
def _reference_stack():
    # Both references were written against timm 0.4.12 (pinned by fire-asufm and by
    # microsoft/Swin-Transformer); its DropPath draws the random numbers the ports reproduce.
    for line in REQUIREMENTS.read_text().splitlines():
        if "==" not in line or line.lstrip().startswith("#"):
            continue
        name, version = line.split("#")[0].strip().split("==")
        try:
            found = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing(f"{name} is not installed; pip install -r tests/oracle/requirements-swin.txt")
        if found != version:
            missing(f"needs {name}=={version}, found {found}")


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _assert_same_state(reference: torch.nn.Module, port: torch.nn.Module) -> None:
    ref_state, port_state = reference.state_dict(), port.state_dict()
    assert list(ref_state) == list(port_state)
    for key, value in ref_state.items():
        assert torch.equal(value, port_state[key]), key


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def _compare_forward(reference: torch.nn.Module, port: torch.nn.Module, x: torch.Tensor, seed: int) -> None:
    reference.eval()
    port.eval()
    with torch.no_grad():
        _assert_close(port(x), reference(x))
    reference.train()
    port.train()
    torch.manual_seed(seed)
    expected = reference(x)
    torch.manual_seed(seed)
    _assert_close(port(x), expected)


# --------------------------------------------------------------------------- Swin-Unet


def _swin_unet_reference(in_chans: int, num_classes: int, pretrained: Path | None = None):
    """Official ``SwinUnet`` built from the official yaml exactly as train.py does."""
    repo = oracle_repo("Swin-Unet")
    config_module = import_from(repo, "config")
    opts = ["MODEL.SWIN.IN_CHANS", in_chans]
    if pretrained is not None:
        opts += ["MODEL.PRETRAIN_CKPT", str(pretrained)]
    args = SimpleNamespace(
        cfg=str(repo / SWIN_UNET_YAML), opts=opts, batch_size=None, zip=False, cache_mode=None,
        resume=None, accumulation_steps=None, use_checkpoint=False, amp_opt_level=None, tag=None,
        eval=False, throughput=False,
    )
    config = config_module.get_config(args)
    vision_transformer = import_from(repo, "networks.vision_transformer")
    return vision_transformer.SwinUnet(config, img_size=config.DATA.IMG_SIZE, num_classes=num_classes), config


def test_swin_unet_official_config_and_unused_decoder_depths():
    repo = oracle_repo("Swin-Unet")
    raw = yaml.safe_load((repo / SWIN_UNET_YAML).read_text())["MODEL"]
    assert raw["DROP_PATH_RATE"] == 0.2
    assert raw["SWIN"]["DEPTHS"] == [2, 2, 2, 2]
    assert raw["SWIN"]["DECODER_DEPTHS"] == [2, 2, 2, 1]
    # vision_transformer.py never passes DECODER_DEPTHS, and SwinTransformerSys ignores its own
    # depths_decoder argument: the decoder mirrors the encoder depths.
    source = (repo / "networks" / "vision_transformer.py").read_text()
    assert "DECODER_DEPTHS" not in source
    sys_module = import_from(repo, "networks.swin_transformer_unet_skip_expand_decoder_sys")
    a = sys_module.SwinTransformerSys(in_chans=3, num_classes=9, depths_decoder=[1, 2, 2, 2])
    b = sys_module.SwinTransformerSys(in_chans=3, num_classes=9, depths_decoder=[2, 2, 2, 1])
    assert list(a.state_dict()) == list(b.state_dict())
    assert [len(layer.blocks) for layer in list(a.layers_up)[1:]] == [2, 2, 2]


def test_swin_unet_wsts_plus_config_matches_reference():
    torch.manual_seed(0)
    reference, _ = _swin_unet_reference(in_chans=40, num_classes=1)
    torch.manual_seed(0)
    port = build_model("swin_unet", task="segmentation", in_channels=40)
    _assert_same_state(reference.swin_unet, port)
    assert _n_params(port) == 27_224_964

    torch.manual_seed(1)
    x = torch.randn(2, 40, 224, 224)
    _compare_forward(reference, port, x, seed=2)


def test_swin_unet_synapse_config_matches_reference():
    torch.manual_seed(0)
    reference, _ = _swin_unet_reference(in_chans=3, num_classes=9)
    torch.manual_seed(0)
    port = build_model("swin_unet", task="segmentation", in_channels=3, out_channels=9)
    _assert_same_state(reference.swin_unet, port)
    assert _n_params(port) == 27_168_900

    torch.manual_seed(3)
    grey = torch.randn(1, 1, 224, 224)  # the official wrapper repeats one channel to RGB
    reference.eval()
    port.eval()
    with torch.no_grad():
        _assert_close(port(grey), reference(grey))


@pytest.mark.parametrize("in_chans", [40, 3])
def test_swin_unet_imagenet_initialisation_matches_load_from(in_chans):
    checkpoint = oracle_asset("swin_tiny_imagenet") / SWIN_TINY_CHECKPOINT
    torch.manual_seed(0)
    reference, config = _swin_unet_reference(in_chans=in_chans, num_classes=1, pretrained=checkpoint)
    reference.load_from(config)
    torch.manual_seed(0)
    port = build_model("swin_unet", task="segmentation", in_channels=in_chans, pretrained=str(checkpoint))
    _assert_same_state(reference.swin_unet, port)

    imagenet = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    state = port.state_dict()
    # Encoder stage i also initialises decoder stage 3 - i; the stem only matches for RGB input.
    assert torch.equal(state["layers_up.1.blocks.0.attn.qkv.weight"], imagenet["layers.2.blocks.0.attn.qkv.weight"])
    assert torch.equal(state["layers_up.3.blocks.1.mlp.fc2.weight"], imagenet["layers.0.blocks.1.mlp.fc2.weight"])
    assert torch.equal(state["norm.weight"], imagenet["norm.weight"])
    stem_loaded = torch.equal(state["patch_embed.proj.weight"], imagenet["patch_embed.proj.weight"]) if in_chans == 3 else False
    assert stem_loaded == (in_chans == 3)

    torch.manual_seed(4)
    x = torch.randn(1, in_chans, 224, 224)
    reference.eval()
    port.eval()
    with torch.no_grad():
        _assert_close(port(x), reference(x))


def test_swin_unet_imagenet_keyword_uses_pinned_checkpoint(tmp_path, monkeypatch):
    manifest = yaml.safe_load((Path(__file__).parent / "repos.yaml").read_text())["assets"]["swin_tiny_imagenet"]
    assert SWIN_TINY_IMAGENET_URL == manifest["url"]
    assert SWIN_TINY_IMAGENET_SHA256 == manifest["sha256"]

    checkpoint = oracle_asset("swin_tiny_imagenet") / SWIN_TINY_CHECKPOINT
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / SWIN_TINY_CHECKPOINT).symlink_to(checkpoint)
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    torch.manual_seed(0)
    by_name = build_model("swin_unet", task="segmentation", in_channels=7, pretrained="imagenet")
    torch.manual_seed(0)
    by_path = SwinUnet(in_chans=7, drop_path_rate=0.2)
    by_path.load_swin_pretrained(str(checkpoint))
    _assert_same_state(by_path, by_name)


# --------------------------------------------------------------------------- ASUFM


def _asufm_reference(config_name: str):
    repo = oracle_repo("fire-asufm")
    configs = import_from(repo, "configs.asufm")
    asufm = import_from(repo, "model.asufm.asufm")
    config = getattr(configs, config_name)()
    return asufm.ASUFM(config=config, num_classes=1), config


@pytest.mark.parametrize(
    "config_name, in_chans, expected",
    [("get_asfum_6_configs", 6, 35_047_840), ("get_asufm_12_configs", 12, 35_057_056)],
)
def test_asufm_matches_reference(config_name, in_chans, expected):
    torch.manual_seed(0)
    reference, config = _asufm_reference(config_name)
    assert (config.in_chans, config.image_size, config.window_size, config.focal) == (in_chans, 64, 8, True)
    assert (config.mode, config.spatial_attention, config.skip_num) == ("swin", "1", 3)
    torch.manual_seed(0)
    port = build_model("asufm", task="segmentation", in_channels=in_chans)
    _assert_same_state(reference, port)
    assert _n_params(port) == expected

    torch.manual_seed(1)
    x = torch.randn(3, in_chans, 64, 64)
    _compare_forward(reference, port, x, seed=2)
    assert port(x).shape == (3, 1, 64, 64)


def test_asufm_gradient_checkpointing_does_not_change_outputs():
    torch.manual_seed(0)
    reference, _ = _asufm_reference("get_asfum_6_configs")  # use_checkpoint=True in the config
    port = build_model("asufm", task="segmentation", in_channels=6, use_checkpoint=True)
    port.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(2, 6, 64, 64, requires_grad=True)
    _compare_forward(reference, port, x, seed=5)

    port.train()
    torch.manual_seed(6)
    port(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_asufm_without_focal_modulation_matches_reference():
    torch.manual_seed(0)
    repo = oracle_repo("fire-asufm")
    configs = import_from(repo, "configs.asufm")
    asufm = import_from(repo, "model.asufm.asufm")
    config = configs.get_swin_unet_attention_configs()  # the same network with focal=False
    reference = asufm.ASUFM(config=config, num_classes=1)
    torch.manual_seed(0)
    port = build_model("asufm", task="segmentation", in_channels=6, focal=False)
    _assert_same_state(reference, port)
    x = torch.randn(2, 6, 64, 64)
    _compare_forward(reference, port, x, seed=7)
