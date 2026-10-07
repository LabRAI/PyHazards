"""SegFormer checked against Hugging Face transformers (pinned in requirements.txt).

transformers' ``SegformerForSemanticSegmentation`` / ``SegformerModel`` are the reference: same
parameter names, same seeded initialisation, same outputs in eval and train mode. The official
NVIDIA checkpoints ``nvidia/mit-b0`` (ImageNet-1k encoder) and
``nvidia/segformer-b0-finetuned-ade-512-512`` (pinned assets in repos.yaml) must load with
``strict=True`` and give the same outputs as transformers. The first-conv adaptation used with
pretrained weights is compared with segmentation_models_pytorch's ``patch_first_conv``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from oracle_utils import oracle_asset, oracle_package
from pyhazards.models import build_model
from pyhazards.models.segformer import (
    MIT_IMAGENET_CHECKPOINTS,
    SEGFORMER_VARIANTS,
    SegFormer,
    SegformerConfig,
    SegformerModel,
    mit_imagenet_url,
)

TRANSFORMERS_VERSION = "4.57.6"
MANIFEST = yaml.safe_load((Path(__file__).parent / "repos.yaml").read_text(encoding="utf-8"))


def _transformers():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return oracle_package("transformers", TRANSFORMERS_VERSION)


def _hf_config(variant: str, **kwargs):
    spec = SEGFORMER_VARIANTS[variant]
    return _transformers().SegformerConfig(
        hidden_sizes=list(spec["hidden_sizes"]),
        depths=list(spec["depths"]),
        decoder_hidden_size=spec["decoder_hidden_size"],
        **kwargs,
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


def _mit_b0_state():
    path = oracle_asset("segformer_mit_b0") / "pytorch_model.bin"
    return torch.load(path, map_location="cpu", weights_only=True), path


@pytest.mark.parametrize(
    "variant, expected",
    [("b0", 3_752_694), ("b1", 13_715_798), ("b2", 27_461_974)],
)
def test_parameter_counts_match_transformers(variant, expected):
    # SegFormer paper Table 1 (ADE20K, 150 classes): B0 3.8M, B1 13.7M, B2 27.5M.
    with torch.device("meta"):
        reference = _transformers().SegformerForSemanticSegmentation(_hf_config(variant, num_labels=150))
        port = SegFormer(in_channels=3, num_labels=150, variant=variant)
    assert _n_params(reference) == _n_params(port) == expected


@pytest.mark.parametrize("variant, in_channels, num_labels", [("b0", 3, 150), ("b2", 40, 1)])
def test_names_initialisation_and_forward_match_transformers(variant, in_channels, num_labels):
    hf = _transformers()
    torch.manual_seed(0)
    reference = hf.SegformerForSemanticSegmentation(
        _hf_config(variant, num_channels=in_channels, num_labels=num_labels)
    )
    torch.manual_seed(0)
    port = SegFormer(in_channels=in_channels, num_labels=num_labels, variant=variant)
    _assert_same_state(reference, port)

    torch.manual_seed(1)
    x = torch.randn(2, in_channels, 64, 96)
    reference.eval()
    port.eval()
    with torch.no_grad():
        expected = reference(pixel_values=x).logits
        _assert_close(port.raw_logits(x), expected)
        upsampled = F.interpolate(expected, size=x.shape[-2:], mode="bilinear", align_corners=False)
        _assert_close(port(x), upsampled)

    # Train mode: stochastic depth (0.1) and decoder dropout (0.1) draw the same random numbers.
    reference.train()
    port.train()
    torch.manual_seed(2)
    expected = reference(pixel_values=x).logits
    torch.manual_seed(2)
    _assert_close(port.raw_logits(x), expected)
    for name, buffer in reference.decode_head.batch_norm.named_buffers():
        assert torch.equal(buffer, getattr(port.decode_head.batch_norm, name)), name


def test_wildfire_configuration_matches_transformers():
    # WSTS+ (Lahrichi et al., WACV 2026): SegFormer-B2, 40 input channels (one day, all features), one output.
    hf = _transformers()
    torch.manual_seed(0)
    reference = hf.SegformerForSemanticSegmentation(_hf_config("b2", num_channels=40, num_labels=1))
    torch.manual_seed(0)
    port = build_model("segformer", task="segmentation", in_channels=40)
    _assert_same_state(reference, port)
    assert _n_params(port) == 27_463_425


def test_encoder_only_model_matches_transformers():
    hf = _transformers()
    torch.manual_seed(0)
    reference = hf.SegformerModel(_hf_config("b0", num_channels=5))
    torch.manual_seed(0)
    port = SegformerModel(SegformerConfig.from_variant("b0", num_channels=5))
    _assert_same_state(reference, port)

    x = torch.randn(2, 5, 64, 64)
    reference.eval()
    port.eval()
    with torch.no_grad():
        expected = reference(pixel_values=x, output_hidden_states=True).hidden_states
        actual = port(x)
    assert len(actual) == len(expected) == 4
    for a, e in zip(actual, expected):
        _assert_close(a, e)

    # A SegformerModel state dict loads into SegFormer.segformer.
    full = SegFormer(in_channels=5, num_labels=1, variant="b0")
    assert str(full.load_hf_state_dict(reference.state_dict())) == "<All keys matched successfully>"
    _assert_same_state(reference, full.segformer)


def test_official_mit_b0_imagenet_weights_load_strict():
    hf = _transformers()
    state, _ = _mit_b0_state()
    reference = hf.SegformerForImageClassification(_hf_config("b0", num_labels=1000))
    reference.load_state_dict(state, strict=True)
    port = SegFormer(in_channels=3, num_labels=1, variant="b0")
    port.load_hf_state_dict(state, strict=True)  # drops the "segformer." prefix and the classifier
    _assert_same_state(reference.segformer, port.segformer)

    torch.manual_seed(3)
    x = torch.randn(2, 3, 128, 128)
    reference.eval()
    port.eval()
    with torch.no_grad():
        out = reference(pixel_values=x, output_hidden_states=True)
        features = port.segformer(x)
        for a, e in zip(features, out.hidden_states):
            _assert_close(a, e)
        pooled = features[-1].flatten(2).mean(-1)
        _assert_close(F.linear(pooled, state["classifier.weight"], state["classifier.bias"]), out.logits)


def test_official_ade20k_segformer_b0_loads_strict_and_matches():
    hf = _transformers()
    safetensors = pytest.importorskip("safetensors.torch")
    state = safetensors.load_file(str(oracle_asset("segformer_b0_ade") / "model.safetensors"))
    reference = hf.SegformerForSemanticSegmentation(_hf_config("b0", num_labels=150))
    reference.load_state_dict(state, strict=True)
    port = SegFormer(in_channels=3, num_labels=150, variant="b0")
    port.load_state_dict(state, strict=True)
    _assert_same_state(reference, port)

    torch.manual_seed(4)
    x = torch.randn(2, 3, 128, 160)
    reference.eval()
    port.eval()
    with torch.no_grad():
        outputs = reference(pixel_values=x)
        _assert_close(port.raw_logits(x), outputs.logits)
        logits = port(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        processor = hf.SegformerImageProcessor()
    maps = processor.post_process_semantic_segmentation(outputs, target_sizes=[tuple(x.shape[-2:])] * 2)
    for i, semantic_map in enumerate(maps):
        assert torch.equal(logits[i].argmax(0), semantic_map)


def test_pretrained_stem_adaptation_matches_smp_patch_first_conv():
    hf = _transformers()
    smp = oracle_package("segmentation_models_pytorch", "0.3.2")
    from segmentation_models_pytorch.encoders._utils import patch_first_conv

    state, path = _mit_b0_state()
    for in_channels in (1, 7, 40):
        reference = hf.SegformerForImageClassification(_hf_config("b0", num_labels=1000))
        reference.load_state_dict(state, strict=True)
        patch_first_conv(reference, new_in_channels=in_channels, pretrained=True)
        port = build_model("segformer", task="segmentation", variant="b0", in_channels=in_channels, encoder_weights=str(path))
        _assert_same_state(reference.segformer, port.segformer)
        x = torch.randn(1, in_channels, 64, 64)
        reference.eval()
        port.eval()
        with torch.no_grad():
            expected = reference(pixel_values=x, output_hidden_states=True).hidden_states
            for a, e in zip(port.segformer(x), expected):
                _assert_close(a, e)
    assert smp.__version__ == "0.3.2"


def test_imagenet_download_is_the_pinned_asset():
    asset = MANIFEST["assets"]["segformer_mit_b0"]
    revision, sha256 = MIT_IMAGENET_CHECKPOINTS["b0"]
    assert mit_imagenet_url("b0") == asset["url"]
    assert sha256 == asset["sha256"]
    for variant, (revision, sha256) in MIT_IMAGENET_CHECKPOINTS.items():
        assert len(revision) == 40 and len(sha256) == 64, variant

    # encoder_weights="imagenet" downloads through torch.hub (TORCH_HOME), checking the sha256 prefix.
    state, _ = _mit_b0_state()
    port = build_model("segformer", task="segmentation", variant="b0", in_channels=3, encoder_weights="imagenet")
    for key, value in port.segformer.state_dict().items():
        assert torch.equal(value, state["segformer." + key]), key
