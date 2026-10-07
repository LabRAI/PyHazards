"""SegFormer: configurations, parameter counts, shapes and input validation.

Numerical equivalence with transformers and the official checkpoints lives in
tests/oracle/test_segformer_oracle.py.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from pyhazards.models import build_model
from pyhazards.models.segformer import SEGFORMER_VARIANTS, SegFormer, adapt_first_conv_weight

# transformers SegformerForSemanticSegmentation, 3 channels, 150 classes (ADE20K). SegFormer paper
# Table 1: 3.8M, 13.7M, 27.5M, 47.3M, 64.1M, 84.7M.
ADE20K_PARAMS = {
    "b0": 3_752_694,
    "b1": 13_715_798,
    "b2": 27_461_974,
    "b3": 47_337_814,
    "b4": 64_108_374,
    "b5": 84_708_694,
}


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("variant", sorted(SEGFORMER_VARIANTS))
def test_parameter_counts_match_paper_table1(variant):
    with torch.device("meta"):
        model = SegFormer(in_channels=3, num_labels=150, variant=variant)
    assert _n_params(model) == ADE20K_PARAMS[variant]


@pytest.mark.parametrize(
    "in_channels, expected",
    [
        (40, 27_463_425),  # WSTS+ T=1, all features: reported 27.5M
        (120, 27_714_305),  # WSTS+ T=5 with WildfireSpreadTS static-feature deduplication: reported 27.7M
    ],
)
def test_wsts_plus_parameter_counts(in_channels, expected):
    with torch.device("meta"):
        model = build_model("segformer", task="segmentation", in_channels=in_channels)
    assert _n_params(model) == expected


def test_decoder_width_follows_variant():
    for variant, spec in SEGFORMER_VARIANTS.items():
        with torch.device("meta"):
            model = SegFormer(in_channels=3, num_labels=2, variant=variant)
        assert model.decode_head.linear_fuse.out_channels == spec["decoder_hidden_size"]
        assert model.decode_head.linear_fuse.out_channels == (256 if variant in ("b0", "b1") else 768)


@pytest.mark.parametrize("height, width", [(64, 64), (33, 47), (128, 96)])
def test_output_is_upsampled_to_input_size(height, width):
    model = build_model("segformer", task="segmentation", variant="b0", in_channels=6, out_channels=3).eval()
    x = torch.randn(2, 6, height, width)
    with torch.no_grad():
        out = model(x)
        raw = model.raw_logits(x)
    assert out.shape == (2, 3, height, width)
    assert raw.shape == (2, 3, -(-height // 4), -(-width // 4))
    torch.testing.assert_close(out, F.interpolate(raw, size=(height, width), mode="bilinear", align_corners=False))


def test_upsample_flag_returns_quarter_resolution():
    model = build_model("segformer", task="segmentation", variant="b0", in_channels=4, upsample=False).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 4, 64, 64)).shape == (1, 1, 16, 16)


def test_multiday_input_is_flattened_time_major():
    model = build_model("segformer", task="segmentation", variant="b0", in_channels=3, history=2).eval()
    x = torch.randn(1, 2, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(model(x), model(torch.cat([x[:, 0], x[:, 1]], dim=1)))


@pytest.mark.parametrize("shape", [(2, 3, 64), (2, 4, 64, 64), (1, 3, 28, 64), (1, 3, 64, 20), (1, 2, 2, 3, 4, 4)])
def test_bad_input_shapes_raise(shape):
    model = build_model("segformer", task="segmentation", variant="b0", in_channels=3)
    with pytest.raises(ValueError):
        model(torch.randn(*shape))


def test_smallest_valid_input():
    model = build_model("segformer", task="segmentation", variant="b0", in_channels=3).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 29, 29)).shape == (1, 1, 29, 29)


def test_bad_configuration_raises():
    with pytest.raises(ValueError):
        build_model("segformer", task="segmentation", in_channels=3, variant="b6")
    with pytest.raises(ValueError):
        build_model("segformer", task="classification", in_channels=3)
    with pytest.raises(ValueError):
        build_model("segformer", task="segmentation", in_channels=3, history=0)
    with pytest.raises(ValueError):
        build_model("segformer", task="segmentation", in_channels=3, variant="b0", encoder_weights="/nonexistent.bin")


def test_first_conv_adaptation_rule():
    weight = torch.randn(32, 3, 7, 7)
    assert adapt_first_conv_weight(weight, 3) is weight
    torch.testing.assert_close(adapt_first_conv_weight(weight, 1), weight.sum(1, keepdim=True))
    adapted = adapt_first_conv_weight(weight, 7)
    assert adapted.shape == (32, 7, 7, 7)
    for i in range(7):
        torch.testing.assert_close(adapted[:, i], weight[:, i % 3] * (3 / 7))


def test_load_hf_state_dict_accepts_all_transformers_layouts(tmp_path):
    source = SegFormer(in_channels=3, num_labels=2, variant="b0")
    full = source.state_dict()
    encoder = source.segformer.state_dict()
    classification = {"segformer." + key: value for key, value in encoder.items()}
    classification["classifier.weight"] = torch.zeros(1000, 256)
    classification["classifier.bias"] = torch.zeros(1000)

    for state, part in [(full, None), (encoder, "segformer"), (classification, "segformer")]:
        target = SegFormer(in_channels=3, num_labels=2, variant="b0")
        target.load_hf_state_dict(state)
        expected = source if part is None else source.segformer
        loaded = target if part is None else target.segformer
        for key, value in expected.state_dict().items():
            assert torch.equal(loaded.state_dict()[key], value), key

    with pytest.raises(RuntimeError):
        SegFormer(in_channels=3, num_labels=2, variant="b1").load_hf_state_dict(classification)

    # A local checkpoint passed as encoder_weights loads the encoder and adapts the stem.
    path = tmp_path / "pytorch_model.bin"
    torch.save(classification, path)
    model = build_model("segformer", task="segmentation", variant="b0", in_channels=5, encoder_weights=str(path))
    stem = "encoder.patch_embeddings.0.proj.weight"
    torch.testing.assert_close(model.segformer.state_dict()[stem], adapt_first_conv_weight(encoder[stem], 5))
    for key, value in encoder.items():
        if key != stem:
            assert torch.equal(model.segformer.state_dict()[key], value), key
