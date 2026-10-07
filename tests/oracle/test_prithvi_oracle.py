"""Prithvi-EO-2.0 ports checked against TerraTorch 0.99.8 and the official ``prithvi_mae.py``.

References (tests/oracle/requirements-prithvi.txt and repos.yaml): TerraTorch 0.99.8 with smp 0.4.0 and
timm 1.0.15, the stack the BurnScars checkpoint was trained with; ``prithvi_mae.py`` and
``burn_scars_config.yaml`` from the Hugging Face repositories at pinned revisions. The routine checks
use randomly initialised official modules (same seed, same parameter names and values, same outputs).
The two tests at the end load the 1.3 GB official checkpoints and are skipped unless those ``large``
assets have been fetched.

Run in the Prithvi oracle environment (Python >= 3.12), not the generic oracle one.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
import yaml

from oracle_utils import import_from, oracle_asset, oracle_large_asset, oracle_package
from pyhazards.models import build_model
from pyhazards.models.prithvi import (
    PRITHVI_BANDS,
    PRITHVI_EO_V2_CONFIGS,
    PRITHVI_EO_V2_MEAN,
    PRITHVI_EO_V2_STD,
    PrithviMAE,
    get_3d_sincos_pos_embed,
)
from pyhazards.models.prithvi_burnscars import BURN_SCARS_CLASSES, BURN_SCARS_MEAN, BURN_SCARS_STD

REQ = "requirements-prithvi.txt"
TINY = dict(embed_dim=64, depth=4, num_heads=2)
TINY_RECIPE = dict(select_indices=[0, 1, 2, 3], decoder_channels=[32, 16, 8, 4])


@pytest.fixture(scope="module")
def terratorch():
    oracle_package("segmentation_models_pytorch", "0.4.0", REQ)
    oracle_package("timm", "1.0.15", REQ)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module = oracle_package("terratorch", "0.99.8", REQ)
        import terratorch.models  # noqa: F401  (registers backbones, necks and decoders)
    return module


@pytest.fixture(scope="module")
def burn_scars_config():
    path = oracle_asset("prithvi_burnscars_config") / "burn_scars_config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def reference_mae():
    oracle_package("timm", "1.0.15", REQ)
    return import_from(oracle_asset("prithvi_eo2_tl_model_code"), "prithvi_mae")


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _assert_same_state(reference: torch.nn.Module, port: torch.nn.Module) -> None:
    ref_state, port_state = reference.state_dict(), port.state_dict()
    assert list(ref_state) == list(port_state)
    for key, value in ref_state.items():
        assert torch.equal(value, port_state[key]), key


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def _official_segmentation(model_args: dict, seed: int, **overrides):
    """TerraTorch EncoderDecoderFactory model, without downloading the backbone weights."""
    from terratorch.models import EncoderDecoderFactory

    args = dict(model_args, backbone_pretrained=False, **overrides)
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return EncoderDecoderFactory().build_model(task="segmentation", **args)


def _coords(batch: int, frames: int):
    years = torch.tensor([[2018.0 + i for i in range(frames)]] * batch)
    days = torch.tensor([[26.0 + 80 * i for i in range(frames)]] * batch) + torch.arange(batch)[:, None]
    temporal = torch.stack([years, days], dim=-1)
    location = torch.tensor([[25.0 + b, -104.5 + 2 * b] for b in range(batch)])
    return temporal, location


def test_burn_scars_recipe_matches_official_config(terratorch, burn_scars_config):
    from terratorch.models.backbones import prithvi_vit

    args = burn_scars_config["model"]["init_args"]["model_args"]
    assert args["backbone"] == "prithvi_eo_v2_300"
    assert args["backbone_bands"] == list(PRITHVI_BANDS)
    assert args["necks"] == [
        {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
        {"name": "ReshapeTokensToImage"},
        {"name": "LearnedInterpolateToPyramidal"},
    ]
    assert args["decoder"] == "UNetDecoder"
    assert args["decoder_channels"] == [512, 256, 128, 64]
    assert args["num_classes"] == 2
    assert burn_scars_config["model"]["init_args"]["class_names"] == list(BURN_SCARS_CLASSES)
    data = burn_scars_config["data"]["init_args"]
    assert data["output_bands"] == list(PRITHVI_BANDS)
    assert tuple(data["means"]) == BURN_SCARS_MEAN
    assert tuple(data["stds"]) == BURN_SCARS_STD

    assert tuple(prithvi_vit.PRITHVI_V2_MEAN) == PRITHVI_EO_V2_MEAN
    assert tuple(prithvi_vit.PRITHVI_V2_STD) == PRITHVI_EO_V2_STD
    for variant, names in {"300m": ["prithvi_eo_v2_300", "prithvi_eo_v2_300_tl"], "600m": ["prithvi_eo_v2_600_tl"]}.items():
        ours = PRITHVI_EO_V2_CONFIGS[variant]
        for name in names:
            cfg = prithvi_vit.prithvi_cfgs[name]
            assert (cfg["embed_dim"], cfg["depth"], cfg["num_heads"]) == (ours["embed_dim"], ours["depth"], ours["num_heads"])
            assert tuple(cfg["patch_size"]) == ours["patch_size"]
            assert (cfg["img_size"], cfg["in_chans"], cfg["mlp_ratio"]) == (224, 6, 4)
            if name.endswith("_tl"):
                assert cfg["coords_encoding"] == ["time", "location"] and cfg["coords_scale_learn"] is True


def test_burnscars_initialisation_and_forward_match_terratorch(terratorch, burn_scars_config):
    model_args = burn_scars_config["model"]["init_args"]["model_args"]
    reference = _official_segmentation(model_args, seed=0)
    torch.manual_seed(0)
    port = build_model("prithvi_burnscars", task="segmentation")
    _assert_same_state(reference, port)
    assert _n_params(port) == _n_params(reference) == 324_204_674

    torch.manual_seed(1)
    x = torch.randn(2, 6, 64, 64)
    padded = torch.randn(1, 6, 120, 120)  # reflect-padded to 128 x 128 and cropped back
    reference.eval()
    port.eval()
    with torch.no_grad():
        _assert_close(port(x), reference(x).output)
        _assert_close(port(x[:, :, None]), reference(x).output)
        _assert_close(port(padded), reference(padded).output)
    reference.train()
    port.train()
    _assert_close(port(x), reference(x).output)
    for key, value in reference.state_dict().items():  # BatchNorm running statistics
        assert torch.equal(value, port.state_dict()[key]), key


def test_tiny_burnscars_configuration_matches_terratorch(terratorch, burn_scars_config):
    model_args = dict(burn_scars_config["model"]["init_args"]["model_args"])
    model_args.update(
        necks=[{"name": "SelectIndices", "indices": [0, 1, 2, 3]}, {"name": "ReshapeTokensToImage"},
               {"name": "LearnedInterpolateToPyramidal"}],
        decoder_channels=[32, 16, 8, 4],
    )
    reference = _official_segmentation(
        model_args, seed=3, backbone_embed_dim=64, backbone_depth=4, backbone_num_heads=2
    ).eval()
    torch.manual_seed(3)
    port = build_model("prithvi_burnscars", task="segmentation", **TINY, **TINY_RECIPE).eval()
    _assert_same_state(reference, port)
    x = torch.randn(3, 6, 96, 96)
    with torch.no_grad():
        _assert_close(port(x), reference(x).output)


def test_tl_segmentation_matches_terratorch(terratorch, burn_scars_config):
    model_args = dict(burn_scars_config["model"]["init_args"]["model_args"], backbone="prithvi_eo_v2_300_tl")
    reference = _official_segmentation(model_args, seed=0)
    torch.manual_seed(0)
    port = build_model("prithvi_eo_2_tl", task="segmentation")
    _assert_same_state(reference, port)
    assert _n_params(port) == _n_params(reference) == 324_204_676

    torch.manual_seed(1)
    x = torch.randn(2, 6, 64, 64)
    temporal, location = _coords(2, 1)
    reference.eval()
    port.eval()
    with torch.no_grad():
        expected = reference(x, temporal_coords=temporal, location_coords=location).output
        _assert_close(port(x, temporal_coords=temporal, location_coords=location), expected)
        _assert_close(port(x), reference(x).output)
        assert not torch.allclose(port(x), expected)  # the coordinates are used


def test_multitemporal_tl_segmentation_matches_terratorch_modules(terratorch, burn_scars_config):
    # Official multi-temporal recipe: backbone_num_frames = T and ReshapeTokensToImage(effective_time_dim=T).
    # TerraTorch 0.99.8's PixelWiseModel cannot pad 5-D input, so the reference modules are chained by hand
    # on an input that needs no padding.
    frames = 3
    model_args = dict(burn_scars_config["model"]["init_args"]["model_args"], backbone="prithvi_eo_v2_300_tl")
    model_args.update(
        necks=[{"name": "SelectIndices", "indices": [0, 1, 2, 3]},
               {"name": "ReshapeTokensToImage", "effective_time_dim": frames},
               {"name": "LearnedInterpolateToPyramidal"}],
        decoder_channels=[32, 16, 8, 4],
    )
    reference = _official_segmentation(
        model_args, seed=5, backbone_embed_dim=64, backbone_depth=4, backbone_num_heads=2, backbone_num_frames=frames
    ).eval()
    torch.manual_seed(5)
    port = build_model("prithvi_eo_2_tl", task="segmentation", num_frames=frames, **TINY, **TINY_RECIPE).eval()
    _assert_same_state(reference, port)

    x = torch.randn(2, 6, frames, 64, 64)
    temporal, location = _coords(2, frames)
    with torch.no_grad():
        features = reference.encoder(x, temporal_coords=temporal, location_coords=location)
        features = reference.neck(features)
        expected = reference.head(reference.decoder([f.clone() for f in features]))
        expected = torch.nn.functional.interpolate(expected, size=(64, 64), mode="bilinear")
        _assert_close(port(x, temporal_coords=temporal, location_coords=location), expected)


def test_600m_tl_encoder_matches_terratorch(terratorch):
    from terratorch.registry import BACKBONE_REGISTRY

    torch.manual_seed(7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = BACKBONE_REGISTRY.build("prithvi_eo_v2_600_tl", pretrained=False, depth=2).eval()
    torch.manual_seed(7)
    port = build_model(
        "prithvi_eo_2_tl", task="segmentation", variant="600m", depth=2, select_indices=[0, 1, 1, 1]
    ).encoder.eval()
    _assert_same_state(reference, port)
    x = torch.randn(1, 6, 56, 56)
    temporal, location = _coords(1, 1)
    with torch.no_grad():
        for actual, expected in zip(
            port.forward_features(x, temporal, location), reference(x, temporal_coords=temporal, location_coords=location)
        ):
            _assert_close(actual, expected)


@pytest.mark.parametrize("coords", [True, False])
def test_mae_matches_official_prithvi_mae(reference_mae, coords):
    config = dict(
        img_size=64, num_frames=4, patch_size=[1, 16, 16], in_chans=6, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4,
        coords_encoding=["time", "location"], coords_scale_learn=True,
    )
    torch.manual_seed(0)
    reference = reference_mae.PrithviMAE(**config)
    torch.manual_seed(0)
    port = PrithviMAE(**config)
    _assert_same_state(reference, port)
    assert _n_params(port) == _n_params(reference) == 330_419_716

    x = torch.randn(2, 6, 4, 64, 64)
    temporal, location = _coords(2, 4) if coords else (None, None)
    reference.eval()
    port.eval()
    with torch.no_grad():
        torch.manual_seed(1)
        ref_loss, ref_pred, ref_mask = reference(x, temporal, location)
        torch.manual_seed(1)
        loss, pred, mask = port(x, temporal, location)
        _assert_close(loss, ref_loss)
        _assert_close(pred, ref_pred)
        assert torch.equal(mask, ref_mask)
        for actual, expected in zip(port.forward_features(x, temporal, location),
                                    reference.forward_features(x, temporal, location)):
            _assert_close(actual, expected)
        # One frame instead of four: the time axis of the position table is recomputed.
        _assert_close(port.forward_features(x[:, :, :1])[-1], reference.forward_features(x[:, :, :1])[-1])
    assert torch.equal(port.patchify(x), reference.patchify(x))
    assert torch.equal(port.unpatchify(port.patchify(x), image_size=(64, 64)), x)


def test_burnscars_official_weights_match_terratorch(terratorch, burn_scars_config):
    rasterio = pytest.importorskip("rasterio")
    checkpoint = oracle_large_asset("prithvi_eo2_300m_burnscars_weights") / "Prithvi_EO_V2_300M_BurnScars.pt"
    chip = oracle_asset("prithvi_burnscars_example") / "subsetted_512x512_HLS.S30.T10SEH.2018190.v1.4_merged.tif"

    reference = _official_segmentation(burn_scars_config["model"]["init_args"]["model_args"], seed=0)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    reference.load_state_dict({k[len("model."):]: v for k, v in state.items()}, strict=True)
    port = build_model("prithvi_burnscars", task="segmentation", weights_path=str(checkpoint))

    # Official inference.py preprocessing: reflectance in 0-1 (divide by 10000 if needed), then the
    # datamodule's per-band standardisation.
    with rasterio.open(chip) as src:
        image = src.read().astype("float32")
    if image.mean() > 1:
        image = image / 10000
    mean = np.array(BURN_SCARS_MEAN, dtype="float32")[:, None, None]
    std = np.array(BURN_SCARS_STD, dtype="float32")[:, None, None]
    x = torch.from_numpy((image - mean) / std)[None]
    reference.eval()
    port.eval()
    with torch.no_grad():
        expected = reference(x).output
        actual = port(x)
    assert actual.shape == (1, 2, 512, 512)
    _assert_close(actual, expected)
    assert torch.equal(actual.argmax(1), expected.argmax(1))
    burned = actual.argmax(1).float().mean().item()
    assert 0.79 < burned < 0.80, burned  # 79.5 % of this chip is predicted burned


def test_tl_official_weights_match_reference(terratorch, reference_mae):
    checkpoint = oracle_large_asset("prithvi_eo2_300m_tl_weights") / "Prithvi_EO_V2_300M_TL.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)

    # The position tables in the checkpoint are the fixed sin/cos tables (up to float32 rounding, max
    # 4.6e-7), so the segmentation encoder rebuilds its own for its frame count, as TerraTorch does.
    for key, dim in [("encoder.pos_embed", 1024), ("decoder.decoder_pos_embed", 512)]:
        table = torch.from_numpy(get_3d_sincos_pos_embed(dim, (4, 14, 14), add_cls_token=True)).float()[None]
        torch.testing.assert_close(state[key], table, rtol=0, atol=1e-6)

    config = dict(
        img_size=224, num_frames=4, patch_size=[1, 16, 16], in_chans=6, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4,
        coords_encoding=["time", "location"], coords_scale_learn=True,
    )
    reference = reference_mae.PrithviMAE(**config).eval()
    reference.load_state_dict(state, strict=True)
    port = PrithviMAE(**config).eval()
    port.load_state_dict(state, strict=True)

    torch.manual_seed(2)
    x = torch.randn(1, 6, 4, 224, 224)
    temporal, location = _coords(1, 4)
    with torch.no_grad():
        for coords in [(temporal, location), (None, None)]:
            torch.manual_seed(3)
            ref_loss, ref_pred, ref_mask = reference(x, *coords)
            torch.manual_seed(3)
            loss, pred, mask = port(x, *coords)
            _assert_close(loss, ref_loss)
            _assert_close(pred, ref_pred)
            assert torch.equal(mask, ref_mask)
    del reference

    # Encoder loaded through the registry (MAE checkpoint -> encoder keys) vs TerraTorch's backbone loader.
    from terratorch.registry import BACKBONE_REGISTRY

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        backbone = BACKBONE_REGISTRY.build("prithvi_eo_v2_300_tl", pretrained=True, ckpt_path=str(checkpoint)).eval()
    model = build_model("prithvi_eo_2_tl", task="segmentation", weights_path=str(checkpoint)).eval()
    _assert_same_state(backbone, model.encoder)
    x = torch.randn(1, 6, 224, 224)
    temporal, location = _coords(1, 1)
    with torch.no_grad():
        for actual, expected in zip(
            model.encoder.forward_features(x, temporal, location),
            backbone(x, temporal_coords=temporal, location_coords=location),
        ):
            _assert_close(actual, expected)
