"""Prithvi-EO-2.0 ports: parameter counts, shapes, input validation and the weight loader.

Numerical equivalence with TerraTorch 0.99.8 and the official prithvi_mae.py, including the released
checkpoints, lives in tests/oracle/test_prithvi_oracle.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyhazards.models import build_model
from pyhazards.models import prithvi
from pyhazards.models.prithvi import (
    PRITHVI_WEIGHTS,
    PrithviMAE,
    download_weights,
    finetuned_state_dict_to_model,
    load_pretrained_state_dict,
    mae_state_dict_to_encoder,
)

TINY = dict(embed_dim=64, depth=4, num_heads=2, select_indices=(0, 1, 2, 3), decoder_channels=(32, 16, 8, 4))
TL_TINY_MAE = dict(
    img_size=32, num_frames=4, in_chans=6, embed_dim=64, depth=4, num_heads=2, decoder_embed_dim=32,
    decoder_depth=1, decoder_num_heads=2, coords_encoding=["time", "location"], coords_scale_learn=True,
)


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize(
    "name, kwargs, expected",
    [
        ("prithvi_burnscars", {}, 324_204_674),
        ("prithvi_eo_2_tl", {}, 324_204_676),
        ("prithvi_eo_2_tl", {"variant": "600m"}, 656_889_540),
    ],
)
def test_parameter_counts(name, kwargs, expected):
    with torch.device("meta"):
        model = build_model(name, task="segmentation", **kwargs)
    assert _n_params(model) == expected


def test_encoder_and_mae_parameter_counts():
    with torch.device("meta"):
        tl = build_model("prithvi_eo_2_tl", task="segmentation")
        tl600 = build_model("prithvi_eo_2_tl", task="segmentation", variant="600m")
        mae = PrithviMAE(
            num_frames=4, in_chans=6, embed_dim=1024, depth=24, num_heads=16,
            coords_encoding=["time", "location"], coords_scale_learn=True,
        )
    assert _n_params(tl.encoder) == 303_886_338
    assert _n_params(tl600.encoder) == 631_188_482
    assert _n_params(mae) == 330_419_716
    assert tl.neck[0].indices == [5, 11, 17, 23]
    assert tl600.neck[0].indices == [7, 15, 23, 31]


def test_state_dict_follows_terratorch_layout():
    keys = list(build_model("prithvi_burnscars", task="segmentation", **TINY).state_dict())
    assert keys[:3] == ["encoder.cls_token", "encoder.pos_embed", "encoder.patch_embed.proj.weight"]
    assert "encoder.blocks.3.attn.qkv.weight" in keys and "encoder.blocks.3.mlp.fc2.bias" in keys
    assert "decoder.decoder.blocks.3.conv2.1.running_var" in keys
    assert "head.head.2.weight" in keys
    assert keys[-1] == "neck.2.fpn2.0.bias"
    tl_keys = build_model("prithvi_eo_2_tl", task="segmentation", **TINY).state_dict()
    assert {"encoder.temporal_embed_enc.scale", "encoder.location_embed_enc.scale"} <= set(tl_keys)


@pytest.mark.parametrize("size", [(64, 64), (70, 90), (40, 96)])
def test_burnscars_output_shape(size):
    model = build_model("prithvi_burnscars", task="segmentation", **TINY).eval()
    x = torch.randn(2, 6, *size)
    with torch.no_grad():
        out = model(x)
        assert out.shape == (2, 2) + size
        torch.testing.assert_close(model(x[:, :, None]), out)


def test_tl_coordinates_and_multitemporal_input():
    model = build_model("prithvi_eo_2_tl", task="segmentation", num_frames=3, num_classes=4, **TINY).eval()
    x = torch.randn(2, 6, 3, 64, 64)
    temporal = torch.tensor([[[2020, 10], [2020, 100], [2020, 200]]] * 2)  # integer years / days are accepted
    location = torch.tensor([[30.0, -100.0], [31.0, -101.0]])
    with torch.no_grad():
        out = model(x, temporal_coords=temporal, location_coords=location)
        assert out.shape == (2, 4, 64, 64)
        assert not torch.allclose(out, model(x))
        assert not torch.allclose(out, model(x, temporal_coords=temporal))
    assert model.neck[2].fpn1[0].in_channels == 3 * 64  # frames stacked along channels


def test_input_validation():
    model = build_model("prithvi_burnscars", task="segmentation", **TINY)
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(2, 6, 64))
    with pytest.raises(ValueError, match="channels"):
        model(torch.randn(2, 5, 64, 64))
    with pytest.raises(ValueError, match="T=1"):
        model(torch.randn(2, 6, 2, 64, 64))
    with pytest.raises(ValueError, match="too small"):
        model(torch.randn(1, 6, 8, 8))
    tl = build_model("prithvi_eo_2_tl", task="segmentation", **TINY)
    x = torch.randn(2, 6, 64, 64)
    with pytest.raises(ValueError, match="temporal_coords"):
        tl(x, temporal_coords=torch.zeros(2, 2))
    with pytest.raises(ValueError, match="temporal_coords"):
        tl(x, temporal_coords=torch.zeros(2, 2, 2))
    with pytest.raises(ValueError, match="location_coords"):
        tl(x, location_coords=torch.zeros(2, 3))


def test_builder_validation():
    with pytest.raises(ValueError, match="segmentation"):
        build_model("prithvi_burnscars", task="classification")
    with pytest.raises(ValueError, match="segmentation"):
        build_model("prithvi_eo_2_tl", task="regression")
    with pytest.raises(ValueError, match="variant"):
        build_model("prithvi_eo_2_tl", task="segmentation", variant="100m")
    with pytest.raises(ValueError, match="default architecture"):
        build_model("prithvi_burnscars", task="segmentation", pretrained=True, depth=4)
    with pytest.raises(ValueError, match="official TL weights"):
        build_model("prithvi_eo_2_tl", task="segmentation", pretrained=True, in_channels=4)
    with pytest.raises(ValueError, match="select_indices"):
        build_model("prithvi_burnscars", task="segmentation", embed_dim=64, depth=4, num_heads=2)


def test_mae_patchify_and_reconstruction_shapes():
    mae = PrithviMAE(**TL_TINY_MAE).eval()
    x = torch.randn(2, 6, 4, 32, 32)
    assert torch.equal(mae.unpatchify(mae.patchify(x), image_size=(32, 32)), x)
    temporal = torch.tensor([[[2018.0, 26], [2018, 106], [2018, 201], [2018, 266]]] * 2)
    with torch.no_grad():
        loss, pred, mask = mae(x, temporal_coords=temporal, location_coords=torch.tensor([[25.0, -104.0]] * 2))
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert pred.shape == (2, 16, 16 * 16 * 6)
    assert mask.shape == (2, 16) and mask.sum().item() == 2 * 12  # 75 % of the patches are masked


def test_weights_are_pinned():
    for spec in PRITHVI_WEIGHTS.values():
        assert spec.repo_id.startswith("ibm-nasa-geospatial/")
        assert len(spec.revision) == 40 and len(spec.sha256) == 64
        assert spec.num_bytes > 10**9


def test_download_weights_uses_cache(tmp_path, monkeypatch):
    spec = prithvi.PretrainedWeights("org/repo", "a" * 40, "w.pt", "b" * 64, 4)
    calls = []

    def fake_download(url, dst, hash_prefix=None, progress=True):
        calls.append((url, hash_prefix))
        Path(dst).write_bytes(b"1234")

    monkeypatch.setattr(torch.hub, "download_url_to_file", fake_download)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    path = download_weights(spec, cache_dir=tmp_path)
    assert path == tmp_path / "org--repo" / ("a" * 40) / "w.pt"
    assert calls == [(f"https://huggingface.co/org/repo/resolve/{'a' * 40}/w.pt", "b" * 64)]
    assert download_weights(spec, cache_dir=tmp_path) == path
    assert len(calls) == 1
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example/")
    assert spec.url.startswith("https://mirror.example/org/repo/resolve/")


def test_finetuned_checkpoint_key_mapping(tmp_path):
    torch.manual_seed(0)
    source = build_model("prithvi_burnscars", task="segmentation", **TINY).eval()
    path = tmp_path / "finetuned.pt"
    torch.save({"epoch": 1, "state_dict": {"model." + k: v for k, v in source.state_dict().items()}}, path)
    state = load_pretrained_state_dict("prithvi_eo_v2_300_burnscars", weights_path=path)
    target = build_model("prithvi_burnscars", task="segmentation", **TINY).eval()
    target.load_state_dict(finetuned_state_dict_to_model(state), strict=True)
    x = torch.randn(1, 6, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(target(x), source(x))


def test_mae_checkpoint_to_encoder_mapping():
    torch.manual_seed(0)
    mae = PrithviMAE(**TL_TINY_MAE)
    model = build_model("prithvi_eo_2_tl", task="segmentation", img_size=32, **TINY)
    model.encoder.load_state_dict(mae_state_dict_to_encoder(mae.state_dict(), model.encoder), strict=True)
    for key, value in model.encoder.state_dict().items():
        if key != "pos_embed":  # 1-frame table instead of the 4-frame pretraining table
            assert torch.equal(value, mae.encoder.state_dict()[key]), key
