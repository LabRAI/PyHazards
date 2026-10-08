import hashlib

import pytest
import torch

from pyhazards.datasets.tc.synthetic import synthetic_tcnd_batch
from pyhazards.models import build_model
from pyhazards.models.tropicalcyclone_mlp import SHIPS_PREDICTORS, load_keras_weights
from pyhazards.models.tropicyclonenet import (
    ENV_FEATURES,
    TrajectoryDiscriminator,
    TropiCycloneNet,
    load_tropicyclonenet_checkpoint,
)


def _n_params(model):
    return sum(p.numel() for p in model.parameters())


def test_tropicalcyclone_mlp_is_the_official_24h_model():
    model = build_model("tropicalcyclone_mlp", task="regression")
    assert _n_params(model) == 4_448_257
    assert len(SHIPS_PREDICTORS) == 121 and len(set(SHIPS_PREDICTORS)) == 121
    assert model(torch.randn(3, 121)).shape == (3, 1)
    for name in model.layer_names:
        assert torch.count_nonzero(getattr(model, name).bias) == 0
    with pytest.raises(ValueError, match="predictors shaped"):
        model(torch.randn(3, 120))
    with pytest.raises(ValueError, match="regression"):
        build_model("tropicalcyclone_mlp", task="classification")


def test_tropicalcyclone_mlp_loads_keras_weight_lists():
    model = build_model("tropicalcyclone_mlp", task="regression", input_dim=4, hidden_dims=[3, 2])
    kernels = [torch.randn(4, 3), torch.randn(3), torch.randn(3, 2), torch.randn(2), torch.randn(2, 1), torch.randn(1)]
    load_keras_weights(model, [k.numpy() for k in kernels])
    x = torch.randn(5, 4)
    expected = torch.relu(torch.sigmoid(x @ kernels[0] + kernels[1]) @ kernels[2] + kernels[3]) @ kernels[4] + kernels[5]
    torch.testing.assert_close(model(x), expected)
    with pytest.raises(ValueError, match="expected 6 arrays"):
        load_keras_weights(model, [k.numpy() for k in kernels[:4]])


def test_saf_net_parameters_shapes_and_validation():
    model = build_model("saf_net", task="regression").eval()
    assert _n_params(model) == 880_233
    wide, deep = torch.rand(2, 96), torch.rand(2, 2, 4, 31, 31, 4)
    out = model(wide, deep)
    assert out.shape == (2, 1) and torch.all(out >= 0)
    torch.testing.assert_close(model({"wide": wide, "deep": deep}), out)
    with pytest.raises(ValueError, match="deep inputs shaped"):
        model(wide, torch.rand(2, 2, 4, 31, 31, 3))
    with pytest.raises(ValueError, match="needs both"):
        model(wide)


def test_tropicyclonenet_generator_and_discriminator():
    model = build_model("tropicyclonenet", task="regression").eval()
    assert isinstance(model, TropiCycloneNet)
    assert _n_params(model) == 4_767_195
    assert _n_params(TrajectoryDiscriminator()) == 231_009
    batch = synthetic_tcnd_batch(3, generator=torch.Generator().manual_seed(0))["inputs"]
    with torch.no_grad():
        rel, frames, logits, index = model(batch, num_samples=6)
    assert rel.shape == (4, 6, 3, 4) and frames.shape == (3, 1, 12, 64, 64)
    assert logits.shape == (3, 6) and index.shape == (3, 6)
    with torch.no_grad():
        every = model(batch, all_g_out=True)[0]
    assert every.shape == (4, 6, 3, 4)

    # Env features may also sit at the top level of the mapping (smoke-test layout).
    flat = {key: value for key, value in batch.items() if key != "env_data"}
    flat.update(batch["env_data"])
    with torch.no_grad():
        torch.manual_seed(0)
        a = model(flat, num_samples=2)[0]
        torch.manual_seed(0)
        b = model(batch, num_samples=2)[0]
    torch.testing.assert_close(a, b)


def test_tropicyclonenet_forecast_is_cumulative_and_in_physical_units():
    model = build_model("tropicyclonenet", task="regression").eval()
    batch = synthetic_tcnd_batch(2, generator=torch.Generator().manual_seed(1))["inputs"]
    with torch.no_grad():
        torch.manual_seed(5)
        forecast = model.forecast(batch, num_samples=6)
        torch.manual_seed(5)
        rel = model(batch, num_samples=6)[0]
    assert set(forecast) >= {"lat", "lon", "pres", "wind", "logits", "generator_index"}
    assert forecast["lat"].shape == (2, 6, 4)
    last = batch["obs_traj"][-1]  # (B, 4) normalised lon, lat, pres, wind
    expected_lon = (last[:, 0].view(2, 1, 1) + torch.cumsum(rel[..., 0].permute(2, 1, 0), dim=-1)) * 5 + 180
    expected_wind = (last[:, 3].view(2, 1, 1) + torch.cumsum(rel[..., 3].permute(2, 1, 0), dim=-1)) * 25 + 40
    torch.testing.assert_close(forecast["lon"], expected_lon)
    torch.testing.assert_close(forecast["wind"], expected_wind)


def test_tropicyclonenet_validation_and_checkpoint_guard(tmp_path):
    model = build_model("tropicyclonenet", task="regression")
    batch = synthetic_tcnd_batch(2)["inputs"]
    bad = dict(batch, image_obs=torch.rand(2, 1, 8, 32, 32))
    with pytest.raises(ValueError, match="image_obs must be shaped"):
        model(bad)
    env = dict(batch["env_data"])
    env.pop(ENV_FEATURES[0][0])
    with pytest.raises(ValueError, match="env_data is missing"):
        model(dict(batch, env_data=env))
    with pytest.raises(ValueError, match="pooling"):
        TropiCycloneNet(pooling_type="pool_net")
    with pytest.raises(ValueError, match="obs_len=8"):
        TropiCycloneNet(obs_len=6)
    fake = tmp_path / "checkpoint.pt"
    torch.save({"g_state": model.state_dict()}, fake)
    with pytest.raises(ValueError, match="sha256"):
        load_tropicyclonenet_checkpoint(fake)
    restored = load_tropicyclonenet_checkpoint(fake, verify_sha256=False)
    assert hashlib.sha256(fake.read_bytes()).hexdigest()
    for key, value in model.state_dict().items():
        assert torch.equal(restored.state_dict()[key], value)


def test_experimental_storm_adapters_keep_the_generic_interface():
    x = torch.randn(2, 6, 8)
    for name in ["hurricast", "tcif_fusion"]:
        model = build_model(name=name, task="regression", input_dim=8, horizon=5, output_dim=3)
        assert model(x).shape == (2, 5, 3)


def test_weather_model_placeholders_are_not_registered():
    # FourCastNet, GraphCast and Pangu-Weather are forecast + tracker pipelines (pyhazards.forecasts).
    from pyhazards.models import available_models

    assert not {"graphcast_tc", "pangu_tc", "fourcastnet_tc"} & set(available_models())
