import torch

from pyhazards.models import available_models, build_model
from pyhazards.models.wildfire_fpa_autoencoder import WildfireFPAAutoencoder
from pyhazards.models.wildfire_fpa_lstm import WildfireFPALSTM


def test_wildfire_fpa_public_registry_name():
    assert [name for name in available_models() if "wildfire_fpa" in name] == ["wildfire_fpa"]


def test_wildfire_fpa_classification_forward():
    model = build_model(
        name="wildfire_fpa",
        task="classification",
        in_dim=8,
        out_dim=5,
    )

    logits = model(torch.randn(4, 8))
    assert model.stage == "classification"
    assert logits.shape == (4, 5)


def test_wildfire_fpa_forecast_forward():
    model = build_model(
        name="wildfire_fpa",
        task="forecasting",
        input_dim=7,
        output_dim=5,
        lookback=12,
        latent_dim=16,
    )

    preds = model(torch.randn(3, 12, 7))
    assert model.stage == "forecasting"
    assert preds.shape == (3, 5)


def test_wildfire_fpa_forecast_reconstruction_output():
    model = build_model(name="wildfire_fpa", task="forecasting", input_dim=7, output_dim=5, lookback=12)

    preds, recon = model.forward_with_reconstruction(torch.randn(2, 12, 7))
    assert preds.shape == (2, 5)
    assert recon.shape == (2, 12, 7)

def test_wildfire_fpa_internal_lstm_forward():
    model = WildfireFPALSTM(input_dim=7, output_dim=5, lookback=12)

    preds = model(torch.randn(3, 12, 7))
    assert preds.shape == (3, 5)


def test_wildfire_fpa_autoencoder_forward():
    model = WildfireFPAAutoencoder(input_dim=7, lookback=12, latent_dim=16)

    recon = model(torch.randn(2, 12, 7))
    assert recon.shape == (2, 12, 7)


def test_added_wildfire_public_methods_forward():
    daily_x = torch.randn(2, 10, 25)
    temporal_spread_x = torch.randn(2, 4, 6, 16, 16)

    danger = build_model(name="wildfire_forecasting", task="classification")
    spread_ts = build_model(
        name="wildfirespreadts",
        task="segmentation",
        history=4,
        in_channels=6,
    )
    # FireCastNet on a 16 x 16 window of 0.25-degree cells (4 x 4 one-degree graph nodes).
    firecastnet = build_model(
        name="firecastnet",
        task="segmentation",
        in_channels=6,
        timeseries_len=4,
        mesh_levels=(0, 1, 2, 3, 4),
        sp_res=0.25,
        max_lat=41.875,
        min_lat=38.125,
        max_lon=-120.125,
        min_lon=-123.875,
        hidden_dim=16,
        embed_cube_dim=16,
    )

    assert danger(daily_x).shape == (2, 2)
    assert spread_ts(temporal_spread_x).shape == (2, 1, 16, 16)
    assert firecastnet(temporal_spread_x).shape == (2, 1, 16, 16)
