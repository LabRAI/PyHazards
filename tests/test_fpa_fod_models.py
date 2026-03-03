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
