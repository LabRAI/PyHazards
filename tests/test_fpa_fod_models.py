import torch

from pyhazards.models import build_model


def test_wildfire_fpa_dnn_forward():
    model = build_model(
        name="wildfire_fpa_dnn",
        task="classification",
        in_dim=8,
        out_dim=5,
    )

    logits = model(torch.randn(4, 8))
    assert logits.shape == (4, 5)

def test_wildfire_fpa_lstm_forward():
    model = build_model(
        name="wildfire_fpa_lstm",
        task="forecasting",
        input_dim=7,
        output_dim=5,
        lookback=12,
    )

    preds = model(torch.randn(3, 12, 7))
    assert preds.shape == (3, 5)


def test_wildfire_fpa_forecast_forward():
    model = build_model(
        name="wildfire_fpa_forecast",
        task="forecasting",
        input_dim=7,
        output_dim=5,
        lookback=12,
        latent_dim=16,
    )

    preds = model(torch.randn(3, 12, 7))
    assert preds.shape == (3, 5)


def test_wildfire_fpa_autoencoder_forward():
    model = build_model(
        name="wildfire_fpa_autoencoder",
        task="autoencoder",
        input_dim=7,
        lookback=12,
        latent_dim=16,
    )

    recon = model(torch.randn(2, 12, 7))
    assert recon.shape == (2, 12, 7)
