import torch

from pyhazards.models import build_model


def test_build_model_wildfire_fpa_mlp_forward():
    B, D, C = 8, 16, 5
    model = build_model(
        name="wildfire_fpa_mlp",
        task="classification",
        in_dim=D,
        out_dim=C,
    )
    x = torch.randn(B, D)
    y = model(x)
    assert y.shape == (B, C)


def test_build_model_wildfire_fpa_lstm_forward():
    B, T, D, O = 4, 12, 10, 1
    model = build_model(
        name="wildfire_fpa_lstm",
        task="forecasting",
        input_dim=D,
        output_dim=O,
        hidden_dim=32,
        num_layers=1,
        dropout=0.0,
        lookback=T,
    )
    x = torch.randn(B, T, D)
    y = model(x)
    assert y.shape == (B, O)
