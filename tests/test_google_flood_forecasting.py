"""google_flood_forecasting without the reference code: configuration, NaN handling, layouts, validation."""

from __future__ import annotations

import pytest
import torch

from pyhazards.models import build_model
from pyhazards.models.google_flood_forecasting import FLOODHUB_CONFIG, FLOODHUB_WEIGHTS


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_floodhub_configuration():
    model = build_model("google_flood_forecasting", task="regression")
    assert _n_params(model) == 3_402_832  # the released model
    sizes = {name: _n_params(module) for name, module in model.named_children()}
    assert sizes == {
        "static_embedding_fc": 20_620,
        "hindcast_embeddings_fc": 8_440,
        "forecast_embeddings_fc": 0,
        "shared_embeddings_fc": 3_500,
        "hindcast_lstm": 1_134_592,
        "forecast_lstm": 2_183_168,
        "dropout": 0,
        "head": 52_512,
    }
    assert list(model.hindcast_embeddings_fc) == ["imerg", "cpc"] and list(model.shared_embeddings_fc) == ["hres", "graphcast"]
    assert torch.all(model.hindcast_lstm.bias_hh_l0[512:1024] == 3.0)
    assert len(FLOODHUB_WEIGHTS["sha256"]) == 64


def _floodhub_inputs(batch=2):
    torch.manual_seed(0)
    total = FLOODHUB_CONFIG["seq_length"] + FLOODHUB_CONFIG["lead_time"]
    return {
        "x_s": torch.randn(batch, 84),
        "x_d_hindcast": torch.randn(batch, FLOODHUB_CONFIG["seq_length"], 9),
        "x_d_forecast": torch.randn(batch, total, 7),
    }


def test_outputs_and_missing_inputs():
    model = build_model("google_flood_forecasting", task="regression").eval()
    data = _floodhub_inputs()
    with torch.no_grad():
        out = model(data)
        assert set(out) == {"mu", "b", "tau", "pi"}
        assert all(v.shape == (2, 372, 3) for v in out.values())
        assert torch.allclose(out["pi"].sum(-1), torch.ones(2, 372), atol=1e-4)
        assert model.point_prediction(out).shape == (2, 372, 1)

        # A missing product (imerg = hindcast column 7) is skipped by the masked mean: outputs stay finite.
        data["x_d_hindcast"][0, :, 7] = float("nan")
        assert torch.isfinite(model(data)["mu"][0]).all()
        # All forecast products missing on one day: NaN from that day on.
        data["x_d_forecast"][1, 200] = float("nan")
        mu = model(data)["mu"][1]
        assert torch.isfinite(mu[:200]).all() and torch.isnan(mu[200:]).all()


def test_streamflow_layout():
    model = build_model("google_flood_forecasting", task="regression", config="streamflow", seq_length=30, hidden_size=16)
    out = model({"x_d": torch.randn(3, 30, 5), "x_s": torch.randn(3, 27)})
    assert out["mu"].shape == (3, 30, 3)
    floodhub = build_model("google_flood_forecasting", task="regression")
    with pytest.raises(ValueError, match="lead_time=0"):
        floodhub({"x_d": torch.randn(2, 372, 9), "x_s": torch.randn(2, 84)})


def test_bad_inputs_raise():
    model = build_model("google_flood_forecasting", task="regression")
    data = _floodhub_inputs()
    with pytest.raises(ValueError, match="shape"):
        model({**data, "x_d_forecast": torch.randn(2, 365, 7)})
    with pytest.raises(ValueError, match="shape"):
        model({**data, "x_d_hindcast": torch.randn(2, 365, 8)})
    with pytest.raises(ValueError, match="x_s"):
        model({**data, "x_s": torch.randn(2, 80)})
    with pytest.raises(ValueError, match="missing features"):
        model({**data, "x_d_hindcast": {"hres_temperature_2m": torch.randn(2, 365, 1)}})
    with pytest.raises(ValueError):
        build_model("google_flood_forecasting", task="regression", config="transformer")
    with pytest.raises(ValueError):
        build_model("google_flood_forecasting", task="regression", history=4)
