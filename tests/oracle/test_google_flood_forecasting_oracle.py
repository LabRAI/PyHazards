"""google_flood_forecasting against googlehydrology's MeanEmbeddingForecastLSTM and the released FloodHub weights.

Reference (pinned in repos.yaml): google-research/flood-forecasting @ cdda28d (Apache-2.0),
``googlehydrology/modelzoo/mean_embedding_forecast_lstm.py``, built from the released run configuration
``pretrained-models/google-floodhub-settings-110-epochs/config.yml``. Weights: ``model_epoch110.pt`` of that
run (asset ``floodhub_weights``, sha256-pinned; saved from a ``torch.compile``d model, hence the
``_orig_mod.`` prefix). googlehydrology needs Python >= 3.10 and ruamel.yaml, xarray, dask, zarr,
more-itertools (tests/oracle/requirements-flood.txt).
"""

from __future__ import annotations

import hashlib
import sys

import pytest
import torch

from oracle_utils import MANIFEST, oracle_asset, oracle_repo
from pyhazards.models import build_model
from pyhazards.models.google_flood_forecasting import FLOODHUB_CONFIG, FLOODHUB_WEIGHTS

RUN = "pretrained-models/google-floodhub-settings-110-epochs"


@pytest.fixture(scope="module")
def gh():
    root = oracle_repo("flood-forecasting")
    for package in ("ruamel.yaml", "zarr", "dask", "more_itertools"):
        pytest.importorskip(package)
    sys.path.insert(0, str(root))
    try:
        from ruamel.yaml import YAML

        from googlehydrology.modelzoo.mean_embedding_forecast_lstm import MeanEmbeddingForecastLSTM
        from googlehydrology.utils.config import Config

        yield {"root": root, "YAML": YAML, "Config": Config, "Model": MeanEmbeddingForecastLSTM}
    finally:
        sys.path.remove(str(root))
        for name in [m for m in sys.modules if m.split(".")[0] == "googlehydrology"]:
            del sys.modules[name]


def _release_config(gh, **overrides):
    run = gh["root"] / RUN
    raw = gh["YAML"](typ="safe").load((run / "config.yml").read_text())
    raw.update({"run_dir": str(run), "img_log_dir": None, "device": "cpu", "compile": False})
    raw.update(overrides)
    return gh["Config"](raw)


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _assert_same_state(reference: torch.nn.Module, port: torch.nn.Module) -> None:
    ref_state, port_state = reference.state_dict(), port.state_dict()
    assert list(ref_state) == list(port_state)
    for key, value in ref_state.items():
        assert torch.equal(value, port_state[key]), key


def _released_state() -> dict:
    path = oracle_asset("floodhub_weights") / "model_epoch110.pt"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == MANIFEST["assets"]["floodhub_weights"]["sha256"] == FLOODHUB_WEIGHTS["sha256"]
    return path, torch.load(path, map_location="cpu", weights_only=True)


def _inputs(batch: int = 3):
    """Random inputs in googlehydrology's dict layout, with missing products and fully missing steps."""
    torch.manual_seed(11)
    hindcast_names = [f for g in FLOODHUB_CONFIG["hindcast_inputs"].values() for f in g]
    forecast_names = [f for g in FLOODHUB_CONFIG["forecast_inputs"].values() for f in g]
    total = FLOODHUB_CONFIG["seq_length"] + FLOODHUB_CONFIG["lead_time"]
    hindcast = {name: torch.randn(batch, FLOODHUB_CONFIG["seq_length"], 1) for name in hindcast_names}
    forecast = {name: torch.randn(batch, total, 1) for name in forecast_names}
    hindcast["imerg_precipitation"][1] = float("nan")  # one product missing for a whole sample
    hindcast["cpc_precipitation"][0, 50:80] = float("nan")  # a gap in one product
    for name in forecast_names:  # every product missing on one day: outputs NaN from that step on
        forecast[name][2, 300] = float("nan")
    x_s = torch.randn(batch, len(FLOODHUB_CONFIG["static_attributes"]))
    return {"x_s": x_s, "x_d_hindcast": hindcast, "x_d_forecast": forecast}


def _tensor_layout(data, model):
    return {
        "x_s": data["x_s"],
        "x_d_hindcast": torch.cat([data["x_d_hindcast"][f] for f in model.hindcast_features], dim=-1),
        "x_d_forecast": torch.cat([data["x_d_forecast"][f] for f in model.forecast_features], dim=-1),
    }


def _assert_outputs(actual, expected):
    assert set(actual) == {"mu", "b", "tau", "pi"} <= set(expected)
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-6, equal_nan=True)


def test_parameter_count_and_seeded_initialisation(gh):
    cfg = _release_config(gh)
    for seed in (0, 449153):
        torch.manual_seed(seed)
        reference = gh["Model"](cfg)
        torch.manual_seed(seed)
        port = build_model("google_flood_forecasting", task="regression")
        assert _n_params(reference) == _n_params(port) == 3_402_832
        _assert_same_state(reference, port)


def test_released_floodhub_weights(gh):
    path, state = _released_state()
    assert all(key.startswith("_orig_mod.") for key in state)
    reference = gh["Model"](_release_config(gh))
    reference.load_state_dict({key[len("_orig_mod."):]: value for key, value in state.items()}, strict=True)
    port = build_model("google_flood_forecasting", task="regression", weights_path=path)
    _assert_same_state(reference, port)

    data = _inputs()
    reference.eval()
    port.eval()
    with torch.no_grad():
        expected = reference(data)
        _assert_outputs(port(data), expected)
        _assert_outputs(port(_tensor_layout(data, port)), expected)
        torch.testing.assert_close(port.point_prediction(port(data)), reference.point_prediction(expected), equal_nan=True)
    assert torch.isnan(expected["mu"][2, 300:]).all() and not torch.isnan(expected["mu"][2, :300]).any()
    assert not torch.isnan(expected["mu"][:2]).any()  # a missing product is skipped by the masked mean

    reference.train()
    port.train()
    torch.manual_seed(3)
    expected = reference(data)
    torch.manual_seed(3)
    _assert_outputs(port(data), expected)


def test_streamflow_layout_matches_reference_configuration(gh):
    """config='streamflow' (PyHazards layout) equals googlehydrology with one shared group and lead_time 0."""
    features = [f"x_d_{i}" for i in range(5)]
    statics = [f"static_{i}" for i in range(27)]
    cfg = _release_config(
        gh, hindcast_inputs={"x_d": features}, forecast_inputs={"x_d": features}, static_attributes=statics,
        lead_time=0, forecast_overlap=30, seq_length=30, hidden_size=32,
    )
    torch.manual_seed(4)
    reference = gh["Model"](cfg)
    torch.manual_seed(4)
    port = build_model("google_flood_forecasting", task="regression", config="streamflow", seq_length=30, hidden_size=32)
    _assert_same_state(reference, port)
    torch.manual_seed(0)
    x_d, x_s = torch.randn(4, 30, 5), torch.randn(4, 27)
    as_dict = {name: x_d[..., i: i + 1] for i, name in enumerate(features)}
    reference.eval()
    port.eval()
    with torch.no_grad():
        expected = reference({"x_s": x_s, "x_d_hindcast": as_dict, "x_d_forecast": as_dict})
        _assert_outputs(port({"x_d": x_d, "x_s": x_s}), expected)
