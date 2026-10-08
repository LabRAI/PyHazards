import torch

from pyhazards.configs import load_experiment_config
from pyhazards.engine.runner import BenchmarkRunner
from pyhazards.models import build_model


def test_streamflow_models_forward():
    batch = {"x_d": torch.randn(2, 30, 5), "x_s": torch.randn(2, 27)}
    for name in ["neuralhydrology_lstm", "neuralhydrology_ealstm"]:
        out = build_model(name=name, task="regression", hidden_size=16)(batch)
        assert out["y_hat"].shape == (2, 30, 1)
    google = build_model("google_flood_forecasting", task="regression", config="streamflow", seq_length=30, hidden_size=16)
    assert google.point_prediction(google(batch)).shape == (2, 30, 1)


def test_inundation_baselines_forward():
    x = torch.randn(2, 4, 3, 16, 16)
    for name in ["floodcast", "urbanfloodcast"]:
        model = build_model(name=name, task="regression", in_channels=3, history=4)
        preds = model(x)
        assert preds.shape == (2, 1, 16, 16)


def test_flood_streamflow_breadth_configs(tmp_path):
    for config_name in [
        "pyhazards/configs/flood/neuralhydrology_lstm_smoke.yaml",
        "pyhazards/configs/flood/neuralhydrology_ealstm_smoke.yaml",
        "pyhazards/configs/flood/google_flood_forecasting_smoke.yaml",
    ]:
        summary = BenchmarkRunner().run(
            load_experiment_config(config_name),
            output_dir=str(tmp_path),
        )
        assert summary.hazard_task == "flood.streamflow"
        for metric in ("nse", "nse_mean", "kge", "alpha_nse", "fhv", "n_basins"):
            assert metric in summary.metrics
        assert summary.metrics["n_basins"] == 3
        assert set(summary.metadata["per_basin"]) == {"synthetic_000", "synthetic_001", "synthetic_002"}


def test_flood_inundation_breadth_configs(tmp_path):
    for config_name in [
        "pyhazards/configs/flood/floodcast_smoke.yaml",
        "pyhazards/configs/flood/urbanfloodcast_smoke.yaml",
        "pyhazards/configs/flood/hydrographnet_smoke.yaml",
    ]:
        summary = BenchmarkRunner().run(
            load_experiment_config(config_name),
            output_dir=str(tmp_path),
        )
        assert summary.hazard_task == "flood.inundation"
        assert "iou" in summary.metrics
        assert "pixel_mae" in summary.metrics
