import torch

from pyhazards.configs import load_experiment_config
from pyhazards.engine.runner import BenchmarkRunner
from pyhazards.models import build_model


def test_neuralhydrology_baselines_forward():
    batch = {"x": torch.randn(2, 4, 6, 2)}
    for name in ["neuralhydrology_lstm", "neuralhydrology_ealstm", "google_flood_forecasting"]:
        model = build_model(name=name, task="regression", input_dim=2, out_dim=1)
        preds = model(batch)
        assert preds.shape == (2, 6, 1)


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
        assert "rmse" in summary.metrics
        assert "nse" in summary.metrics


def test_flood_inundation_breadth_configs(tmp_path):
    for config_name in [
        "pyhazards/configs/flood/floodcast_smoke.yaml",
        "pyhazards/configs/flood/urbanfloodcast_smoke.yaml",
    ]:
        summary = BenchmarkRunner().run(
            load_experiment_config(config_name),
            output_dir=str(tmp_path),
        )
        assert summary.hazard_task == "flood.inundation"
        assert "iou" in summary.metrics
        assert "pixel_mae" in summary.metrics
