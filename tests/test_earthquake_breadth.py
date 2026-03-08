import torch

from pyhazards.configs import load_experiment_config
from pyhazards.engine.runner import BenchmarkRunner
from pyhazards.models import build_model


def test_additional_earthquake_picking_models_forward():
    x = torch.randn(3, 3, 256)
    for name in ["eqtransformer", "gpd", "eqnet"]:
        model = build_model(name=name, task="regression", in_channels=3)
        preds = model(x)
        assert preds.shape == (3, 2)


def test_earthquake_breadth_configs(tmp_path):
    for config_name in [
        "pyhazards/configs/earthquake/eqtransformer_smoke.yaml",
        "pyhazards/configs/earthquake/gpd_smoke.yaml",
        "pyhazards/configs/earthquake/eqnet_smoke.yaml",
    ]:
        summary = BenchmarkRunner().run(
            load_experiment_config(config_name),
            output_dir=str(tmp_path),
        )
        assert summary.hazard_task == "earthquake.picking"
        assert "f1" in summary.metrics


def test_wavecastnet_forecasting_benchmark(tmp_path):
    config = load_experiment_config("pyhazards/configs/earthquake/wavecastnet_benchmark_smoke.yaml")
    summary = BenchmarkRunner().run(config, output_dir=str(tmp_path))

    assert summary.benchmark_name == "earthquake"
    assert summary.hazard_task == "earthquake.forecasting"
    assert "mae" in summary.metrics
    assert "mse" in summary.metrics
