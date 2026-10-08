import torch

from pyhazards.configs import load_experiment_config
from pyhazards.engine.runner import BenchmarkRunner
from pyhazards.models import build_model


def test_earthquake_models_follow_their_original_contracts():
    with torch.no_grad():
        assert build_model("phasenet", task="picking").eval()(torch.randn(2, 3, 3001)).shape == (2, 3, 3001)
        assert build_model("eqtransformer", task="picking").eval()(torch.randn(1, 3, 6000)).shape == (1, 3, 6000)
        assert build_model("gpd", task="classification").eval()(torch.randn(2, 3, 400)).shape == (2, 3)
        # The experimental eqnet stand-in still regresses one (P, S) pair per trace.
        assert build_model("eqnet", task="regression", in_channels=3)(torch.randn(3, 3, 256)).shape == (3, 2)


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
        assert "p_f1" in summary.metrics and "s_pick_mae" in summary.metrics
        assert summary.metadata["protocol"] == "eqtransformer"


def test_wavecastnet_forecasting_benchmark(tmp_path):
    config = load_experiment_config("pyhazards/configs/earthquake/wavecastnet_benchmark_smoke.yaml")
    summary = BenchmarkRunner().run(config, output_dir=str(tmp_path))

    assert summary.benchmark_name == "earthquake"
    assert summary.hazard_task == "earthquake.forecasting"
    assert "mae" in summary.metrics
    assert "mse" in summary.metrics
