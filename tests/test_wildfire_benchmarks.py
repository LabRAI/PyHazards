from pyhazards.configs import load_experiment_config
from pyhazards.engine.runner import BenchmarkRunner


def test_wildfire_danger_vertical_slice(tmp_path):
    config = load_experiment_config("pyhazards/configs/wildfire/wildfire_danger_smoke.yaml")
    summary = BenchmarkRunner().run(config, output_dir=str(tmp_path))

    assert summary.benchmark_name == "wildfire"
    assert summary.hazard_task == "wildfire.danger"
    assert "accuracy" in summary.metrics


def test_wildfire_spread_vertical_slice(tmp_path):
    config = load_experiment_config("pyhazards/configs/wildfire/wildfire_spread_smoke.yaml")
    summary = BenchmarkRunner().run(config, output_dir=str(tmp_path))

    assert summary.hazard_task == "wildfire.spread"
    assert "iou" in summary.metrics
