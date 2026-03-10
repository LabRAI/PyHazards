from pyhazards.configs import load_experiment_config
from pyhazards.engine.runner import BenchmarkRunner


def test_earthquake_vertical_slice(tmp_path):
    config = load_experiment_config("pyhazards/configs/earthquake/phasenet_smoke.yaml")
    summary = BenchmarkRunner().run(config, output_dir=str(tmp_path))

    assert summary.benchmark_name == "earthquake"
    assert summary.hazard_task == "earthquake.picking"
    assert "p_pick_mae" in summary.metrics
    assert "json" in summary.report_paths


def test_earthquake_forecasting_exports_pycsep_style_report(tmp_path):
    config = load_experiment_config("pyhazards/configs/earthquake/wavecastnet_benchmark_smoke.yaml")
    summary = BenchmarkRunner().run(config, output_dir=str(tmp_path))

    assert summary.hazard_task == "earthquake.forecasting"
    assert "pycsep" in summary.report_paths
