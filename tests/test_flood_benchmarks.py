from pyhazards.configs import load_experiment_config
from pyhazards.engine.runner import BenchmarkRunner


def test_flood_streamflow_vertical_slice(tmp_path):
    config = load_experiment_config("pyhazards/configs/flood/hydrographnet_smoke.yaml")
    summary = BenchmarkRunner().run(config, output_dir=str(tmp_path))

    assert summary.benchmark_name == "flood"
    assert summary.hazard_task == "flood.streamflow"
    assert "mae" in summary.metrics
    assert "rmse" in summary.metrics
    assert "nse" in summary.metrics
    assert "kge" in summary.metrics
