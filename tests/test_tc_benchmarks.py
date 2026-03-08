from pyhazards.configs import load_experiment_config
from pyhazards.engine.runner import BenchmarkRunner


def test_tc_vertical_slice(tmp_path):
    config = load_experiment_config("pyhazards/configs/tc/hurricast_smoke.yaml")
    summary = BenchmarkRunner().run(config, output_dir=str(tmp_path))

    assert summary.benchmark_name == "tc"
    assert summary.hazard_task == "tc.track_intensity"
    assert "track_error" in summary.metrics
    assert "intensity_mae" in summary.metrics
