from pyhazards.configs import load_experiment_config
from pyhazards.engine.runner import BenchmarkRunner


def test_flood_streamflow_vertical_slice(tmp_path):
    config = load_experiment_config("pyhazards/configs/flood/neuralhydrology_lstm_smoke.yaml")
    summary = BenchmarkRunner().run(config, output_dir=str(tmp_path))

    assert summary.benchmark_name == "flood"
    assert summary.hazard_task == "flood.streamflow"
    # NeuralHydrology's metrics per basin, summarised by the median (and mean) over basins.
    for metric in ("nse", "kge", "alpha_nse", "beta_nse", "fhv", "fms", "flv", "peak_timing", "nse_mean"):
        assert metric in summary.metrics
    assert summary.metrics["n_basins"] == len(summary.metadata["per_basin"])


def test_flood_mesh_vertical_slice(tmp_path):
    config = load_experiment_config("pyhazards/configs/flood/hydrographnet_smoke.yaml")
    summary = BenchmarkRunner().run(config, output_dir=str(tmp_path))

    assert summary.hazard_task == "flood.inundation"
    # HydroGraphNet rollouts on synthetic mesh hydrographs, scored in metres.
    assert {"rollout_rmse", "pixel_mae", "rmse", "iou", "f1", "csi_10cm"} <= set(summary.metrics)
    assert summary.metadata["rollout_rmse_per_step_m"]
