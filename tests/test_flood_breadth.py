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


def test_inundation_models_forward():
    floodcast = build_model(name="floodcast", task="regression", in_channels=3, history=4)
    assert floodcast(torch.randn(2, 4, 3, 16, 16)).shape == (2, 1, 16, 16)
    # UrbanFloodCast: (batch, Sy, Sx, T, T_in, 5) -> depth and discharges at all T steps.
    urbanfloodcast = build_model(name="urbanfloodcast", task="regression")
    assert urbanfloodcast(torch.randn(1, 28, 28, 14, 1, 5)).shape == (1, 28, 28, 14, 3)
    hydrographnet = build_model(name="hydrographnet", task="regression")
    edge_index = torch.stack([torch.arange(8), (torch.arange(8) + 1) % 8])
    assert hydrographnet(torch.randn(8, 16), torch.randn(8, 3), edge_index).shape == (8, 2)


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
        for metric in ("iou", "pixel_mae", "rmse", "csi_10cm", "relative_l2", "nse"):
            assert metric in summary.metrics
