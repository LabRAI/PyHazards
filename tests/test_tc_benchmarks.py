import math

import pytest
import torch

from pyhazards.benchmarks.tc import (
    great_circle_distance_km,
    intensity_metrics,
    tcn_equirectangular_distance_km,
    track_intensity_metrics,
)
from pyhazards.configs import load_experiment_config
from pyhazards.datasets.base import DataBundle, DataSplit, FeatureSpec, LabelSpec
from pyhazards.engine.runner import BenchmarkRunner

TC_CONFIGS = [
    "hurricast_smoke",
    "tropicalcyclone_mlp_smoke",
    "tropicyclonenet_smoke",
    "saf_net_smoke",
    "tcif_fusion_smoke",
]


def test_great_circle_distance_in_km():
    one_degree = 2 * math.pi * 6371.0 / 360
    assert float(great_circle_distance_km(10.0, 120.0, 11.0, 120.0)) == pytest.approx(one_degree, rel=1e-9)
    # Across the dateline, with either longitude convention.
    assert float(great_circle_distance_km(0.0, 179.5, 0.0, -179.5)) == pytest.approx(one_degree, rel=1e-9)
    assert float(great_circle_distance_km(0.0, 179.5, 0.0, 180.5)) == pytest.approx(one_degree, rel=1e-9)
    # A degree of longitude shrinks with the cosine of latitude.
    assert float(great_circle_distance_km(60.0, 0.0, 60.0, 1.0)) == pytest.approx(one_degree * 0.5, rel=1e-3)
    # TropiCycloneNet's formula: 111 km per degree, longitude scaled by cos(observed latitude).
    assert float(tcn_equirectangular_distance_km(11.0, 120.0, 10.0, 120.0)) == pytest.approx(111.0)
    assert float(tcn_equirectangular_distance_km(60.0, 1.0, 60.0, 0.0)) == pytest.approx(55.5)


def test_best_of_k_is_per_lead_and_per_variable():
    target = {"lat": torch.zeros(1, 2, dtype=torch.float64), "lon": torch.zeros(1, 2, dtype=torch.float64), "wind": torch.full((1, 2), 50.0, dtype=torch.float64)}
    # Sample 0 has the better track, sample 1 the better wind.
    forecast = {
        "lat": torch.tensor([[[0.1, 0.2], [1.0, 2.0]]], dtype=torch.float64),
        "lon": torch.zeros(1, 2, 2, dtype=torch.float64),
        "wind": torch.tensor([[[40.0, 30.0], [49.0, 52.0]]], dtype=torch.float64),
    }
    metrics = track_intensity_metrics(forecast, target, [12, 24])
    km = 2 * math.pi * 6371.0 / 360
    assert metrics["best_of_k_track_error_km_12h"] == pytest.approx(0.1 * km, rel=1e-6)
    assert metrics["best_of_k_track_error_km_24h"] == pytest.approx(0.2 * km, rel=1e-6)
    assert metrics["best_of_k_intensity_mae_12h"] == pytest.approx(1.0)
    assert metrics["best_of_k_intensity_mae_24h"] == pytest.approx(2.0)
    # Deterministic metrics score the sample mean.
    assert metrics["intensity_mae_12h"] == pytest.approx(5.5)
    assert metrics["track_error_km_24h"] == pytest.approx(1.1 * km, rel=1e-3)
    assert metrics["num_samples"] == 2


def test_nan_targets_are_skipped():
    target = {"lat": torch.tensor([[0.0], [float("nan")]]), "lon": torch.zeros(2, 1), "pres": torch.tensor([[1000.0], [990.0]])}
    forecast = {"lat": torch.tensor([[1.0], [5.0]]), "lon": torch.zeros(2, 1), "pres": torch.tensor([[1002.0], [995.0]])}
    metrics = track_intensity_metrics(forecast, target, [24])
    assert metrics["track_error_km_24h"] == pytest.approx(2 * math.pi * 6371.0 / 360, rel=1e-6)
    assert metrics["pressure_mae"] == pytest.approx(3.5)


def test_intensity_metrics_and_yearly_groups():
    metrics = intensity_metrics(torch.tensor([1.0, 2.0, 3.0, 10.0]), torch.tensor([0.0, 0.0, 0.0, 0.0]), [24], [2015, 2015, 2016, 2016])
    assert metrics["intensity_mae"] == pytest.approx(4.0)
    assert metrics["intensity_mae_24h"] == pytest.approx(4.0)
    assert metrics["intensity_mae_2015"] == pytest.approx(1.5)
    assert metrics["intensity_mae_2016"] == pytest.approx(6.5)
    assert metrics["intensity_mae_group_mean"] == pytest.approx(4.0)
    assert metrics["intensity_rmse"] == pytest.approx(math.sqrt((1 + 4 + 9 + 100) / 4))


class _Constant(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full((x.size(0), 1), self.value)


def test_prediction_transform_maps_scaled_outputs_to_units(tmp_path):
    config = load_experiment_config("pyhazards/configs/tc/saf_net_smoke.yaml")
    split = DataSplit(torch.zeros(3, 4), torch.tensor([20.0, 30.0, 40.0]), metadata={"groups": [2018] * 3})
    bundle = DataBundle(
        splits={"test": split},
        feature_spec=FeatureSpec(),
        label_spec=LabelSpec(),
        metadata={"lead_hours": [24], "units": {"wind": "m/s"}, "prediction_transform": {"kind": "minmax", "min": 10.0, "max": 70.0}},
    )
    summary = BenchmarkRunner().run(config, model=_Constant(0.5), data=bundle, output_dir=str(tmp_path))
    assert summary.metrics["intensity_mae"] == pytest.approx(abs(40 - 20) / 3 + abs(40 - 30) / 3 + 0.0)
    assert summary.metadata["units"]["intensity_mae"] == "m/s"


@pytest.mark.parametrize("name", TC_CONFIGS)
def test_tc_smoke_configs(name, tmp_path):
    config = load_experiment_config(f"pyhazards/configs/tc/{name}.yaml")
    summary = BenchmarkRunner().run(config, output_dir=str(tmp_path))
    assert summary.benchmark_name == "tc"
    for metric in config.benchmark.metrics:
        assert metric in summary.metrics, metric
    assert summary.metadata["synthetic"] is True
    if config.benchmark.hazard_task == "tc.track_intensity":
        assert summary.metadata["units"]["track_error_km"] == "km"
        assert summary.metadata["lead_hours"]


class _Persistence(torch.nn.Module):
    """Deterministic stand-in with a forecast() method: keeps the last observed TCND state."""

    def forecast(self, batch):
        last = batch["obs_traj"][-1]  # (B, 4) normalised; time-first input
        steps = torch.ones(1, 4)
        return {
            "lon": (last[:, 0:1] * 5 + 180) * steps,
            "lat": (last[:, 1:2] * 5) * steps,
            "pres": (last[:, 2:3] * 50 + 960) * steps,
            "wind": (last[:, 3:4] * 25 + 40) * steps,
        }


def test_batched_scoring_matches_one_pass(tmp_path):
    from pyhazards.datasets import load_dataset
    from pyhazards.models import build_model

    config = load_experiment_config("pyhazards/configs/tc/saf_net_smoke.yaml")
    data = load_dataset("safnet_cma_era_interim_synthetic", samples=20).load()
    torch.manual_seed(0)
    model = build_model("saf_net", task="regression")
    whole = BenchmarkRunner().run(config, model=model, data=data, output_dir=str(tmp_path))
    config.benchmark.params = {"batch_size": 2}
    chunked = BenchmarkRunner().run(config, model=model, data=data, output_dir=str(tmp_path))
    assert whole.metrics == pytest.approx(chunked.metrics)

    # Time-first TCND inputs are sliced along their batch axis (metadata["batch_dims"]).
    config = load_experiment_config("pyhazards/configs/tc/tropicyclonenet_smoke.yaml")
    data = load_dataset("tropicyclonenet_dataset_synthetic", samples=30).load()
    whole = BenchmarkRunner().run(config, model=_Persistence(), data=data, output_dir=str(tmp_path))
    config.benchmark.params = {"batch_size": 2}
    chunked = BenchmarkRunner().run(config, model=_Persistence(), data=data, output_dir=str(tmp_path))
    assert whole.metrics == pytest.approx(chunked.metrics)
    assert whole.metrics["track_error_km_24h"] > 0
