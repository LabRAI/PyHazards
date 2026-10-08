"""Streamflow task: hydrological metrics, CAMELS-US / Caravan readers on files in the real layout, evaluator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from pyhazards.benchmarks.flood import evaluate_streamflow
from pyhazards.datasets import available_datasets, load_dataset
from pyhazards.datasets.flood import (
    CamelsUSStreamflowDataset,
    CaravanStreamflowDataset,
    build_streamflow_bundle,
    load_camels_us_basin,
)
from pyhazards.metrics import hydrology


# -- metrics -----------------------------------------------------------------------------------------

def test_metric_definitions_on_known_series():
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert hydrology.nse(obs, obs) == 1.0
    assert hydrology.kge(obs, obs) == pytest.approx(1.0)
    assert hydrology.nse(obs, np.full(5, obs.mean())) == pytest.approx(0.0)
    assert hydrology.alpha_nse(obs, 2 * obs) == pytest.approx(2.0)
    assert hydrology.beta_kge(obs, obs + 3.0) == pytest.approx(2.0)
    assert hydrology.beta_nse(obs, obs + np.std(obs)) == pytest.approx(1.0)  # population std
    assert hydrology.rmse(obs, obs + 2.0) == pytest.approx(2.0)
    # KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2): doubling the flows gives alpha = beta = 2.
    assert hydrology.kge(obs, 2 * obs) == pytest.approx(1 - np.sqrt(2.0))


def test_metrics_skip_missing_values():
    obs = np.array([1.0, np.nan, 3.0, 4.0, 2.0])
    sim = np.array([1.0, 5.0, 3.0, np.nan, 2.0])
    assert hydrology.nse(obs, sim) == 1.0
    values = hydrology.calculate_metrics(np.full(4, np.nan), np.arange(4.0))
    assert set(values) == set(hydrology.STREAMFLOW_METRICS) and all(np.isnan(v) for v in values.values())
    with pytest.raises(ValueError, match="1-D"):
        hydrology.nse(np.ones((2, 2)), np.ones((2, 2)))
    with pytest.raises(ValueError, match="Shapes"):
        hydrology.nse(np.ones(3), np.ones(4))


def test_peak_timing_uses_dates():
    n = 400
    dates = pd.date_range("2000-01-01", periods=n, freq="1D").values
    obs = np.ones(n)
    obs[200] = 50.0
    sim = np.ones(n)
    sim[202] = 40.0  # peak two days late
    assert hydrology.mean_peak_timing(obs, sim, dates=dates) == 2.0
    assert hydrology.missed_peaks(obs, sim, dates=dates) == 1.0  # outside the 1-day window
    sim_gap = sim.copy()
    sim_gap[199] = np.nan  # a gap inside the window: the peak is skipped
    assert np.isnan(hydrology.mean_peak_timing(obs, sim_gap, dates=dates))


def test_aggregation_median_mean_and_counts():
    summary = hydrology.aggregate_basin_metrics(
        {"a": {"nse": 0.8}, "b": {"nse": -0.2}, "c": {"nse": 0.5}, "d": {"nse": float("nan")}}
    )
    assert summary["nse"] == pytest.approx(0.5)
    assert summary["nse_mean"] == pytest.approx(1.1 / 3)
    assert summary["n_basins"] == 4
    assert summary["n_basins_nse_le_0"] == 1
    assert set(hydrology.streamflow_metric_names()) >= {"nse", "nse_mean", "kge", "fhv_mean", "n_basins"}


# -- CAMELS-US reader --------------------------------------------------------------------------------

BASINS = ["01022500", "01031500"]
DYN = ["prcp(mm/day)", "srad(W/m2)", "tmax(C)", "tmin(C)", "vp(Pa)"]


def _write_camels(root: Path, days: int = 120, start: str = "2000-01-01") -> Path:
    """Two basins in the CAMELS-US layout (header, whitespace tables, HUC folders, ';' attribute files)."""
    rng = np.random.default_rng(0)
    dates = pd.date_range(start, periods=days, freq="1D")
    for i, basin in enumerate(BASINS):
        forcing_dir = root / "basin_mean_forcing" / "maurer_extended" / "01"
        flow_dir = root / "usgs_streamflow" / "01"
        forcing_dir.mkdir(parents=True, exist_ok=True)
        flow_dir.mkdir(parents=True, exist_ok=True)
        area = 573_623_053 + i * 1_000_000
        lines = ["44.82", "133.00", str(area), "Year\tMnth\tDay\tHr\tdayl(s)\tprcp(mm/day)\tsrad(W/m2)\tswe(mm)\ttmax(C)\ttmin(C)\tvp(Pa)"]
        for d in dates:
            row = [d.year, d.month, d.day, 12, 31185.94, *np.round(rng.gamma(1, 3, 1), 2), *np.round(rng.uniform(100, 300, 1), 2),
                   0.0, *np.round(rng.uniform(0, 20, 1), 2), *np.round(rng.uniform(-10, 0, 1), 2), *np.round(rng.uniform(100, 900, 1), 2)]
            lines.append("\t".join(str(v) for v in row))
        (forcing_dir / f"{basin}_lump_maurer_forcing_leap.txt").write_text("\n".join(lines) + "\n")
        flows = []
        for j, d in enumerate(dates):
            q = -999.0 if j == 40 else float(np.round(rng.uniform(50, 500), 2))
            flows.append(f"{basin} {d.year} {d.month:02d} {d.day:02d} {q:8.2f} A")
        (flow_dir / f"{basin}_streamflow_qc.txt").write_text("\n".join(flows) + "\n")
    attributes = root / "camels_attributes_v2.0"
    attributes.mkdir(parents=True)
    (attributes / "camels_clim.txt").write_text(
        "gauge_id;p_mean;aridity;high_prec_timing\n01022500;3.6;0.58;son\n01031500;3.5;0.62;son\n"
    )
    (attributes / "camels_topo.txt").write_text(
        "gauge_id;huc_02;elev_mean;area_gages2\n01022500;1;92.68;573.6\n01031500;1;443.25;298.4\n"
    )
    return root


def test_camels_us_reader_reads_the_official_layout(tmp_path):
    root = _write_camels(tmp_path / "CAMELS_US")
    frame = load_camels_us_basin(root, "01022500", ["maurer_extended"])
    raw = pd.read_csv(root / "usgs_streamflow" / "01" / "01022500_streamflow_qc.txt", sep=r"\s+", header=None)
    expected = 28316846.592 * raw.iloc[0, 4] * 86400 / (573_623_053 * 10**6)  # cfs -> mm/day
    assert frame["QObs(mm/d)"].iloc[0] == pytest.approx(expected)
    assert np.isnan(frame["QObs(mm/d)"].iloc[40])  # -999 flag (2000-02-10)
    assert set(DYN) <= set(frame.columns)

    bundle = CamelsUSStreamflowDataset(
        data_dir=root, basins=BASINS, static_attributes=["elev_mean", "p_mean", "aridity"], seq_length=20,
        periods={"train": ("2000-01-21", "2000-03-15"), "test": ("2000-03-16", "2000-04-29")},
    ).load()
    assert bundle.metadata["static_attributes"] == ["aridity", "elev_mean", "p_mean"]  # alphabetical, as NH
    test = bundle.get_split("test").inputs
    assert len(test) == 2 * 45  # one sample per basin and test day, warm-up taken from before the period
    assert pd.Timestamp(test.sample_date.min()) == pd.Timestamp("2000-03-16")
    inputs, y = test[0]
    assert inputs["x_d"].shape == (20, 5) and inputs["x_s"].shape == (3,) and y.shape == (20, 1)
    assert torch.isnan(y[:-1]).all()  # targets inside the warm-up are masked
    train = bundle.get_split("train").inputs
    assert len(train) == 2 * 55 - 2  # training drops the sample whose (only) target is missing
    with pytest.raises(ValueError, match="data_dir"):
        CamelsUSStreamflowDataset(basins=BASINS)
    with pytest.raises(ValueError, match="missing"):
        CamelsUSStreamflowDataset(data_dir=root, basins=BASINS, static_attributes=["no_such_attribute"],
                                  periods={"train": ("2000-01-21", "2000-03-15")}, seq_length=20).load()


# -- Caravan reader ----------------------------------------------------------------------------------

def _write_caravan(root: Path, days: int = 200) -> list:
    import xarray as xr

    rng = np.random.default_rng(1)
    dates = pd.date_range("1990-01-01", periods=days, freq="1D")
    basins = ["camelsgb_28015", "lamah_215"]
    variables = ["total_precipitation_sum", "temperature_2m_mean", "potential_evaporation_sum", "streamflow"]
    for basin in basins:
        source = basin.split("_")[0]
        values = {v: rng.gamma(1.0, 2.0, days).astype(np.float32) for v in variables}
        nc_dir = root / "timeseries" / "netcdf" / source
        csv_dir = root / "timeseries" / "csv" / source
        nc_dir.mkdir(parents=True, exist_ok=True)
        csv_dir.mkdir(parents=True, exist_ok=True)
        xr.Dataset({v: (["date"], a) for v, a in values.items()}, coords={"date": dates}).to_netcdf(nc_dir / f"{basin}.nc")
        pd.DataFrame({"date": dates, **values}).to_csv(csv_dir / f"{basin}.csv", index=False)
        attr_dir = root / "attributes" / source
        attr_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"gauge_id": [basin], "p_mean": [rng.uniform(1, 4)], "aridity": [rng.uniform(0.3, 2)]}).to_csv(
            attr_dir / f"attributes_caravan_{source}.csv", index=False)
        pd.DataFrame({"gauge_id": [basin], "ele_mt_sav": [rng.uniform(10, 900)]}).to_csv(
            attr_dir / f"attributes_hydroatlas_{source}.csv", index=False)
        pd.DataFrame({"gauge_id": [basin], "gauge_name": ["x"], "area": [rng.uniform(10, 900)]}).to_csv(
            attr_dir / f"attributes_other_{source}.csv", index=False)
    return basins


def test_caravan_reader_reads_netcdf_and_csv(tmp_path):
    basins = _write_caravan(tmp_path / "Caravan")
    kwargs = dict(
        data_dir=tmp_path / "Caravan", basins=basins, seq_length=30,
        dynamic_inputs=["total_precipitation_sum", "temperature_2m_mean", "potential_evaporation_sum"],
        static_attributes=["p_mean", "ele_mt_sav", "area"],
        periods={"train": ("1990-02-01", "1990-05-31"), "test": ("1990-06-01", "1990-07-19")},
    )
    from_nc = CaravanStreamflowDataset(**kwargs).load()
    from_csv = CaravanStreamflowDataset(filetype="csv", **kwargs).load()
    assert from_nc.metadata["static_attributes"] == ["area", "ele_mt_sav", "p_mean"]
    assert len(from_nc.get_split("test").inputs) == 2 * 49
    a, b = from_nc.get_split("test").inputs[3], from_csv.get_split("test").inputs[3]
    torch.testing.assert_close(a[0]["x_d"], b[0]["x_d"])
    torch.testing.assert_close(a[0]["x_s"], b[0]["x_s"])
    with pytest.raises(ValueError, match="dynamic_inputs"):
        CaravanStreamflowDataset(data_dir=tmp_path, basins=basins, periods=kwargs["periods"])
    with pytest.raises(ValueError, match="periods"):
        CaravanStreamflowDataset(data_dir=tmp_path, basins=basins, dynamic_inputs=["streamflow"])
    with pytest.raises(FileNotFoundError):
        CaravanStreamflowDataset(**{**kwargs, "basins": ["camelsgb_00000"]}).load()


# -- synthetic data, evaluator ------------------------------------------------------------------------

def test_registry_names_are_honest():
    names = set(available_datasets())
    assert {"camels_us_streamflow", "caravan_streamflow", "flood_streamflow_synthetic", "flood_mesh_synthetic"} <= names
    assert not {"waterbench_streamflow", "hydrobench_streamflow", "floodcastbench_inundation"} & names
    with pytest.raises(ValueError, match="data_dir"):
        load_dataset("caravan_streamflow")


class _Persistence(nn.Module):
    """Predicts the target scaler's centre plus the window's first forcing (a deterministic test model)."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        return {"y_hat": batch["x_d"][..., :1] * self.scale}


def test_evaluator_rescales_clips_and_scores_per_basin():
    bundle = load_dataset("flood_streamflow_synthetic", micro=True).load()
    model = _Persistence().eval()
    summary, per_basin = evaluate_streamflow(model, bundle, "test", clip_negative=True)
    windows = bundle.get_split("test").inputs
    center = np.array(bundle.metadata["scaler"]["target_center"])
    scale = np.array(bundle.metadata["scaler"]["target_scale"])
    expected = {}
    for b, basin in enumerate(windows.basins):
        rows = np.flatnonzero(windows.sample_basin == b)
        sim = np.array([windows[i][0]["x_d"][-1, 0].item() for i in rows]) * scale[0] + center[0]
        obs = np.array([windows[i][1][-1, 0].item() for i in rows]) * scale[0] + center[0]
        expected[basin] = hydrology.nse(obs, np.maximum(sim, 0.0))
    assert set(per_basin) == set(windows.basins)
    for basin, value in expected.items():
        assert per_basin[basin]["nse"] == pytest.approx(value, rel=1e-5)
    assert summary["nse"] == pytest.approx(float(np.median(list(expected.values()))), rel=1e-5)
    assert summary["n_basins"] == len(windows.basins)


def test_bundle_rejects_bad_periods_and_inputs():
    frame = pd.DataFrame({"p": np.arange(50.0), "q": np.arange(50.0)},
                         index=pd.date_range("2000-01-01", periods=50, freq="1D"))
    with pytest.raises(ValueError, match="train"):
        build_streamflow_bundle({"a": frame}, None, ["p"], [], ["q"], {"test": ("2000-01-10", "2000-02-01")}, 5)
    with pytest.raises(ValueError, match="not available"):
        build_streamflow_bundle({"a": frame}, None, ["x"], [], ["q"], {"train": ("2000-01-10", "2000-02-01")}, 5)
    with pytest.raises(ValueError, match="before it starts"):
        build_streamflow_bundle({"a": frame}, None, ["p"], [], ["q"], {"train": ("2000-02-10", "2000-02-01")}, 5)
