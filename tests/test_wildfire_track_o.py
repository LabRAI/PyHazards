"""Track-O wildfire occurrence datasets, cache builder and baseline runner (offline, synthetic fixtures)."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pyhazards.datasets import load_dataset
from pyhazards.datasets.wildfire.track_o_cache import (
    build_track_o_cache,
    date_from_name,
    grid_bounds,
    rasterize_points,
    read_daily_weather,
    read_firms_detections,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_wildfire_track_o_baselines", REPO_ROOT / "scripts" / "run_wildfire_track_o_baselines.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------------------------
# Synthetic source files
# ---------------------------------------------------------------------------------------------
LAT = np.arange(-10.0, 10.0 + 1e-9, 2.0)  # 11 cell centres
LON = np.arange(-20.0, 18.0 + 1e-9, 2.0)  # 20 cell centres


def _write_weather(directory: Path, day: str, value: float) -> None:
    """Two files per day, as MERRA-2 collections: hourly 'slv' (latitude stored north-to-south) and 'lnd'."""
    import xarray as xr

    stamp = day.replace("-", "")
    hours = np.arange(3)
    t2m = np.stack([np.full((LAT.size, LON.size), value + hour) for hour in hours])  # daily mean = value + 1
    t2m[:, :, 0] += np.arange(LAT.size)[None, :]  # a latitude-dependent column to check the orientation
    xr.Dataset(
        {"T2M": (("time", "lat", "lon"), t2m[:, ::-1, :])},
        coords={"time": hours, "lat": LAT[::-1], "lon": LON},
    ).to_netcdf(directory / f"MERRA2_400.tavg1_2d_slv_Nx.{stamp}.nc4")
    gwet = np.full((1, LAT.size, LON.size), 0.5)
    gwet[0, :, -2:] = np.nan  # undefined over an "ocean" strip
    xr.Dataset(
        {"GWETROOT": (("time", "lat", "lon"), gwet)},
        coords={"time": [0], "lat": LAT, "lon": LON},
    ).to_netcdf(directory / f"MERRA2_400.tavg1_2d_lnd_Nx.{stamp}.nc4")


def _write_firms(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["latitude", "longitude", "acq_date", "acq_time", "confidence", "type"])
        writer.writerows(rows)


@pytest.fixture()
def track_o_cache(tmp_path: Path) -> Path:
    weather = tmp_path / "weather"
    weather.mkdir()
    days = [f"2024-01-{d:02d}" for d in range(1, 13) if d != 9]  # 2024-01-09 has no weather
    for index, day in enumerate(days):
        _write_weather(weather, day, 280.0 + index)
    firms = tmp_path / "firms"
    firms.mkdir()
    _write_firms(
        firms / "fire_archive_M-C61_0001.csv",
        [
            [0.4, 0.9, "2024-01-01", "0130", 80, 0],  # -> cell (lat 0, lon 0)
            [9.9, -20.9, "2024-01-01", "0200", 90, 0],  # -> cell (lat 10, lon -20): within half a cell
            [3.0, 5.0, "2024-01-01", "0300", 90, 2],  # static land source: dropped
            [40.0, 0.0, "2024-01-01", "0400", 90, 0],  # far outside the grid: dropped
            [-4.2, 17.9, "2024-01-02", "1200", 50, 0],
            [-4.2, 17.9, "2024-01-12", "1200", 50, 0],
        ],
    )
    build_track_o_cache(
        tmp_path / "cache",
        sorted(weather.glob("*.nc4")),
        sorted(firms.glob("*.csv")),
        weather_vars=["T2M", "GWETROOT"],
        splits={"train": ("2024-01-01", "2024-01-06"), "val": ("2024-01-07", "2024-01-09"), "test": ("2024-01-10", "2024-01-12")},
    )
    return tmp_path / "cache"


# ---------------------------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------------------------
def test_date_from_name() -> None:
    assert date_from_name("pred_20240101_18.nc") == "2024-01-01"
    assert date_from_name("MERRA2_sfc_20241231.nc") == "2024-12-31"
    assert date_from_name("MERRA2_400.tavg1_2d_slv_Nx.20240229.nc4") == "2024-02-29"
    assert date_from_name("2024-03-05.csv") == "2024-03-05"
    assert date_from_name("fire_archive_M-C61_123456.csv") is None
    assert date_from_name("fire_archive_SV-C2_20241399.csv") is None  # not a calendar date


def test_rasterize_points_nearest_cell_wrap_and_extent() -> None:
    lat = np.array([-1.0, 0.0, 1.0])
    lon = np.arange(0.0, 360.0, 90.0)  # global 0..360 grid, 4 columns
    grid = rasterize_points(np.array([0.2, 0.9, 5.0]), np.array([-44.0, 179.0, 0.0]), lat, lon)
    expected = np.zeros((3, 4), dtype=np.float32)
    expected[1, 0] = 1.0  # lon -44 = 316 -> nearest centre 0 (wraps around)
    expected[2, 2] = 1.0  # lon 179 -> 180
    np.testing.assert_array_equal(grid, expected)  # lat 5.0 is outside the grid and dropped

    regional_lon = np.array([10.0, 11.0, 12.0])
    grid = rasterize_points(np.array([0.0, 0.0, 0.0]), np.array([9.6, 9.4, 12.6]), lat, regional_lon)
    assert grid.sum() == 1 and grid[1, 0] == 1  # 9.6 is within half a cell of 10; 9.4 and 12.6 are not


def test_read_daily_weather_means_and_orients(tmp_path: Path) -> None:
    _write_weather(tmp_path, "2024-05-01", 300.0)
    fields, lat, lon = read_daily_weather(sorted(tmp_path.glob("*.nc4")), ["T2M", "GWETROOT"])
    np.testing.assert_allclose(lat, LAT)
    np.testing.assert_allclose(lon, LON)
    assert fields.shape == (2, LAT.size, LON.size)
    np.testing.assert_allclose(fields[0, :, 1], 301.0)  # mean over the three hours
    np.testing.assert_allclose(fields[0, :, 0], 301.0 + np.arange(LAT.size))  # south-to-north rows
    assert np.isnan(fields[1, :, -2:]).all() and np.allclose(fields[1, :, :-2], 0.5)
    with pytest.raises(KeyError, match="LAI"):
        read_daily_weather(sorted(tmp_path.glob("*.nc4")), ["T2M", "LAI"])


def test_read_firms_detections_filters_type(tmp_path: Path) -> None:
    _write_firms(tmp_path / "a.csv", [[1.0, 2.0, "2024-01-01", "0000", 50, 0], [1.0, 2.0, "2024-01-01", "0000", 50, 1]])
    detections, stats = read_firms_detections([tmp_path / "a.csv"])
    assert list(detections) == ["2024-01-01"] and detections["2024-01-01"][0].tolist() == [1.0]
    assert stats["dropped_type"] == 1
    detections, _ = read_firms_detections([tmp_path / "a.csv"], firms_types=None)
    assert detections["2024-01-01"][0].size == 2
    # one file per day without acq_date: the day comes from the name
    (tmp_path / "2024-02-03.csv").write_text("latitude,longitude\n1.5,2.5\n", encoding="utf-8")
    detections, stats = read_firms_detections([tmp_path / "2024-02-03.csv"])
    assert list(detections) == ["2024-02-03"] and stats["files_without_type"] == 1


def test_build_cache_layout_labels_and_splits(track_o_cache: Path) -> None:
    summary = json.loads((track_o_cache / "cache_summary.json").read_text())
    assert summary["days"] == 11 and summary["splits"] == {"train": 6, "val": 2, "test": 3}
    assert summary["firms"]["dropped_type"] == 1
    assert (track_o_cache / "splits" / "val_dates.txt").read_text().split() == ["2024-01-07", "2024-01-08"]
    lat, lon = np.load(track_o_cache / "metadata" / "lat.npy"), np.load(track_o_cache / "metadata" / "lon.npy")
    np.testing.assert_allclose(lat, LAT)
    label = np.load(track_o_cache / "labels" / "2024-01-01.npy")
    expected = np.zeros((LAT.size, LON.size), dtype=np.float32)
    expected[5, 10] = 1.0  # (0.4, 0.9) -> lat 0, lon 0
    expected[10, 0] = 1.0  # (9.9, -20.9) -> lat 10, lon -20
    np.testing.assert_array_equal(label, expected)
    assert np.load(track_o_cache / "labels" / "2024-01-05.npy").sum() == 0  # covered day without fires
    met = np.load(track_o_cache / "met" / "2024-01-03.npy")
    np.testing.assert_allclose(met[0, :, 1], 283.0)
    assert grid_bounds(lat.astype(float), lon.astype(float)) == pytest.approx((-21.0, -11.0, 19.0, 11.0))
    assert not (track_o_cache / "static" / "fuel.npy").exists()


# ---------------------------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------------------------
def test_datasets_read_cache(track_o_cache: Path) -> None:
    raster = load_dataset("wildfire_track_o_raster", cache_dir=str(track_o_cache), downsample_factor=1, spatial_multiple=2).load()
    train = raster.get_split("train")
    assert tuple(train.inputs.shape) == (6, 2, 10, 20) and tuple(train.targets.shape) == (6, 1, 10, 20)
    assert torch.isfinite(train.inputs).all()  # NaN GWETROOT -> 0 after standardisation
    assert raster.metadata["normalization"]["fit_split"] == "train"
    assert raster.get_split("test").metadata["dates"] == ["2024-01-10", "2024-01-11", "2024-01-12"]

    temporal = load_dataset(
        "wildfire_track_o_temporal", cache_dir=str(track_o_cache), history=2, downsample_factor=1, spatial_multiple=2
    ).load()
    # val has 01-07 and 01-08 -> one window; test has 01-10..01-12 -> two (no 01-09 in the cache)
    assert temporal.get_split("val").metadata["dates"] == ["2024-01-08"]
    assert temporal.get_split("test").metadata["dates"] == ["2024-01-11", "2024-01-12"]
    assert tuple(temporal.get_split("test").inputs.shape) == (2, 2, 2, 10, 20)

    tabular = load_dataset("wildfire_track_o_tabular", cache_dir=str(track_o_cache), downsample_factor=2, spatial_multiple=2).load()
    split = tabular.get_split("train")
    assert split.targets.dtype == torch.int64 and tuple(split.inputs.shape) == (6 * 6 * 10, 6)
    assert tabular.feature_spec.extra["feature_names"][-2:] == ["sin_day_of_year", "cos_day_of_year"]


def test_dataset_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cache_dir"):
        load_dataset("wildfire_track_o_raster")
    with pytest.raises(FileNotFoundError, match="micro=True"):
        load_dataset("wildfire_track_o_raster", cache_dir=str(tmp_path / "missing")).load()
    with pytest.raises(ValueError, match="spatial_multiple"):
        load_dataset("wildfire_track_o_raster", micro=True, downsample_factor=8, spatial_multiple=16).load()


@pytest.mark.parametrize(
    "name,input_shape,target_shape,target_dtype",
    [
        ("wildfire_track_o_raster", (24, 5, 16, 24), (24, 1, 16, 24), torch.float32),
        ("wildfire_track_o_temporal", (19, 6, 5, 8, 12), (19, 1, 8, 12), torch.float32),
        ("wildfire_track_o_tabular", (24 * 96, 9), (24 * 96,), torch.int64),
    ],
)
def test_micro_layouts(name, input_shape, target_shape, target_dtype) -> None:
    bundle = load_dataset(name, micro=True).load()
    train = bundle.get_split("train")
    assert tuple(train.inputs.shape) == input_shape
    assert tuple(train.targets.shape) == target_shape and train.targets.dtype == target_dtype
    assert torch.isfinite(train.inputs).all()
    assert bundle.metadata["source_dataset"] == "micro_synthetic"


# ---------------------------------------------------------------------------------------------
# Benchmark and runner
# ---------------------------------------------------------------------------------------------
def test_danger_map_metrics_match_sklearn_and_check_shapes() -> None:
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    from pyhazards.benchmarks.wildfire import _binary_map_metrics

    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(3, 1, 6, 7, generator=generator)
    targets = (torch.rand(3, 1, 6, 7, generator=generator) > 0.8).float()
    metrics = _binary_map_metrics(logits, targets)
    probs, truth = torch.sigmoid(logits).flatten().numpy(), targets.flatten().numpy() > 0.5
    assert metrics["pr_auc"] == pytest.approx(average_precision_score(truth, probs))
    assert metrics["auc"] == pytest.approx(roc_auc_score(truth, probs))
    assert metrics["macro_f1"] == pytest.approx(f1_score(truth, probs >= 0.5, average="macro"))

    from pyhazards.benchmarks import run_benchmark
    from pyhazards.configs import BenchmarkConfig, DatasetRef, ExperimentConfig, ModelRef, ReportConfig

    bundle = load_dataset("wildfire_track_o_raster", micro=True).load()
    config = ExperimentConfig(
        benchmark=BenchmarkConfig(name="wildfire", hazard_task="wildfire.danger", params={"batch_size": 3}),
        dataset=DatasetRef(name="wildfire_track_o_raster"),
        model=ModelRef(name="cnn", task="segmentation"),
        report=ReportConfig(formats=[]),
    )
    two_channel = torch.nn.Conv2d(5, 2, 1)
    with pytest.raises(ValueError, match="one logit per cell"):
        run_benchmark("wildfire", two_channel, bundle, config, output_dir="unused")


def test_probability_diagnostics_and_day_to_day_change() -> None:
    from sklearn.metrics import brier_score_loss, log_loss

    runner = _load_runner()
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 2, 500), rng.random(500)
    scores = runner.probability_diagnostics(y, p)
    assert scores["brier"] == pytest.approx(brier_score_loss(y, p))
    assert scores["nll"] == pytest.approx(log_loss(y, p), rel=1e-6)
    perfect = runner.probability_diagnostics(np.array([0, 1, 1]), np.array([0.0, 1.0, 1.0]))
    assert perfect["ece"] == pytest.approx(0.0) and perfect["brier"] == 0.0

    maps = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 0.0]])
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-05"]  # the last pair is not consecutive
    assert runner.mean_day_to_day_change(maps, dates) == pytest.approx((0.25 + 0.5) / 2)
    assert np.isnan(runner.mean_day_to_day_change(maps[:1], dates[:1]))


def test_runner_end_to_end_on_micro_data(tmp_path: Path) -> None:
    runner = _load_runner()
    out = tmp_path / "runs"
    code = runner.main(
        [
            "--micro",
            "--output-dir",
            str(out),
            "--models",
            "logistic_regression,random_forest,convlstm",
            "--model-kwargs",
            json.dumps({"random_forest": {"n_estimators": 5}, "convlstm": {"hidden_dim": 4}}),
            "--max-epochs",
            "2",
            "--patience",
            "1",
            "--device",
            "cpu",
            "--n-jobs",
            "1",
        ]
    )
    assert code == 0
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary["models"]) == {"logistic_regression", "random_forest", "convlstm"}
    for name in summary["models"]:
        seed_dir = out / name / "seed_42"
        metrics = json.loads((seed_dir / "metrics.json").read_text())
        for key in ("accuracy", "macro_f1", "auc", "pr_auc", "brier", "nll", "ece", "mean_day_to_day_change"):
            assert np.isfinite(metrics["test"][key]), (name, key)
        assert (seed_dir / "history.csv").exists() and (seed_dir / "experiment_setting.json").exists()
        assert (seed_dir / "report" / "wildfire.json").exists()
    assert json.loads((out / "convlstm" / "seed_42" / "metrics.json").read_text())["train_unit"] == "epoch"


# ---------------------------------------------------------------------------------------------
# Fuel alignment (optional rasterio)
# ---------------------------------------------------------------------------------------------
def test_fuel_alignment_keeps_north_up(tmp_path: Path) -> None:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    from pyhazards.datasets.wildfire.track_o_cache import add_fuel_to_cache, align_fuel_to_grid

    # A projected (CONUS Albers, EPSG:5070) raster: code 7 in the north half, 3 in the south half,
    # -9999 fill in the western third under a nodata tag of 32767, as in LANDFIRE's LF2024 GeoTIFFs.
    height, width = 200, 300
    data = np.where(np.arange(height)[:, None] < height // 2, 7, 3).astype(np.int16) * np.ones((1, width), np.int16)
    data[:, :100] = -9999
    path = tmp_path / "fuel.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1, dtype="int16",
        crs="EPSG:5070", transform=from_origin(-1_500_000, 2_500_000, 10_000, 10_000), nodata=32767,
    ) as dst:
        dst.write(data, 1)
    lat = np.arange(20.25, 55.0, 0.5)
    lon = np.arange(-129.75, -60.0, 0.5)
    fuel, mask = align_fuel_to_grid(path, lat, lon)
    rows = np.flatnonzero(mask.any(axis=1))
    north_rows = np.flatnonzero((fuel == 7).any(axis=1))
    south_rows = np.flatnonzero((fuel == 3).any(axis=1))
    assert rows.size and north_rows.mean() > south_rows.mean()  # rows are stored south-to-north
    assert set(np.unique(fuel[mask > 0])) == {3, 7} and (fuel[mask == 0] == 0).all()
    # with the file's own nodata tag (32767) the -9999 fill takes part in the mode and wins in the
    # western third, where those cells are then marked invalid instead of getting a fuel code
    tagged, tagged_mask = align_fuel_to_grid(path, lat, lon, src_nodata=None)
    assert tagged_mask.sum() <= mask.sum() and set(np.unique(tagged[tagged_mask > 0])) <= {3, 7}

    cache = tmp_path / "cache"
    (cache / "metadata").mkdir(parents=True)
    np.save(cache / "metadata" / "lat.npy", lat)
    np.save(cache / "metadata" / "lon.npy", lon)
    info = add_fuel_to_cache(cache, path)
    assert info["codes"] == [3, 7] and np.array_equal(np.load(cache / "static" / "fuel.npy"), fuel)
