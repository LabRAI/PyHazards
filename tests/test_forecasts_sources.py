"""Forecast sources of pyhazards.forecasts (TCBench files, WeatherBench 2, earth2studio) on local fixtures."""

import importlib.util

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pyhazards.forecasts import (
    FOUNDATION_MODELS,
    TCBENCH_MODELS,
    WEATHERBENCH2_FORECASTS,
    read_tcbench_fields,
    read_tcbench_matched_tracks,
    run_earth2studio_forecast,
    standardize_fields,
    tcbench_path,
)
from pyhazards.forecasts.tcbench import download_tcbench_file, tcbench_url
from pyhazards.forecasts.weatherbench2 import weatherbench2_url


def test_tcbench_paths_follow_the_released_layout():
    init = "2023-07-16 12:00"
    assert tcbench_path("pangu", "fields", init) == (
        "neural_weather_models/panguweather/panguweather_2023.07.16-12h00_maxltd-120_timeres-6.nc"
    )
    assert tcbench_path("fourcastnet_v2", "unmatched", init) == (
        "unmatched_tracks/2023_fcnet/fcnet_2023.07.16-12h00m_maxltd-120_timeres-6.csv"
    )
    assert tcbench_path("aifs", "unmatched", "2023-01-01") == "unmatched_tracks/2023_aifs/AIFS_init-2023.01.01-00h00_max-lead-120.csv"
    assert tcbench_path("pangu", "matched") == "matched_tracks/2023_PANGU.csv"
    assert "0124d14d7f1f468096f46aec1e79696ab9c880c1" in tcbench_url("2023_IBTrACS.csv")
    with pytest.raises(ValueError, match="unknown TCBench model"):
        tcbench_path("graphcast", "matched")
    with pytest.raises(ValueError, match="3 GB"):
        download_tcbench_file(tcbench_path("pangu", "fields", init), "/nonexistent")
    assert set(TCBENCH_MODELS) == {"pangu", "fourcastnet_v2", "aifs"}


def test_read_tcbench_matched_tracks_units(tmp_path):
    path = tmp_path / "2023_PANGU.csv"
    path.write_text(
        "SID,Initial Time,Valid Time,wind max,pressure min,lat,lon\n"
        "2023005S18142,2023-01-09 12:00:00,2023-01-10 12:00:00,13.39205,98565.07,-36.75,176.5\n"
        "2023005S18142,2023-01-09 12:00:00,2023-01-10 18:00:00,13.1802,98512.62,-37.5,176.25\n"
    )
    table = read_tcbench_matched_tracks(path)
    assert table.columns.tolist() == ["SID", "init_time", "valid_time", "lead_hours", "lat", "lon", "wind_ms", "pres_pa"]
    assert table["lead_hours"].tolist() == [24.0, 30.0]
    assert table["wind_ms"].iloc[0] == pytest.approx(13.39205)
    converted = tmp_path / "converted.csv"
    converted.write_text(  # TCBench's track_matcher.py output: knots (x 1.94384) and hPa
        "SID,Initial Time,Valid Time,wind max,pressure min,lat,lon\n"
        f"2023005S18142,2023-01-09 12:00:00,2023-01-10 12:00:00,{13.39205 * 1.94384!r},985.6507,-36.75,176.5\n"
    )
    back = read_tcbench_matched_tracks(converted, units="kt_hpa")
    assert back["wind_ms"].iloc[0] == pytest.approx(13.39205, rel=1e-12)
    assert back["pres_pa"].iloc[0] == pytest.approx(98565.07)
    bad = tmp_path / "bad.csv"
    bad.write_text("SID,lat\nX,1\n")
    with pytest.raises(ValueError, match="matched-tracks"):
        read_tcbench_matched_tracks(bad)


def _write_tcbench_like_file(path):
    """A small netCDF4 file with the layout of TCBench's raw ai-models outputs (int16, scale/offset)."""
    import netCDF4

    rng = np.random.default_rng(0)
    lat = np.array([20.0, 10.0, 0.0, -10.0, -20.0])
    lon = np.arange(0.0, 360.0, 45.0)
    levels = np.array([1000, 850, 500, 300])
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("leadtime_hours", 3)
        ds.createDimension("latitude", lat.size)
        ds.createDimension("longitude", lon.size)
        ds.createDimension("level", levels.size)
        ds.createVariable("time", "i8", ("time",))[:] = [0]
        ds.createVariable("leadtime_hours", "i8", ("leadtime_hours",))[:] = [6, 12, 18]
        ds.createVariable("latitude", "f8", ("latitude",))[:] = lat
        ds.createVariable("longitude", "f8", ("longitude",))[:] = lon
        ds.createVariable("level", "i8", ("level",))[:] = levels
        packed = {}
        for name, dims, scale, offset in (
            ("msl", ("time", "leadtime_hours", "latitude", "longitude"), 1.1892572, 100941.74),
            ("z", ("time", "leadtime_hours", "level", "latitude", "longitude"), 59.82942969, 78446.516),
        ):
            var = ds.createVariable(name, "i2", dims, fill_value=np.int16(-32767))
            var.set_auto_maskandscale(False)
            var.scale_factor = np.float64(scale)
            var.add_offset = np.float32(offset)
            shape = tuple(ds.dimensions[d].size for d in dims)
            values = rng.integers(-30000, 30000, size=shape, dtype=np.int16)
            values.flat[0] = -32767
            var[:] = values
            packed[name] = (values, np.float64(scale), float(np.float32(offset)))
    return lat, lon, levels, packed


def test_read_tcbench_fields_unpacks_selected_levels(tmp_path):
    path = tmp_path / "panguweather_2023.07.16-12h00_maxltd-120_timeres-6.nc"
    lat, lon, levels, packed = _write_tcbench_like_file(path)
    ds = read_tcbench_fields("pangu", "2023-07-16 12:00", variables=("msl", "z500"), path=path)
    assert list(ds.data_vars) == ["msl", "z500"] and ds["msl"].dtype == np.float32
    assert ds["time"].values[0] == np.datetime64("2023-07-16T18:00")
    np.testing.assert_array_equal(ds["lat"].values, lat)
    raw, scale, offset = packed["z"]
    expected = (raw[0, :, list(levels).index(500)].astype(np.float64) * scale + offset).astype(np.float32)
    np.testing.assert_array_equal(ds["z500"].values, expected)
    raw, scale, offset = packed["msl"]
    assert np.isnan(ds["msl"].values[0, 0, 0])  # the fill value
    np.testing.assert_array_equal(ds["msl"].values[0, 0, 1:], (raw[0, 0, 0, 1:].astype(np.float64) * scale + offset).astype(np.float32))
    with pytest.raises(KeyError, match="level 200"):
        read_tcbench_fields("pangu", "2023-07-16 12:00", variables=("z200",), path=path)
    with pytest.raises(ValueError, match="cannot map"):
        read_tcbench_fields("pangu", "2023-07-16 12:00", variables=("vorticity",), path=path)


def test_standardize_fields_accepts_common_layouts():
    time = pd.date_range("2020-01-01", periods=2, freq="6h")
    lat, lon = np.array([10.0, 0.0]), np.array([0.0, 90.0, 180.0, 270.0])
    wb2 = xr.Dataset(
        {
            "mean_sea_level_pressure": (("time", "latitude", "longitude"), np.ones((2, 2, 4))),
            "u_component_of_wind": (("time", "level", "latitude", "longitude"), np.ones((2, 2, 2, 4))),
        },
        coords={"time": time, "latitude": lat, "longitude": lon, "level": [500, 850]},
    )
    out = standardize_fields(wb2)
    assert sorted(out.data_vars) == ["msl", "u500", "u850"] and out["msl"].dims == ("time", "lat", "lon")
    e2s = xr.Dataset({"u10m": (("valid_time", "lat", "lon"), np.ones((2, 2, 4)))}, coords={"valid_time": time, "lat": lat, "lon": lon})
    assert list(standardize_fields(e2s, variables=["u10"]).data_vars) == ["u10"]
    with pytest.raises(KeyError, match="lacks"):
        standardize_fields(e2s, variables=["msl"])


def test_weatherbench2_store_lookup():
    assert weatherbench2_url("graphcast", "2018-09-30T00").endswith("graphcast/2018/date_range_2017-11-16_2019-02-01_12_hours.zarr")
    assert weatherbench2_url("graphcast", "2019-01-15T00").endswith("graphcast/2018/date_range_2017-11-16_2019-02-01_12_hours.zarr")
    assert weatherbench2_url("pangu", "2021-08-01").endswith("pangu/2018-2022_0012_0p25.zarr")
    assert weatherbench2_url("pangu_hres_init", "2021-08-01").endswith("pangu_hres_init/2021_0012_0p25.zarr")
    with pytest.raises(ValueError, match="no WeatherBench 2 graphcast store"):
        weatherbench2_url("graphcast", "2022-06-01")
    assert "fourcastnet" not in WEATHERBENCH2_FORECASTS


@pytest.mark.skipif(importlib.util.find_spec("zarr") is None, reason="zarr is an optional dependency (pyhazards[weather])")
def test_weatherbench2_reader_on_a_local_store(tmp_path):
    from pyhazards.forecasts import read_weatherbench2_forecast

    init = pd.Timestamp("2018-09-30T00")
    lat, lon = np.linspace(90, -90, 7), np.arange(0, 360, 60.0)
    store = xr.Dataset(
        {
            "mean_sea_level_pressure": (("time", "prediction_timedelta", "latitude", "longitude"), np.arange(2 * 3 * 7 * 6, dtype=np.float32).reshape(2, 3, 7, 6)),
            "geopotential": (("time", "prediction_timedelta", "level", "latitude", "longitude"), np.ones((2, 3, 2, 7, 6), dtype=np.float32)),
        },
        coords={"time": [init, init + pd.Timedelta(hours=12)], "prediction_timedelta": np.array([6, 12, 18]), "level": [300, 500], "latitude": lat, "longitude": lon},
    )
    store["prediction_timedelta"].attrs["units"] = "hours"
    path = tmp_path / "pangu.zarr"
    store.to_zarr(path, consolidated=True)
    ds = read_weatherbench2_forecast("pangu", init, variables=("msl", "z300"), lead_hours=[6, 18], url=str(path), lon_bounds=(300, 60))
    assert ds.sizes == {"time": 2, "lat": 7, "lon": 3}
    assert ds["lon"].values.tolist() == [0.0, 60.0, 300.0]
    np.testing.assert_array_equal(ds["msl"].values[1], store["mean_sea_level_pressure"].values[0, 2][:, [0, 1, 5]])
    with pytest.raises(KeyError, match="lead times"):
        read_weatherbench2_forecast("pangu", init, variables=("msl",), lead_hours=[24], url=str(path))


def test_earth2studio_runner_reports_missing_dependency():
    if importlib.util.find_spec("earth2studio") is not None:
        pytest.skip("earth2studio is installed; covered by tests/oracle/test_forecasts_weather_oracle.py")
    with pytest.raises(ImportError, match="earth2studio"):
        run_earth2studio_forecast(object(), object(), "2020-01-01")


def test_foundation_model_catalog_states_licences():
    assert set(FOUNDATION_MODELS) == {"fourcastnet", "graphcast", "pangu_weather"}
    pangu = FOUNDATION_MODELS["pangu_weather"]
    assert "NC" in pangu.weights_license and "commercial" in pangu.weights_license
    assert "120.29" in pangu.paper_tc_evaluation
    assert FOUNDATION_MODELS["graphcast"].code_license == "Apache-2.0"
    assert "BSD-3" in FOUNDATION_MODELS["fourcastnet"].code_license
