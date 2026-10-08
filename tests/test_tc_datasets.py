import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from pyhazards.datasets import available_datasets, load_dataset
from pyhazards.datasets.tc import build_track_samples, read_ibtracs, read_tcnd_track
from pyhazards.datasets.tc.tcnd import GPH_RANGE, gph_to_image
from pyhazards.models.tropicalcyclone_mlp import SHIPS_PREDICTORS

FIXTURES = Path(__file__).resolve().parent / "fixtures"
IBTRACS_CSV = FIXTURES / "ibtracs" / "ibtracs.fixture.list.v04r01.csv"
IBTRACS_NC = FIXTURES / "ibtracs" / "IBTrACS.fixture.v04r01.nc"
SHIPS_CSV = FIXTURES / "ships_xu2021" / "train_global_fill_REA_na_wo_img_scaled.fixture.csv"
ERA5_NC = FIXTURES / "era5" / "era5.pressure_levels.ophelia_2023092200_2023092300.fixture.nc"
TCND = FIXTURES / "tcnd"


def _raw_rows(sid):
    with IBTRACS_CSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows[1:] if row["SID"] == sid]  # rows[0] is the units row


def test_ibtracs_csv_layout_is_parsed():
    table = read_ibtracs(IBTRACS_CSV)
    assert set(table["SID"]) == {"2023265N29284", "2026058S18168", "2023266N16116", "2025214N26167"}
    assert set(table["BASIN"]) == {"NA", "WP", "SP"}  # "NA" (North Atlantic) is not a missing value
    assert table["LAT"].dtype.kind == "f" and table["USA_WIND"].dtype.kind == "f"
    assert table["ISO_TIME"].dtype.kind == "M"
    assert table.loc[table["NAME"] == "OPHELIA", "SEASON"].iloc[0] == 2023
    urmil = table[table["SID"] == "2026058S18168"]
    assert urmil["LON"].max() > 180  # merged positions stay continuous across the dateline
    assert urmil["USA_LON"].max() < 180 and urmil["USA_LON"].min() < 0  # agency positions wrap
    assert np.isnan(table["WMO_WIND"]).any()  # blanks (" ") are missing values
    assert "US-PROVISIONAL_spur" in set(table["TRACK_TYPE"])


def test_ibtracs_netcdf_matches_csv():
    csv_table = read_ibtracs(IBTRACS_CSV)
    nc_table = read_ibtracs(IBTRACS_NC)
    assert len(nc_table) == len(csv_table)
    for column in ("SID", "BASIN", "NAME", "NATURE", "TRACK_TYPE", "WMO_AGENCY", "USA_AGENCY"):
        assert list(nc_table[column]) == list(csv_table[column]), column
    assert (nc_table["ISO_TIME"].values == csv_table["ISO_TIME"].values).all()
    for column in ("LAT", "LON", "WMO_WIND", "WMO_PRES", "USA_WIND", "USA_PRES", "USA_LON", "CMA_WIND", "TOKYO_PRES", "DIST2LAND", "STORM_SPEED"):
        np.testing.assert_allclose(nc_table[column].to_numpy(float), csv_table[column].to_numpy(float), atol=1e-4, equal_nan=True, err_msg=column)


def test_track_windows_follow_the_raw_rows():
    table = read_ibtracs(IBTRACS_CSV)
    x, y, meta = build_track_samples(table, variables=("lat", "lon", "wind", "pres"), history=4, lead_hours=(6, 12, 18, 24))
    assert x.shape[1:] == (4, 4) and y.shape[1:] == (4, 4)
    assert "2025214N26167" not in meta["sid"]  # spur tracks are skipped by default
    # First OPHELIA window, recomputed from the raw CSV rows at synoptic hours.
    rows = [r for r in _raw_rows("2023265N29284") if r["ISO_TIME"][11:13] in {"00", "06", "12", "18"}]
    values = [[float(r[c]) if r[c].strip() else np.nan for c in ("LAT", "LON", "USA_WIND", "USA_PRES")] for r in rows]
    first = next(i for i in range(3, len(values) - 4) if not np.isnan(values[i - 3 : i + 5]).any())
    k = meta["sid"].index("2023265N29284")
    np.testing.assert_allclose(x[k], values[first - 3 : first + 1], rtol=1e-6)
    np.testing.assert_allclose(y[k], values[first + 1 : first + 5], rtol=1e-6)
    assert meta["iso_time"][k] == rows[first]["ISO_TIME"].replace(" ", "T")

    # Agency positions are unwrapped inside a window, like the merged ones.
    x_usa, y_usa, meta_usa = build_track_samples(table, variables=("lat", "lon", "wind"), positions="agency", history=2, lead_hours=(24,))
    urmil = [i for i, sid in enumerate(meta_usa["sid"]) if sid == "2026058S18168"]
    lons = np.concatenate([x_usa[urmil][..., 1], y_usa[urmil][..., 1]], axis=1)
    assert np.abs(np.diff(lons, axis=1)).max() < 90  # no 360-degree jump at the dateline
    assert lons.max() > 180 and table.loc[table["SID"] == "2026058S18168", "USA_LON"].min() < -170
    with pytest.raises(ValueError, match="multiples of 6"):
        build_track_samples(table, lead_hours=(3,))
    with pytest.raises(ValueError, match="WMO has no position"):
        build_track_samples(table, agency="wmo", positions="agency")


def test_ibtracs_dataset_bundle(tmp_path):
    bundle = load_dataset("ibtracs_tracks", path=str(IBTRACS_NC), variables=["lat", "lon", "wind"], history=2, lead_hours=[12, 24], test_seasons=[2026]).load()
    assert bundle.metadata["units"] == {"lat": "degrees_north", "lon": "degrees_east", "wind": "kt"}
    assert bundle.metadata["lead_hours"] == [12, 24] and bundle.metadata["synthetic"] is False
    test = bundle.splits["test"]
    assert set(test.metadata["season"]) == {2026} and test.inputs.shape[1:] == (2, 3) and test.targets.shape[1:] == (2, 3)
    assert set(bundle.splits["train"].metadata["season"]) == {2023}
    basin = load_dataset("ibtracs_tracks", path=str(IBTRACS_CSV), variables=["lat", "lon"], basins=["NA"]).load()
    for split in basin.splits.values():
        assert set(split.metadata["basin"]) <= {"NA"}
    with pytest.raises(ValueError, match="needs path"):
        load_dataset("ibtracs_tracks")
    with pytest.raises(FileNotFoundError, match="download=True"):
        load_dataset("ibtracs_tracks", cache_dir=str(tmp_path))


def test_ships_xu2021_loyo_split():
    bundle = load_dataset("ships_xu2021", path=str(SHIPS_CSV), leave_out_year=2018, val_fraction=0.25, seed=0).load()
    train, val, test = (bundle.splits[name] for name in ("train", "val", "test"))
    assert test.inputs.shape == (3, 121) and set(test.metadata["groups"]) == {2018}
    # Training uses reanalysis rows only (AL 2016 and EP 1982), never operational ones.
    assert len(train.targets) + len(val.targets) == 5
    assert set(train.metadata["basin"]) | set(val.metadata["basin"]) == {"AL", "EP"}
    import pandas as pd

    raw = pd.read_csv(SHIPS_CSV)
    expected = raw[(raw.year == 2018) & (raw.type == "opr")]
    np.testing.assert_allclose(test.inputs.numpy(), expected[list(SHIPS_PREDICTORS)].to_numpy(np.float32))
    np.testing.assert_allclose(test.targets.numpy(), expected["dvs24"].to_numpy(np.float32))
    assert bundle.metadata["units"] == {"wind": "kt"} and bundle.metadata["hazard_task"] == "tc.intensity"
    fold_2011 = load_dataset("ships_xu2021", path=str(SHIPS_CSV), leave_out_year=2011, val_fraction=0.0).load()
    assert len(fold_2011.splits["test"].targets) == 2 and len(fold_2011.splits["train"].targets) == 5


def _write_gph(root: Path):
    """GPH crops in the official layout (100 x 100 float64 geopotential, m^2 s^-2)."""
    rng = np.random.default_rng(0)
    _, dates, _ = read_tcnd_track(TCND / "BST_data" / "EP" / "test" / "EP2018BSTCARLOTTA.txt")
    storm = root / "EP" / "2018" / "CARLOTTA"
    storm.mkdir(parents=True)
    for date in dates:
        np.save(storm / f"{date}.npy", rng.uniform(55000.0, 59500.0, size=(100, 100)))
    return root


def test_tcnd_reader(tmp_path):
    gph = _write_gph(tmp_path / "ERA5_gph500")
    bundle = load_dataset(
        "tropicyclonenet_dataset", data1d_dir=str(TCND / "BST_data"), env_dir=str(TCND / "Env_data"), gph_dir=str(gph), areas=["EP"], splits=["test"]
    ).load()
    test = bundle.splits["test"]
    values, dates, names = read_tcnd_track(TCND / "BST_data" / "EP" / "test" / "EP2018BSTCARLOTTA.txt")
    assert len(values) == 18 and set(names) == {"CARLOTTA"}
    n = 18 - 12 + 1
    assert test.inputs["obs_traj"].shape == (8, n, 4) and test.inputs["image_obs"].shape == (n, 1, 8, 64, 64)
    assert test.targets.shape == (n, 4, 4)
    torch.testing.assert_close(test.inputs["obs_traj"][:, 0], torch.as_tensor(values[:8, 2:6], dtype=torch.float32))
    assert torch.all(test.inputs["obs_traj_rel"][0] == 0)
    # Targets in physical units: lat = 5 * y, lon = 5 * x + 180, pres = 50 * p + 960, wind = 25 * w + 40.
    future = values[8:12, 2:6]
    expected = np.stack([5 * future[:, 1], 5 * future[:, 0] + 180, 50 * future[:, 2] + 960, 25 * future[:, 3] + 40], axis=-1)
    np.testing.assert_allclose(test.targets[0].numpy(), expected, rtol=1e-6)
    # Env-Data: the 12/24-hour direction history is -1 at the first steps and filled from later ones.
    direction24 = test.inputs["env_data"]["history_direction24"][0]
    assert direction24.shape == (8, 8) and torch.all(direction24.sum(dim=-1) == 1)
    assert torch.all((test.inputs["image_obs"] >= 0) & (test.inputs["image_obs"] <= 1))
    assert bundle.metadata["units"] == {"wind": "m/s", "pres": "hPa"}
    with pytest.raises(FileNotFoundError):
        load_dataset("tropicyclonenet_dataset", root=str(tmp_path / "missing"))


def test_gph_scaling():
    low, high = GPH_RANGE
    image = gph_to_image(np.full((100, 100), (low + high) / 2))
    assert image.shape == (64, 64) and np.allclose(image, 0.5)
    assert gph_to_image(np.full((100, 100), high + 1000)).max() == 1.0


def test_synthetic_datasets_are_labelled_synthetic():
    names = [
        "tc_tracks_synthetic",
        "ships_xu2021_synthetic",
        "safnet_cma_era_interim_synthetic",
        "tropicyclonenet_dataset_synthetic",
        "hurricast_synthetic",
        "tcif_fusion_synthetic",
    ]
    assert set(names) <= set(available_datasets())
    assert not {"tcbench_alpha", "tropicyclonenet_dataset_alias"} & set(available_datasets())
    for name in names:
        bundle = load_dataset(name, micro=True).load()
        assert bundle.metadata["synthetic"] is True and bundle.metadata["dataset"] == name
    tcnd = load_dataset("tropicyclonenet_dataset_synthetic", micro=True).load().splits["test"]
    assert tcnd.inputs["image_obs"].shape[1:] == (1, 8, 64, 64) and tcnd.targets.shape[1:] == (4, 4)


def test_hurricast_statistics_and_windows():
    from pyhazards.datasets.tc.hurricast import hurricast_storms, hurricast_windows, wind_category
    from pyhazards.models.hurricast import HURRICAST_STAT_FEATURES as NAMES

    table = read_ibtracs(IBTRACS_CSV)
    storms = {s["sid"]: s for s in hurricast_storms(table, min_wind=34, min_steps=5)}
    assert set(storms) == {"2023265N29284", "2026058S18168"}  # the WP storms never reach 34 kt
    assert hurricast_storms(table) == []  # the paper's selection needs more than 20 steps of 34 kt
    ophelia = storms["2023265N29284"]
    times = ophelia["times"].astype("datetime64[m]").astype(str)
    assert times[0] == "2023-09-22T00:00" and "2023-09-23T10:15" not in times  # first 34 kt; off-grid row dropped
    assert np.all(np.diff(ophelia["times"]).astype("timedelta64[h]") == np.timedelta64(3, "h"))
    f = ophelia["features"]
    assert np.isfinite(f).all() and f.shape == (23, 30)
    rows = [r for r in _raw_rows("2023265N29284") if r["ISO_TIME"] >= "2023-09-22 00:00:00" and r["ISO_TIME"][14:16] == "00"]
    np.testing.assert_allclose(f[:, NAMES.index("LAT")], [float(r["LAT"]) for r in rows])
    np.testing.assert_allclose(f[1, NAMES.index("WMO_WIND")], (35 + 40) / 2)  # interpolated 03 UTC wind
    np.testing.assert_allclose(f[1:, NAMES.index("STORM_DISPLACEMENT_X")], np.diff(f[:, NAMES.index("LAT")]))
    assert f[0, NAMES.index("STORM_DISPLACEMENT_X")] == 0
    assert np.all(f[:, NAMES.index("cat_basin_AN")] == 1) and f[:, NAMES.index("cat_nature_TS")].sum() > 0
    np.testing.assert_allclose(f[:, NAMES.index("cat_storm_category")], wind_category(f[:, NAMES.index("WMO_WIND")]))
    assert list(wind_category([33, 34, 64, 83, 96, 113, 137, np.nan])) == [0, 1, 2, 3, 4, 5, 6, 7]

    windows = hurricast_windows([ophelia, storms["2026058S18168"]], window_size=8, predict_at=8)
    assert windows["x_stat"].shape == (8, 8, 30)  # URMIL's 7 steps are too short
    first = windows["x_stat"][0]
    np.testing.assert_allclose(first, f[:8])
    np.testing.assert_allclose(windows["position"][0], f[7, [0, 1]])
    np.testing.assert_allclose(windows["intensity"][0], f[15, NAMES.index("WMO_WIND")])
    np.testing.assert_allclose(windows["displacement"][0], f[15, [0, 1]] - f[7, [0, 1]], atol=1e-9)
    # Outside the North Atlantic and Eastern Pacific, 10-minute winds become 1-minute winds (/ 0.93).
    urmil = hurricast_windows([storms["2026058S18168"]], window_size=2, predict_at=2)
    raw = storms["2026058S18168"]["features"]
    np.testing.assert_allclose(urmil["x_stat"][0, :, NAMES.index("WMO_WIND")], raw[:2, NAMES.index("WMO_WIND")] / 0.93)
    np.testing.assert_allclose(urmil["intensity"][0], raw[3, NAMES.index("WMO_WIND")] / 0.93)


def _ophelia_bundle(**kwargs):
    params = dict(path=str(IBTRACS_CSV), era5=str(ERA5_NC), min_steps=5, max_steps=17, train_seasons=[2023], val_seasons=None, test_seasons=None)
    params.update(kwargs)
    return load_dataset("hurricast_ibtracs_era5", **params).load()


def test_hurricast_dataset_reads_era5_maps():
    import netCDF4

    bundle = _ophelia_bundle(standardize=False)
    train = bundle.splits["train"]
    assert train.inputs["x_viz"].shape == (2, 8, 9, 25, 25) and train.inputs["x_stat"].shape == (2, 8, 30)
    assert train.metadata["iso_time"] == ["2023-09-22T21:00:00", "2023-09-23T00:00:00"]
    assert bundle.metadata["map_channels"][:3] == ["u225", "u500", "u700"] and bundle.metadata["hazard_task"] == "tc.intensity"
    # First step of the first window: OPHELIA at 29.5N 75.3W, 2023-09-22 00 UTC -> centre 30N 75W.
    with netCDF4.Dataset(ERA5_NC) as nc:
        lat, lon, levels = nc["latitude"][:], nc["longitude"][:], list(nc["pressure_level"][:])
        rows = [int(np.flatnonzero(lat == value)[0]) for value in range(42, 17, -1)]
        cols = [int(np.flatnonzero(lon == value)[0]) for value in range(-87, -62)]
        for channel, (var, level) in enumerate((v, l) for v in ("u", "v", "z") for l in (225, 500, 700)):
            expected = nc[var][0, levels.index(level)][np.ix_(rows, cols)]
            np.testing.assert_array_equal(train.inputs["x_viz"][0, 0, channel].numpy(), expected)
    torch.testing.assert_close(train.targets, torch.tensor([35.0, 30.0]))  # WMO wind 24 h after 21 and 00 UTC

    standardized = _ophelia_bundle()
    stats = standardized.metadata["standardization"]
    maps = standardized.splits["train"].inputs["x_viz"]
    assert torch.allclose(maps.mean(dim=(0, 1, 3, 4)), torch.zeros(9), atol=1e-4)
    assert len(stats["map_mean"]) == 9 and stats["stat_std"][2] != 1.0
    cos_lat = standardized.splits["train"].inputs["x_stat"][..., 10]
    assert torch.all((cos_lat > 0.8) & (cos_lat < 0.9))  # cyclic encodings are not standardised

    track = _ophelia_bundle(target="displacement", standardize=False)
    t = track.splits["train"]
    assert t.targets.shape == (2, 1, 2) and track.metadata["target_variables"] == ["lat", "lon"]
    stats_only = _ophelia_bundle(era5=None)
    assert "x_viz" not in stats_only.splits["train"].inputs


def test_era5_layouts_and_errors():
    import xarray as xr

    from pyhazards.datasets.tc.hurricast import era5_maps, open_era5

    cds = open_era5(ERA5_NC)
    times, lats, lons = [np.datetime64("2023-09-22T06:00")], [30.3], [-75.1]
    expected = era5_maps(cds, times, lats, lons)
    # NCAR RDA layout: upper-case variables, "level", "time", longitudes 0-360, levels ascending.
    with xr.open_dataset(ERA5_NC) as raw:
        rda = raw.rename({"u": "U", "v": "V", "z": "Z", "pressure_level": "level", "valid_time": "time"}).sortby("level")
        rda = rda.assign_coords(longitude=(rda.longitude % 360)).load()
    np.testing.assert_array_equal(era5_maps(rda, times, lats, lons), expected)
    with pytest.raises(KeyError, match="lacks the maps"):
        era5_maps(cds, [np.datetime64("2023-09-25T00:00")], lats, lons)
    with pytest.raises(KeyError, match="lacks the maps"):
        era5_maps(cds, times, [10.0], lons)  # outside the stored area
    with pytest.raises(ValueError, match="no z variable"):
        open_era5(cds.drop_vars("z"))
    with pytest.raises(ValueError, match="needs path"):
        load_dataset("hurricast_ibtracs_era5")
