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
    names = ["tc_tracks_synthetic", "ships_xu2021_synthetic", "safnet_cma_era_interim_synthetic", "tropicyclonenet_dataset_synthetic"]
    assert set(names) <= set(available_datasets())
    assert not {"tcbench_alpha", "tropicyclonenet_dataset_alias"} & set(available_datasets())
    for name in names:
        bundle = load_dataset(name, micro=True).load()
        assert bundle.metadata["synthetic"] is True and bundle.metadata["dataset"] == name
    tcnd = load_dataset("tropicyclonenet_dataset_synthetic", micro=True).load().splits["test"]
    assert tcnd.inputs["image_obs"].shape[1:] == (1, 8, 64, 64) and tcnd.targets.shape[1:] == (4, 4)
