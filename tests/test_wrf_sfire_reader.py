"""WRF-SFIRE wrfout reader on small synthetic files with the layout of real WRF-SFIRE history output.

The synthetic files copy the dimension names, variable dimensions, dtypes and attributes recorded in
tests/fixtures/wrf_sfire_hill_wrfout_header.json, the header of a real history file of the official
WRF-SFIRE W4.4-S0.1 ideal case test/em_fire/hill (fire-grid variables on (Time, south_north_subgrid,
west_east_subgrid), subgrid sizes (south_north + 1) * sr_y by (west_east + 1) * sr_x with the last
sr_y rows / sr_x columns as padding, no sr_x/sr_y global attributes, XTIME in minutes). Their values
follow what that file holds: TIGN_G is the arrival time where LFN < 0 and the frame time plus one fire
step (2 steps at t = 0) elsewhere.
"""

import json
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from pyhazards.datasets import load_dataset  # noqa: E402
from pyhazards.datasets.wrf_sfire import (  # noqa: E402
    WRF_SFIRE_FIRE_GRID_VARIABLES,
    read_wrf_sfire_fire_grid,
)
from pyhazards.datasets.wrf_sfire.inspection import main as inspection_main  # noqa: E402

REAL = json.loads((Path(__file__).parent / "fixtures" / "wrf_sfire_hill_wrfout_header.json").read_text())
NX, NY, SR, DX = 4, 3, 5, 60.0  # atmospheric cells and refinement; fire cells are DX / SR = 12 m
FIRE_DT = REAL["observations"]["fire_dt_s"]
MINUTES = (0.0, 1.0, 2.0, 3.0)
PAD = -777.0
SIZES = {
    "Time": None,
    "DateStrLen": 19,
    "west_east": NX,
    "south_north": NY,
    "west_east_subgrid": (NX + 1) * SR,
    "south_north_subgrid": (NY + 1) * SR,
}


def _truth():
    """Arrival time (s) on the fire grid, south-up, for a fire ignited at t = 30 s in the south-west."""
    rows, cols = NY * SR, NX * SR
    y, x = (np.mgrid[0:rows, 0:cols] + 0.5) * (DX / SR)
    return 30.0 + np.hypot(x - 40.0, y - 30.0) / 0.5  # 0.5 m/s spread


def _write(path, frames, with_sr_attrs=False, variables=("TIGN_G", "LFN", "FIRE_AREA", "FGRNHFX", "ROS", "NFUEL_CAT", "ZSF")):
    """A wrfout with the real file's layout (REAL), small sizes and the given fire-grid variables."""
    truth = _truth()
    with netCDF4.Dataset(path, "w", format=REAL["file_format"]) as ds:
        for name, size in SIZES.items():
            ds.createDimension(name, size)
        for key in ("TITLE", "SIMULATION_START_DATE"):
            ds.setncattr(key, REAL["global_attributes"][key])
        ds.setncattr("WEST-EAST_GRID_DIMENSION", np.int32(NX + 1))
        ds.setncattr("SOUTH-NORTH_GRID_DIMENSION", np.int32(NY + 1))
        ds.DX = np.float32(DX)
        ds.DY = np.float32(DX)
        if with_sr_attrs:  # accepted too (wrfxpy looks for them), though WRF's own output has none
            ds.sr_x = np.int32(SR)
            ds.sr_y = np.int32(SR)
        out = {}
        for name in ("Times", "XTIME", *variables):
            spec = REAL["variables"][name]
            var = ds.createVariable(name, spec["dtype"], tuple(spec["dimensions"]))
            for key, value in spec["attributes"].items():
                var.setncattr(key, np.int32(value) if isinstance(value, int) else value)
            out[name] = var
        grn = ds.createVariable("GRNHFX", "f4", ("Time", "south_north", "west_east"))  # atmospheric grid
        grn.description = "heat flux from ground fire"
        for i, minute in enumerate(frames):
            t = minute * 60.0
            out["Times"][i, :] = np.frombuffer("0001-01-01_00:{:02d}:00".format(int(minute)).encode(), dtype="S1")
            out["XTIME"][i] = minute
            grn[i] = 0.0
            burned = truth <= t
            fields = {
                "TIGN_G": np.where(burned, truth, t + (2 * FIRE_DT if t == 0 else FIRE_DT)),
                "LFN": truth - t,  # negative inside the fire
                "FIRE_AREA": burned.astype(float),
                "FGRNHFX": np.where(burned, 5e4, 0.0),
                "ROS": np.full(truth.shape, 0.5),
                "NFUEL_CAT": np.full(truth.shape, 3.0),
                "ZSF": np.linspace(0.0, 10.0, truth.shape[1])[None, :].repeat(truth.shape[0], 0),
            }
            for name in variables:
                padded = np.full(((NY + 1) * SR, (NX + 1) * SR), PAD, dtype=np.float32)
                padded[: NY * SR, : NX * SR] = fields[name]
                out[name][i] = padded
    return truth


def test_fixture_is_a_real_wrf_sfire_header():
    dims = REAL["dimensions"]
    assert dims["west_east_subgrid"] == (dims["west_east"] + 1) * 10 == 420
    assert dims["south_north_subgrid"] == (dims["south_north"] + 1) * 10 == 420
    assert "sr_x" not in REAL["global_attributes"]
    for name, (description, units) in WRF_SFIRE_FIRE_GRID_VARIABLES.items():
        spec = REAL["variables"].get(name)
        if spec is None:
            continue
        assert spec["dimensions"] == ["Time", "south_north_subgrid", "west_east_subgrid"]
        assert (spec["attributes"]["description"], spec["attributes"]["units"]) == (description, units)


@pytest.fixture
def wrfout(tmp_path):
    path = tmp_path / "wrfout_d01_0001-01-01_00:00:00"
    truth = _write(path, MINUTES)
    return path, truth


def test_reads_fire_grid_without_padding(wrfout):
    path, truth = wrfout
    grid = read_wrf_sfire_fire_grid(path, origin="lower")
    assert grid.shape == (NY * SR, NX * SR)
    assert (grid.sr_x, grid.sr_y) == (SR, SR)
    assert grid.fire_dx == grid.fire_dy == DX / SR
    np.testing.assert_array_equal(grid.times, np.array(MINUTES) * 60.0)
    assert grid.time_strings[1] == "0001-01-01_00:01:00"
    assert set(grid.variables) == {"TIGN_G", "LFN", "FIRE_AREA", "FGRNHFX", "ROS"}
    for values in grid.variables.values():
        assert values.shape == (len(MINUTES), NY * SR, NX * SR)
        assert not np.any(values == PAD)

    masks = grid.burned_masks()
    for i, minute in enumerate(MINUTES):
        np.testing.assert_array_equal(masks[i], truth <= minute * 60.0)
    arrival = grid.arrival_time()
    np.testing.assert_allclose(arrival[np.isfinite(arrival)], truth[truth <= 180.0], rtol=1e-6)
    assert np.isinf(arrival[truth > 180.0]).all()


def test_north_up_is_the_default_and_flips_rows(wrfout):
    path, _ = wrfout
    lower = read_wrf_sfire_fire_grid(path, origin="lower")
    upper = read_wrf_sfire_fire_grid(path)
    assert upper.origin == "upper"
    np.testing.assert_array_equal(upper.variables["TIGN_G"], lower.variables["TIGN_G"][:, ::-1])
    np.testing.assert_array_equal(upper.burned_masks(), lower.burned_masks()[:, ::-1])
    keep = read_wrf_sfire_fire_grid(path, variables=("LFN",), strip_padding=False, origin="lower")
    assert keep.variables["LFN"].shape == (len(MINUTES), (NY + 1) * SR, (NX + 1) * SR)


def test_refinement_from_dimensions_and_multiple_files(tmp_path, wrfout):
    path, _ = wrfout
    single = read_wrf_sfire_fire_grid(path, variables=("TIGN_G", "LFN"))
    parts = []
    for minute in reversed(MINUTES):  # one frame per file, listed out of order
        part = tmp_path / f"wrfout_d01_0001-01-01_00:{int(minute):02d}:00"
        _write(part, (minute,), with_sr_attrs=True, variables=("TIGN_G", "LFN"))
        parts.append(part)
    merged = read_wrf_sfire_fire_grid(parts, variables=("TIGN_G", "LFN"))
    assert (merged.sr_x, merged.sr_y) == (SR, SR)
    np.testing.assert_array_equal(merged.times, single.times)
    np.testing.assert_array_equal(merged.variables["TIGN_G"], single.variables["TIGN_G"])
    globbed = read_wrf_sfire_fire_grid(str(tmp_path / "wrfout_d01_0001-01-01_00:0[0-3]:00"), variables=("LFN",))
    assert globbed.variables["LFN"].shape[0] == len(MINUTES)


def test_tign_only_files_use_the_arrival_time(tmp_path):
    path = tmp_path / "wrfout_tign_only"
    truth = _write(path, MINUTES, variables=("TIGN_G",))
    grid = read_wrf_sfire_fire_grid(path, variables=("TIGN_G",), origin="lower")
    masks = grid.burned_masks()
    for i, minute in enumerate(MINUTES):
        np.testing.assert_array_equal(masks[i], truth <= minute * 60.0)


def test_errors(wrfout, tmp_path):
    path, _ = wrfout
    with pytest.raises(KeyError, match="fire-grid variables in the file"):
        read_wrf_sfire_fire_grid(path, variables=("F_ROS",))
    with pytest.raises(ValueError, match="only fire-grid variables"):
        read_wrf_sfire_fire_grid(path, variables=("GRNHFX",))
    with pytest.raises(FileNotFoundError):
        read_wrf_sfire_fire_grid(tmp_path / "missing_wrfout")
    with pytest.raises(ValueError, match="origin"):
        read_wrf_sfire_fire_grid(path, origin="north")
    grid = read_wrf_sfire_fire_grid(path, variables=("FIRE_AREA",))
    with pytest.raises(KeyError, match="LFN or TIGN_G"):
        grid.burned_masks()


def test_spread_pairs_and_registered_dataset(wrfout):
    path, truth = wrfout
    grid = read_wrf_sfire_fire_grid(path, variables=("TIGN_G", "LFN", "NFUEL_CAT"), origin="lower")
    inputs, targets, times = grid.spread_pairs(horizon=1, features=("NFUEL_CAT",))
    assert inputs.shape == (3, 2, NY * SR, NX * SR) and targets.shape == (3, 1, NY * SR, NX * SR)
    np.testing.assert_array_equal(inputs[:, 0], grid.burned_masks()[:3])
    np.testing.assert_array_equal(targets[:, 0], grid.burned_masks()[1:])
    np.testing.assert_array_equal(inputs[:, 1], 3.0)
    np.testing.assert_array_equal(times, [0.0, 60.0, 120.0])
    with pytest.raises(ValueError, match="more than 4"):
        grid.spread_pairs(horizon=4)

    bundle = load_dataset("wrf_sfire_spread", paths=str(path), features=("ZSF",), train_fraction=0.34, val_fraction=0.33).load()
    assert bundle.metadata["hazard_task"] == "wildfire.spread"
    assert bundle.feature_spec.channels == 2
    assert bundle.label_spec.task_type == "segmentation"
    sizes = [bundle.get_split(name).inputs.shape[0] for name in ("train", "val", "test")]
    assert sizes == [1, 1, 1]
    assert bundle.get_split("test").targets.shape == (1, 1, NY * SR, NX * SR)
    with pytest.raises(ValueError, match="wrfout"):
        load_dataset("wrf_sfire_spread")


def test_inspection_cli(wrfout, capsys):
    path, _ = wrfout
    assert inspection_main([]) == 0
    assert inspection_main(["--path", str(path)]) == 0
    out = capsys.readouterr().out
    assert "fire grid: 15 x 20 cells of 12 x 12 m (sr_x=5, sr_y=5)" in out
    assert inspection_main(["--path", str(path) + "_missing"]) == 2
