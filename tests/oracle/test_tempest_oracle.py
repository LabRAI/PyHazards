"""pyhazards.forecasts.tempest checked against the official TempestExtremes binaries and TCBench's tracks.

TempestExtremes v2.4.2 (https://github.com/ClimateGlobalChange/tempestextremes, BSD-2-Clause per its
LICENSE; pinned in repos.yaml) is built from source with CMake on first use (needs a C++ compiler and
the netCDF C library, e.g. ``apt-get install cmake libnetcdf-dev``) and run as a separate program;
nothing of it is copied into PyHazards.

1. Synthetic global fields (cyclones in both hemispheres, one crossing the dateline, a cold-core low,
   a shallow low, a polar low) written to netCDF: DetectNodes with TCBench's command line must give the
   same nodes and printed values as :func:`detect_nodes`, and StitchNodes the same CSV, byte for byte.
   Further option sets cover diagonal connectivity, regional grids, ``--maxgap`` and thresholds.
2. Large / network (``PYHAZARDS_ORACLE_LARGE=1``): the TempestExtremes variables of TCBench's
   Pangu-Weather forecast initialised 2023-07-16 12 UTC are read from the pinned Hugging Face revision
   (about 200 MB of range requests). Both the official binaries and PyHazards must reproduce TCBench's
   released ``unmatched_tracks`` file for that forecast byte for byte, and PyHazards' IBTrACS matching
   must find TCBench's matched storms.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oracle_utils import missing, oracle_asset, oracle_repo

from pyhazards.forecasts import (
    TCBENCH_DETECT_NODES,
    TCBENCH_STITCH_NODES,
    ClosedContourCriterion,
    DetectNodesConfig,
    NodeOutput,
    StitchNodesConfig,
    StitchThreshold,
    detect_nodes,
    stitch_nodes,
    write_stitchnodes_csv,
)
from pyhazards.forecasts.synthetic import SyntheticVortex, synthetic_cyclone_fields

TCBENCH_DETECT_ARGS = [
    "--searchbymin", "msl",
    "--closedcontourcmd", "msl,200.0,5.5,0;_DIFF(z300,z500),-58.8,6.5,1.0",
    "--mergedist", "6.0",
    "--outputcmd", "msl,min,0;_VECMAG(u10,v10),max,2",
]
TCBENCH_STITCH_ARGS = [
    "--in_fmt", "lon,lat,slp,wind10",
    "--range", "8.0",
    "--mintime", "12h",
    "--threshold", "wind10,>=,10.0,2;lat,<=,50.0,1;lat,>=,-50.0,1",
    "--out_file_format", "csv",
]


@pytest.fixture(scope="module")
def tempest_bin() -> Path:
    repo = oracle_repo("tempestextremes")
    binaries = repo / "build" / "bin"
    if not (binaries / "DetectNodes").exists() or not (binaries / "StitchNodes").exists():
        if shutil.which("cmake") is None:
            missing("cmake is needed to build TempestExtremes (apt-get install cmake libnetcdf-dev)")
        build = repo / "build"
        build.mkdir(exist_ok=True)
        subprocess.run(["cmake", "-DCMAKE_BUILD_TYPE=Release", "-DENABLE_MPI=OFF", ".."], cwd=build, check=True, capture_output=True)
        jobs = str(min(8, os.cpu_count() or 1))
        subprocess.run(["make", f"-j{jobs}", "DetectNodes", "StitchNodes"], cwd=build, check=True, capture_output=True)
    return binaries


def _write_netcdf(path: Path, fields, lat, lon, times) -> None:
    import netCDF4

    times = pd.to_datetime(list(times))
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", None)
        ds.createDimension("lat", len(lat))
        ds.createDimension("lon", len(lon))
        var = ds.createVariable("time", "f8", ("time",))
        var.units = f"hours since {times[0]:%Y-%m-%d %H:%M:%S}"
        var.calendar = "standard"
        var[:] = (times - times[0]).total_seconds() / 3600.0
        ds.createVariable("lat", "f8", ("lat",))[:] = lat
        ds.createVariable("lon", "f8", ("lon",))[:] = lon
        for name, values in fields.items():
            ds.createVariable(name, "f4", ("time", "lat", "lon"))[:] = np.asarray(values, dtype=np.float32)


def _parse_nodes(path: Path):
    """DetectNodes text output -> list of (time, i, j, lon, lat, values...) string tuples."""
    rows = []
    lines = path.read_text().splitlines()
    k = 0
    while k < len(lines):
        year, month, day, count, hour = lines[k].split("\t")
        stamp = pd.Timestamp(int(year), int(month), int(day), int(hour))
        for line in lines[k + 1 : k + 1 + int(count)]:
            rows.append((stamp,) + tuple(line.strip().split("\t")))
        k += 1 + int(count)
    return rows


def _run_official(tempest_bin: Path, workdir: Path, fields, lat, lon, times, detect_args, stitch_args, extra=()):
    nc = workdir / "fields.nc"
    _write_netcdf(nc, fields, lat, lon, times)
    nodes = workdir / "nodes.txt"
    tracks = workdir / "tracks.csv"
    subprocess.run([str(tempest_bin / "DetectNodes"), "--in_data", str(nc), "--out", str(nodes), *detect_args, *extra], check=True, capture_output=True)
    subprocess.run([str(tempest_bin / "StitchNodes"), "--in", str(nodes), "--out", str(tracks), *stitch_args], check=True, capture_output=True)
    return nodes, tracks


def _compare(tmp_path, tempest_bin, ds, detect, stitch, detect_args, stitch_args, extra=(), crop=None):
    names = ("msl", "u10", "v10", "z300", "z500")
    lat, lon = ds["lat"].values, ds["lon"].values
    fields = {name: ds[name].values for name in names}
    if crop is not None:
        rows, cols = crop
        lat, lon = lat[rows], lon[cols]
        fields = {name: value[:, rows][:, :, cols] for name, value in fields.items()}
    times = ds["time"].values
    nodes_path, tracks_path = _run_official(tempest_bin, tmp_path, fields, lat, lon, times, detect_args, stitch_args, extra)
    ours = detect_nodes(fields, lat, lon, times, detect)
    official = _parse_nodes(nodes_path)
    columns = ["time", "i", "j", "text_lon", "text_lat"] + [c for c in ours.columns if c.startswith("text_") and c not in ("text_lon", "text_lat")]
    mine = [(row[0],) + tuple(str(v) for v in row[1:]) for row in ours[columns].itertuples(index=False)]
    assert mine == official
    assert len(mine) > 0
    ours.attrs["times"] = list(times)
    tracks = stitch_nodes(ours, stitch)
    out = tmp_path / "ours.csv"
    write_stitchnodes_csv(tracks, out)
    assert out.read_text() == tracks_path.read_text()
    return tracks


@pytest.fixture(scope="module")
def synthetic():
    vortices = [
        SyntheticVortex(15.0, 140.0, u=-5.0, v=2.0),
        SyntheticVortex(-15.0, 60.0, u=-4.0, v=-2.0),
        SyntheticVortex(-18.0, 179.0, u=6.0, v=-1.0),  # crosses the dateline
        SyntheticVortex(25.0, 300.0, u=3.0, v=3.0, warm_core=0.0),  # cold core: rejected
        SyntheticVortex(10.0, 200.0, pressure_drop=100.0),  # too shallow: rejected
        SyntheticVortex(55.0, 20.0, u=8.0, v=0.0, vmax=20.0),  # warm-core low poleward of 50 degrees
    ]
    return synthetic_cyclone_fields(vortices, list(range(0, 49, 6)), resolution=0.5)


def test_tcbench_configuration_matches_official_binaries(tmp_path, tempest_bin, synthetic):
    tracks = _compare(tmp_path, tempest_bin, synthetic, TCBENCH_DETECT_NODES, TCBENCH_STITCH_NODES, TCBENCH_DETECT_ARGS, TCBENCH_STITCH_ARGS)
    assert tracks["track_id"].nunique() == 3  # the poleward low fails the |lat| <= 50 threshold


def test_option_variants_match_official_binaries(tmp_path, tempest_bin, synthetic):
    detect = DetectNodesConfig(
        search_by_min="msl",
        closed_contours=(ClosedContourCriterion("msl", 150.0, 4.0, 0.0), ClosedContourCriterion("_DIFF(z300,z500)", -40.0, 5.0, 1.5)),
        merge_dist=4.0,
        outputs=(NodeOutput("msl", "min", 0.0, "slp"), NodeOutput("_VECMAG(u10,v10)", "max", 3.0, "wind10"), NodeOutput("z300", "max", 1.0, "zmax")),
        diagonal_connectivity=True,
    )
    detect_args = [
        "--searchbymin", "msl",
        "--closedcontourcmd", "msl,150.0,4.0,0;_DIFF(z300,z500),-40.0,5.0,1.5",
        "--mergedist", "4.0",
        "--outputcmd", "msl,min,0;_VECMAG(u10,v10),max,3;z300,max,1",
        "--diag_connect",
    ]
    stitch = StitchNodesConfig(
        columns=("lon", "lat", "slp", "wind10", "zmax"),
        range_deg=6.0,
        min_time=3,
        max_gap=1,
        thresholds=(StitchThreshold("wind10", ">=", 15.0, "all"), StitchThreshold("lat", "|<=", 52.0, "first")),
    )
    stitch_args = [
        "--in_fmt", "lon,lat,slp,wind10,zmax", "--range", "6.0", "--mintime", "3", "--maxgap", "1",
        "--threshold", "wind10,>=,15.0,all;lat,|<=,52.0,first", "--out_file_format", "csv",
    ]
    _compare(tmp_path, tempest_bin, synthetic, detect, stitch, detect_args, stitch_args)


def test_regional_grid_matches_official_binaries(tmp_path, tempest_bin, synthetic):
    lat, lon = synthetic["lat"].values, synthetic["lon"].values
    rows = np.flatnonzero((lat <= 40) & (lat >= -40))
    cols = np.flatnonzero((lon >= 30) & (lon <= 200))
    detect = dataclasses.replace(TCBENCH_DETECT_NODES, regional=True)
    _compare(tmp_path, tempest_bin, synthetic, detect, TCBENCH_STITCH_NODES, TCBENCH_DETECT_ARGS, TCBENCH_STITCH_ARGS, extra=("--regional",), crop=(rows, cols))


def test_tcbench_released_pangu_tracks_reproduced(tmp_path, tempest_bin):
    if os.environ.get("PYHAZARDS_ORACLE_LARGE") != "1":
        pytest.skip("reads ~200 MB of TCBench's raw Pangu-Weather output over HTTP; set PYHAZARDS_ORACLE_LARGE=1")
    from pyhazards.forecasts import read_tcbench_fields, read_tcbench_matched_tracks, tempest_forecast_tracks

    init = pd.Timestamp("2023-07-16 12:00")
    released = oracle_asset("tcbench_pangu_unmatched_20230716_12") / "panguweather_2023.07.16-12h00_maxltd-120_timeres-6.csv"
    fields = read_tcbench_fields("pangu", init)
    arrays = {name: fields[name].values for name in fields.data_vars}
    _, official = _run_official(tempest_bin, tmp_path, arrays, fields["lat"].values, fields["lon"].values, fields["time"].values, TCBENCH_DETECT_ARGS, TCBENCH_STITCH_ARGS)
    assert official.read_text() == released.read_text()

    matched = read_tcbench_matched_tracks(oracle_asset("tcbench_matched_pangu") / "2023_PANGU.csv")
    matched = matched[matched["init_time"] == init]
    # Best tracks for the matching step: TCBench's own 2023 IBTrACS extract (pinned with the dataset).
    ibtracs = pd.read_csv(oracle_asset("tcbench_ibtracs_2023") / "2023_IBTrACS.csv", keep_default_na=False, na_values=[""])
    table, tracks = tempest_forecast_tracks(fields, ibtracs, init, return_tracks=True)
    ours = tmp_path / "ours.csv"
    write_stitchnodes_csv(tracks, ours)
    assert ours.read_text() == released.read_text()
    assert set(table["SID"]) == set(matched["SID"])
    key = ["SID", "valid_time"]
    merged = table.merge(matched, on=key, suffixes=("", "_tcbench"))
    assert len(merged) == len(matched)
    np.testing.assert_allclose(merged["lat"], merged["lat_tcbench"])
    np.testing.assert_allclose(merged["lon"], merged["lon_tcbench"])
    np.testing.assert_allclose(merged["wind_ms"], merged["wind_ms_tcbench"], rtol=1e-6)
    np.testing.assert_allclose(merged["pres_pa"], merged["pres_pa_tcbench"], rtol=1e-6)
