"""Cyclone trackers of pyhazards.forecasts on synthetic vortices with known tracks."""

import ast
import math

import numpy as np
import pandas as pd
import pytest

from pyhazards.forecasts import (
    ECMWF_TRACKER,
    GRAPHCAST_TRACKER,
    PANGU_TRACKER,
    TCBENCH_DETECT_NODES,
    TCBENCH_STITCH_NODES,
    ClosedContourCriterion,
    DetectNodesConfig,
    StitchNodesConfig,
    StitchThreshold,
    detect_nodes,
    follow_cyclone,
    great_circle_km,
    read_stitchnodes_csv,
    relative_vorticity,
    required_variables,
    stitch_nodes,
    tempest_tracks,
    write_stitchnodes_csv,
)
from pyhazards.forecasts.following import with_overrides
from pyhazards.forecasts.synthetic import SyntheticVortex, synthetic_cyclone_fields

HOURS = list(range(0, 73, 6))
# Grid points are 0.5 degrees apart: a centre is never more than half a cell diagonal from the truth.
HALF_DIAGONAL_KM = 0.5 * math.hypot(0.5, 0.5) * math.pi * 6371.0 / 180.0


def _errors(track, truth, hours=HOURS, times=None):
    out = []
    for lat, lon, lead in zip(track["lat"], track["lon"], track["lead_hours"]):
        true_lat, true_lon = truth[hours.index(int(lead))]
        out.append(float(great_circle_km(lat, lon, true_lat, true_lon)))
    return out


@pytest.fixture(scope="module")
def storms():
    vortices = [
        SyntheticVortex(15.0, 140.0, u=-5.0, v=2.0),  # NH cyclone moving west-north-west
        SyntheticVortex(-15.0, 60.0, u=-4.0, v=-2.0),  # SH cyclone (clockwise)
        SyntheticVortex(25.0, 300.0, u=3.0, v=3.0, warm_core=0.0),  # cold-core low
        SyntheticVortex(10.0, 200.0, pressure_drop=100.0),  # too shallow for a 200 Pa closed contour
    ]
    ds = synthetic_cyclone_fields(vortices, HOURS)
    return vortices, ds, ast.literal_eval(ds.attrs["tracks"])


def _tempest_inputs(ds):
    return {name: ds[name].values for name in ("msl", "u10", "v10", "z300", "z500")}, ds["lat"].values, ds["lon"].values, ds["time"].values


def test_tempest_tracks_follow_warm_core_cyclones_only(storms):
    _, ds, truth = storms
    tracks = tempest_tracks(*_tempest_inputs(ds))
    assert sorted(tracks["track_id"].unique()) == [0, 1]
    # Paths are ordered by their first node's grid index: with latitudes stored north to south, the
    # Northern Hemisphere storm comes first.
    for track_id, expected in ((0, truth[0]), (1, truth[1])):
        points = tracks[tracks["track_id"] == track_id]
        leads = ((pd.to_datetime(points["time"]) - pd.Timestamp(ds["time"].values[0])).dt.total_seconds() / 3600).astype(int)
        assert len(points) == len(HOURS)
        for lat, lon, lead in zip(points["lat"], points["lon"], leads):
            true_lat, true_lon = expected[HOURS.index(lead)]
            assert great_circle_km(lat, lon, true_lat, true_lon) <= HALF_DIAGONAL_KM
        assert points["wind10"].max() > 25.0
        assert (points["slp"] < 101325 - 1500).all()


def test_tempest_nodes_columns_and_formatting(storms):
    _, ds, _ = storms
    nodes = detect_nodes(*_tempest_inputs(ds))
    assert list(nodes.columns[:7]) == ["time", "i", "j", "text_lon", "text_lat", "text_slp", "text_wind10"]
    first = nodes.iloc[0]
    assert first["text_lon"] == "%3.6f" % first["lon"] and "e+0" in first["text_slp"]
    # Nodes of one time are in DetectNodes' order (ascending flat grid index).
    one = nodes[nodes["time"] == nodes["time"].iloc[0]]
    flat = one["j"] * ds.sizes["lon"] + one["i"]
    assert list(flat) == sorted(flat)


def test_tempest_csv_round_trip(storms, tmp_path):
    _, ds, _ = storms
    tracks = tempest_tracks(*_tempest_inputs(ds))
    path = tmp_path / "tracks.csv"
    write_stitchnodes_csv(tracks, path)
    lines = path.read_text().splitlines()
    assert lines[0] == "track_id, year, month, day, hour, i, j, lon, lat, slp, wind10"
    back = read_stitchnodes_csv(path)
    assert (back["text_lat"].to_numpy() == tracks["text_lat"].to_numpy()).all()
    pd.testing.assert_series_equal(back["time"], tracks["time"], check_names=False)
    write_stitchnodes_csv(back, tmp_path / "again.csv")
    assert (tmp_path / "again.csv").read_text() == path.read_text()


def _nodes(rows):
    frame = pd.DataFrame(rows, columns=["time", "i", "j", "lon", "lat", "wind"])
    frame["time"] = pd.to_datetime(frame["time"])
    return frame


def test_stitch_first_come_first_served_and_filters():
    t = ["2020-01-01T00", "2020-01-01T06", "2020-01-01T12", "2020-01-01T18"]
    rows = [
        (t[0], 0, 0, 100.0, 10.0, 20.0),
        (t[0], 1, 0, 101.0, 10.0, 20.0),  # also nearest to the same node at t1: shares it, path ends there
        (t[1], 0, 0, 100.5, 10.0, 20.0),
        (t[2], 0, 0, 101.0, 10.0, 5.0),
        (t[3], 0, 0, 130.0, 10.0, 20.0),  # beyond 8 degrees: not linked
    ]
    config = StitchNodesConfig(columns=("lon", "lat", "wind"), range_deg=8.0, min_time=1)
    tracks = stitch_nodes(_nodes(rows), config)
    assert tracks.groupby("track_id").size().tolist() == [3, 2]
    assert tracks[tracks["track_id"] == 0]["lon"].tolist() == [100.0, 100.5, 101.0]
    assert tracks[tracks["track_id"] == 1]["lon"].tolist() == [101.0, 100.5]
    # min_time as a duration; thresholds with counts.
    longer = stitch_nodes(_nodes(rows), StitchNodesConfig(columns=("lon", "lat", "wind"), range_deg=8.0, min_time="12h"))
    assert longer["track_id"].nunique() == 1
    strict = StitchNodesConfig(columns=("lon", "lat", "wind"), range_deg=8.0, min_time=1, thresholds=(StitchThreshold("wind", ">=", 10.0, "all"),))
    assert stitch_nodes(_nodes(rows), strict)["track_id"].nunique() == 1  # the 3-point path has a weak point
    last = StitchNodesConfig(columns=("lon", "lat", "wind"), range_deg=8.0, min_time=1, thresholds=(StitchThreshold("wind", "<", 10.0, "last"),))
    assert stitch_nodes(_nodes(rows), last)["lon"].tolist() == [100.0, 100.5, 101.0]


def test_stitch_max_gap_bridges_missing_steps():
    t = ["2020-01-01T00", "2020-01-01T06", "2020-01-01T12"]
    rows = [(t[0], 0, 0, 100.0, 10.0, 20.0), (t[1], 0, 0, 150.0, -40.0, 20.0), (t[2], 0, 0, 101.0, 10.0, 20.0)]
    nodes = _nodes(rows)
    no_gap = stitch_nodes(nodes, StitchNodesConfig(columns=("lon", "lat", "wind"), range_deg=8.0, min_time=2))
    assert no_gap.empty
    gap = stitch_nodes(nodes, StitchNodesConfig(columns=("lon", "lat", "wind"), range_deg=8.0, min_time=2, max_gap=1))
    assert gap["lon"].tolist() == [100.0, 101.0]


def test_tempest_configuration_validation():
    with pytest.raises(ValueError, match="non-zero"):
        ClosedContourCriterion("msl", 0.0, 5.5)
    with pytest.raises(NotImplementedError):
        from pyhazards.forecasts import NodeOutput

        NodeOutput("msl", "avg", 1.0)
    with pytest.raises(ValueError, match="threshold column"):
        StitchNodesConfig(columns=("lon", "lat"), thresholds=(StitchThreshold("wind", ">=", 1.0, 1),))
    with pytest.raises(NotImplementedError, match="_LAT"):
        detect_nodes(
            {"msl": np.random.rand(1, 5, 8)},
            np.linspace(-60, 60, 5),
            np.arange(0, 360, 45.0),
            ["2020-01-01"],
            DetectNodesConfig(closed_contours=(ClosedContourCriterion("_LAT(msl,1)", 1.0, 5.0),)),
        )
    with pytest.raises(ValueError, match="shaped"):
        detect_nodes({"msl": np.zeros((2, 5, 8))}, np.linspace(-60, 60, 5), np.arange(0, 360, 45.0), ["2020-01-01"])
    assert TCBENCH_DETECT_NODES.merge_dist == 6.0 and TCBENCH_STITCH_NODES.range_deg == 8.0


@pytest.mark.parametrize("config", [PANGU_TRACKER, ECMWF_TRACKER, GRAPHCAST_TRACKER], ids=lambda c: c.name)
def test_following_tracker_presets_follow_both_hemispheres(storms, config):
    vortices, ds, truth = storms
    for k in (0, 1):
        track = follow_cyclone(ds, vortices[k].lat, vortices[k].lon, ds["time"].values[0], config)
        expected_points = len(HOURS) - (0 if config.relocate_start else 1)
        assert len(track) == expected_points, track.attrs["stop_reason"]
        assert max(_errors(track, truth[k])) <= HALF_DIAGONAL_KM
        assert track.attrs["stop_reason"] == "end of forecast"
        assert (track["msl"] < 101325 - 1500).all() and track["wind_max"].min() > 20.0


def test_graphcast_preset_uses_the_selected_hyperparameters():
    assert GRAPHCAST_TRACKER.search_radius_km == pytest.approx(222.5)
    assert GRAPHCAST_TRACKER.check_radius_km == pytest.approx(208.5)
    assert GRAPHCAST_TRACKER.steering_weight == 0.5 and GRAPHCAST_TRACKER.max_turn_deg == 90.0
    assert GRAPHCAST_TRACKER.max_first_guess_km is None
    assert ECMWF_TRACKER.search_radius_km == 445.0 and ECMWF_TRACKER.check_radius_km == 278.0
    assert not PANGU_TRACKER.use_first_guess and not PANGU_TRACKER.relocate_start
    assert set(required_variables(GRAPHCAST_TRACKER)) >= {"msl", "u850", "v850", "u200", "v700"}
    assert "z200" in required_variables(with_overrides(PANGU_TRACKER, extratropical_latitude=30.0))


def test_vorticity_check_rejects_lows_without_circulation():
    hours = [0, 6]
    storm = SyntheticVortex(15.0, 140.0, u=-15.0, v=0.0)
    decoy = SyntheticVortex(15.0, 141.5, u=0.0, v=0.0, vmax=0.0)  # a pressure low with no cyclonic winds
    ds = synthetic_cyclone_fields([storm, decoy], hours)
    truth = ast.literal_eval(ds.attrs["tracks"])
    start = ds["time"].values[0]
    # Searching around the start (Pangu rules), the decoy is the closer minimum but fails the vorticity check.
    assert great_circle_km(15.0, 140.0, *truth[1][1]) < great_circle_km(15.0, 140.0, *truth[0][1]) < PANGU_TRACKER.search_radius_km
    track = follow_cyclone(ds, 15.0, 140.0, start, PANGU_TRACKER)
    assert len(track) == 1
    assert great_circle_km(track["lat"].iloc[0], track["lon"].iloc[0], *truth[0][1]) <= HALF_DIAGONAL_KM
    vort = relative_vorticity(ds["u850"].values[1], ds["v850"].values[1], ds["lat"].values, ds["lon"].values)
    from pyhazards.forecasts.following import _disk

    rows, cols, _ = _disk(ds["lat"].values, ds["lon"].values, 15.0, 141.5, PANGU_TRACKER.check_radius_km)
    assert np.nanmax(vort[rows, cols]) < PANGU_TRACKER.vorticity_threshold  # what the check sees around the decoy


def test_graphcast_turn_limit_and_land_and_thickness_checks():
    # The storm sits still while its previous motion was eastward: any move west turns by > 90 degrees.
    ds = synthetic_cyclone_fields([SyntheticVortex(15.0, 140.0, u=-6.0, v=0.0)], [0, 6])
    previous = (15.0, 139.0)  # 6 h earlier the storm was west of its start: it was moving east
    ecmwf = follow_cyclone(ds, 15.0, 140.0, ds["time"].values[0], with_overrides(ECMWF_TRACKER, steering_weight=0.0), previous_position=previous)
    graphcast = follow_cyclone(ds, 15.0, 140.0, ds["time"].values[0], with_overrides(GRAPHCAST_TRACKER, steering_weight=0.0, search_radius_km=445.0), previous_position=previous)
    assert len(ecmwf) == 2 and len(graphcast) == 1  # GraphCast keeps lead 0 and stops at 6 h
    assert "no qualifying minimum" in graphcast.attrs["stop_reason"]

    # A weak storm over land (lsm = 1) fails the 8 m/s wind check.
    weak = SyntheticVortex(30.0, 260.0, u=0.0, v=0.0, vmax=6.0)
    land = synthetic_cyclone_fields([weak], [0, 6], land_mask=True)
    assert float(land["lsm"].sel(lat=30.0, lon=260.0)) == 1.0
    assert len(follow_cyclone(land, 30.0, 260.0, land["time"].values[0], PANGU_TRACKER)) == 0
    sea = land.drop_vars("lsm")
    assert len(follow_cyclone(sea, 30.0, 260.0, sea["time"].values[0], PANGU_TRACKER)) == 1

    # Extratropical (poleward of the configured latitude) cold-core lows fail the thickness check.
    cold = synthetic_cyclone_fields([SyntheticVortex(40.0, 170.0, u=0.0, v=0.0, warm_core=0.0)], [0, 6])
    warm = synthetic_cyclone_fields([SyntheticVortex(40.0, 170.0, u=0.0, v=0.0)], [0, 6])
    config = with_overrides(PANGU_TRACKER, extratropical_latitude=35.0)
    assert len(follow_cyclone(cold, 40.0, 170.0, cold["time"].values[0], config)) == 0
    assert len(follow_cyclone(warm, 40.0, 170.0, warm["time"].values[0], config)) == 1
    assert len(follow_cyclone(cold, 40.0, 170.0, cold["time"].values[0], PANGU_TRACKER)) == 1


def test_following_tracker_input_validation(storms):
    _, ds, _ = storms
    with pytest.raises(KeyError, match="u850"):
        follow_cyclone(ds.drop_vars("u850"), 15.0, 140.0, ds["time"].values[0], PANGU_TRACKER)
    with pytest.raises(ValueError, match="steering_weight"):
        with_overrides(ECMWF_TRACKER, steering_weight=1.5)


def test_relative_vorticity_of_solid_body_rotation():
    lat = np.linspace(-60, 60, 121)
    lon = np.arange(0, 360, 1.0)
    omega = 1e-5
    a = 6371e3
    # Zonal flow u = omega * a * cos(lat) (solid-body rotation about the polar axis) has zeta = 2 omega sin(lat).
    u = omega * a * np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size))
    zeta = relative_vorticity(u, np.zeros_like(u), lat, lon)
    expected = 2 * omega * np.sin(np.deg2rad(lat))[:, None]
    np.testing.assert_allclose(zeta[1:-1], np.broadcast_to(expected, zeta.shape)[1:-1], atol=2e-9)
