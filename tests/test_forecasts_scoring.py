"""IBTrACS matching and track scoring of pyhazards.forecasts, including TCBench's evaluation quirks."""

import math

import numpy as np
import pandas as pd
import pytest

from pyhazards.forecasts import (
    KT_PER_MS,
    TCBENCH_KT_PER_MS,
    forecast_track_errors,
    great_circle_km,
    match_tracks,
    matched_forecast_tracks,
    score_forecast_tracks,
    storms_at,
    tcbench_position_error_km,
)

# Rows printed by TCBench's official notebook (TCBench_Alpha dev/Getting_Started.ipynb, cell 8,
# Pangu-Weather 2023): forecast position, wind (kt) and pressure (hPa) from the released
# matched_tracks/2023_PANGU.csv (m/s and Pa there), IBTrACS v04r01 position / USA_WIND / USA_PRES, and
# the published DPE_GCD (km), AE_wind (kt) and AE_pressure (hPa). Cheneso's IBTrACS longitude is the
# value in TCBench's own 2023_IBTrACS.csv (55.8); NCEI has since revised it to 55.9.
PUBLISHED = [
    # SID, init, valid, wind m/s, pres Pa, lat, lon, ib lat, ib lon, USA_WIND, USA_PRES, DPE, AE wind, AE pres
    ("2023013S08081", "2023-01-15 12:00", "2023-01-18 00:00", 15.55148, 100156.1, -14.5, 57.75, -14.0, 55.8, 35, 995, 216.013123, 4.770411, 6.5610),
    ("2023290N12256", "2023-10-19 12:00", "2023-10-20 00:00", 21.7195, 98798.56, 17.75, 251.75, 17.9, -108.2, 105, 948, 18.454374, 62.780767, 39.9856),
    ("2023129N08091", "2023-05-07 00:00", "2023-05-12 00:00", 20.2564, 99153.78, 12.75, 84.25, 13.3, 88.1, 65, 982, 424.145817, 25.624799, 9.5378),
    ("2023193N37305", "2023-07-16 12:00", "2023-07-20 00:00", 12.21571, 101048.8, 32.0, 317.0, 33.8, -40.6, 45, 1001, 302.565885, 21.254614, 9.4880),
]


def _published_tables():
    tracks = pd.DataFrame(
        {
            "SID": [r[0] for r in PUBLISHED] + ["2023251N16334"],
            "init_time": pd.to_datetime([r[1] for r in PUBLISHED] + ["2023-09-17 00:00"]),
            "valid_time": pd.to_datetime([r[2] for r in PUBLISHED] + ["2023-09-19 06:00"]),
            "wind_ms": [r[3] for r in PUBLISHED] + [12.74171],
            "pres_pa": [r[4] for r in PUBLISHED] + [101188.7],
            "lat": [r[5] for r in PUBLISHED] + [38.75],
            "lon": [r[6] for r in PUBLISHED] + [319.5],
        }
    )
    ibtracs = pd.DataFrame(
        {
            "SID": [r[0] for r in PUBLISHED] + ["2023251N16334"],
            "ISO_TIME": pd.to_datetime([r[2] for r in PUBLISHED] + ["2023-09-19 03:00"]),  # Lee: no 06 UTC record
            "LAT": [r[7] for r in PUBLISHED] + [38.0],
            "LON": [r[8] for r in PUBLISHED] + [-66.0],
            "USA_WIND": [r[9] for r in PUBLISHED] + [70],
            "USA_PRES": [r[10] for r in PUBLISHED] + [960],
        }
    )
    return tracks, ibtracs


def test_tcbench_protocol_reproduces_the_published_rows():
    tracks, ibtracs = _published_tables()
    errors = forecast_track_errors(tracks, ibtracs, protocol="tcbench")
    for row, published in zip(errors.itertuples(), PUBLISHED):
        assert row.track_error_km == pytest.approx(published[11], abs=5e-7)
        assert abs(row.wind_error_kt) == pytest.approx(published[12], abs=5e-7)
        assert abs(row.pres_error_hpa) == pytest.approx(published[13], abs=5e-5)
    assert math.isnan(errors["track_error_km"].iloc[-1])  # TCBench prints NaN for Lee at 2023-09-19 06 UTC
    # The float16 quirk matters: double-precision positions give different numbers.
    exact = forecast_track_errors(tracks, ibtracs, protocol="pyhazards")
    assert abs(exact["track_error_km"].iloc[3] - 302.565885) > 1.0
    assert exact["track_error_km"].iloc[3] == pytest.approx(float(great_circle_km(33.8, -40.6, 32.0, 317.0)))
    assert exact["wind_kt"].iloc[0] == pytest.approx(15.55148 * KT_PER_MS)
    assert errors["wind_kt"].iloc[0] == pytest.approx(15.55148 * TCBENCH_KT_PER_MS)


def test_tcbench_position_error_emulates_half_precision_latitudes():
    # TCBench evaluates np.radians / np.cos on the float16 latitude column.
    value = tcbench_position_error_km([33.8], [-40.6], [32.0], [317.0])[0]
    assert value == pytest.approx(302.565885, abs=5e-7)
    lat16 = float(np.float16(33.8))
    assert lat16 != 33.8 and abs(value - float(great_circle_km(lat16, float(np.float16(-40.6)) + 360, 32.0, 317.0))) > 1e-3


def test_tcbench_protocol_filters_and_deduplicates():
    tracks, ibtracs = _published_tables()
    extra = tracks.iloc[[3]].copy()
    extra["init_time"] = pd.Timestamp("2023-07-16 06:00")  # 06 UTC initialisations are not evaluated by TCBench
    duplicate = tracks.iloc[[3]].copy()
    duplicate["lat"] = 0.0  # duplicates of (SID, init, valid) keep the first row
    unknown = tracks.iloc[[3]].copy()
    unknown["SID"] = "2099001N00000"
    table = pd.concat([tracks, extra, duplicate, unknown], ignore_index=True)
    errors = forecast_track_errors(table, ibtracs, protocol="tcbench")
    assert len(errors) == len(tracks)
    assert errors["track_error_km"].iloc[3] == pytest.approx(302.565885, abs=5e-7)
    assert len(forecast_track_errors(table, ibtracs, protocol="pyhazards")) == len(tracks) + 1  # keeps 06 UTC


def test_score_forecast_tracks_per_lead_metrics():
    ibtracs = pd.DataFrame(
        {
            "SID": ["A"] * 3 + ["B"] * 3,
            "ISO_TIME": pd.to_datetime(["2020-01-01 00:00", "2020-01-02 00:00", "2020-01-03 00:00"] * 2),
            "LAT": [10.0, 11.0, 12.0, -10.0, -11.0, -12.0],
            "LON": [130.0, 129.0, 128.0, 179.5, -179.5, -178.5],
            "USA_WIND": [50, 60, 70, 40, 45, 50],
            "USA_PRES": [990, 985, 980, 995, 993, 990],
        }
    )
    one_degree = 2 * math.pi * 6371.0 / 360
    tracks = pd.DataFrame(
        {
            "SID": ["A", "A", "B", "B"],
            "init_time": pd.to_datetime(["2020-01-01"] * 4),
            "valid_time": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-02", "2020-01-03"]),
            "lat": [12.0, 12.0, -11.0, -12.0],
            "lon": [129.0, 128.0, 179.5, 181.5],  # B crosses the dateline in the other convention
            "wind_ms": [60 / KT_PER_MS, 70 / KT_PER_MS, 55 / KT_PER_MS, 50 / KT_PER_MS],
            "pres_pa": [98000.0, 98000.0, 99300.0, 99000.0],
        }
    )
    result = score_forecast_tracks(tracks, ibtracs, lead_hours=[24, 48], source="unit test")
    m = result.metrics
    assert result.benchmark_name == "tc" and result.hazard_task == "tc.track_intensity"
    assert m["track_error_km_24h"] == pytest.approx((one_degree + one_degree * math.cos(math.radians(11.0))) / 2, rel=1e-3)
    assert m["track_error_km_48h"] == pytest.approx(0.0, abs=1e-6)
    assert m["intensity_mae_24h"] == pytest.approx(5.0)
    assert m["pressure_mae_24h"] == pytest.approx(2.5)
    assert m["count_24h"] == 2 and m["count_48h"] == 2
    assert result.metadata["units"] == {"track_error_km": "km", "intensity_mae": "kt", "pressure_mae": "hPa"}
    assert result.metadata["num_storms"] == 2
    with pytest.raises(ValueError, match="no forecast rows"):
        score_forecast_tracks(tracks.assign(SID="C"), ibtracs)


def test_match_tracks_follows_huracanpy_rule():
    reference = pd.DataFrame(
        {
            "SID": ["S1", "S1", "S2"],
            "ISO_TIME": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 06:00", "2020-01-01 06:00"]),
            "LAT": [10.0, 10.5, 20.0],
            "LON": [130.0, 129.5, 330.0],
        }
    )
    km_per_deg = math.pi * 6371.0088 / 180
    tracks = pd.DataFrame(
        {
            "track_id": [0, 0, 1, 2],
            "time": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 06:00", "2020-01-01 06:00", "2020-01-01 06:00"]),
            "lat": [10.0 + 299.0 / km_per_deg, 10.5, 20.0, 20.0 + 301.0 / km_per_deg],
            "lon": [130.0, 129.5, -30.0, -30.0],
            "wind10": [20.0, 25.0, 15.0, 15.0],
            "slp": [99000.0, 98900.0, 100000.0, 100000.0],
        }
    )
    pairs = match_tracks(tracks, reference)
    assert pairs[["SID", "track_id", "overlap"]].values.tolist() == [["S1", 0, 2], ["S2", 1, 1]]
    assert pairs["mean_dist_km"].iloc[0] == pytest.approx(149.5, rel=1e-6)
    assert match_tracks(tracks, reference, min_overlap=2)["SID"].tolist() == ["S1"]
    table = matched_forecast_tracks(tracks, reference, "2019-12-31 18:00")
    assert table.groupby("SID").size().to_dict() == {"S1": 2, "S2": 1}
    assert table["lead_hours"].tolist() == [6.0, 12.0, 12.0]
    assert table["wind_ms"].tolist() == [20.0, 25.0, 15.0]


def test_storms_at_skips_spurs():
    ibtracs = pd.DataFrame(
        {
            "SID": ["A", "B", "B"],
            "ISO_TIME": pd.to_datetime(["2020-01-01"] * 3),
            "LAT": [10.0, 11.0, 11.5],
            "LON": [130.0, 140.0, 141.0],
            "TRACK_TYPE": ["main", "main", "spur-main"],
        }
    )
    assert storms_at(ibtracs, "2020-01-01")[["SID", "LAT"]].values.tolist() == [["A", 10.0], ["B", 11.0]]
