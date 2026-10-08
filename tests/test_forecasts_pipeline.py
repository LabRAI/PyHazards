"""End-to-end forecast pipeline on synthetic fields: track -> match to best tracks -> score."""

import ast
import math

import pandas as pd
import pytest

from pyhazards.forecasts import (
    ECMWF_TRACKER,
    GRAPHCAST_TRACKER,
    PANGU_TRACKER,
    follow_forecast_tracks,
    score_forecast_tracks,
    tempest_forecast_tracks,
)
from pyhazards.forecasts.synthetic import SyntheticVortex, synthetic_cyclone_fields

HOURS = list(range(0, 49, 6))
HALF_DIAGONAL_KM = 0.5 * math.hypot(0.5, 0.5) * math.pi * 6371.0 / 180.0


@pytest.fixture(scope="module")
def case():
    vortices = [SyntheticVortex(15.0, 140.0, u=-5.0, v=2.0), SyntheticVortex(-15.0, 60.0, u=-4.0, v=-2.0)]
    fields = synthetic_cyclone_fields(vortices, HOURS, init_time="2021-08-01T00")
    truth = ast.literal_eval(fields.attrs["tracks"])
    times = pd.to_datetime(fields["time"].values)
    rows = []
    for sid, track in zip(["2021213N15140", "2021213S15060"], truth):
        # Best-track rows: the true centres, plus one 6 h before the forecast starts.
        before = (track[0][0] - (track[1][0] - track[0][0]), track[0][1] - (track[1][1] - track[0][1]))
        rows.append((sid, times[0] - pd.Timedelta(hours=6), *before, 60.0, 980.0))
        rows += [(sid, time, lat, lon, 60.0, 980.0) for time, (lat, lon) in zip(times, track)]
    ibtracs = pd.DataFrame(rows, columns=["SID", "ISO_TIME", "LAT", "LON", "USA_WIND", "USA_PRES"])
    return fields, ibtracs, times[0]


def test_tempest_route(case):
    fields, ibtracs, init = case
    table = tempest_forecast_tracks(fields, ibtracs, init)
    assert sorted(table["SID"].unique()) == ["2021213N15140", "2021213S15060"]
    result = score_forecast_tracks(table, ibtracs, lead_hours=[6, 24, 48])
    for hours in (6, 24, 48):
        assert result.metrics[f"track_error_km_{hours}h"] <= HALF_DIAGONAL_KM
        assert result.metrics[f"count_{hours}h"] == 2
    assert "intensity_mae_24h" in result.metrics and "pressure_mae_24h" in result.metrics


@pytest.mark.parametrize("config", [PANGU_TRACKER, ECMWF_TRACKER, GRAPHCAST_TRACKER], ids=lambda c: c.name)
def test_following_route(case, config):
    fields, ibtracs, init = case
    table = follow_forecast_tracks(fields, ibtracs, init, config)
    assert table.groupby("SID").size().to_dict() == {"2021213N15140": len(HOURS) - 1, "2021213S15060": len(HOURS) - 1}
    assert (table["lead_hours"] > 0).all()
    result = score_forecast_tracks(table, ibtracs, lead_hours=[6, 24, 48], protocol="pyhazards")
    assert result.metrics["track_error_km"] <= HALF_DIAGONAL_KM
    only = follow_forecast_tracks(fields, ibtracs, init, config, sids=["2021213S15060"], max_lead_hours=24)
    assert set(only["SID"]) == {"2021213S15060"} and only["lead_hours"].max() == 24
