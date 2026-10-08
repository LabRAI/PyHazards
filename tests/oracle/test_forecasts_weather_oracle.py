"""Forecast sources of pyhazards.forecasts against the real packages and stores they wrap.

1. ``run_earth2studio_forecast`` drives NVIDIA earth2studio 0.19.0 (``tests/oracle/requirements-weather.txt``)
   exactly like ``earth2studio.run.deterministic``: earth2studio's own ``Persistence`` prognostic model
   (the identity model earth2studio uses in its tests; no weights) is started from a data source that
   serves synthetic cyclone fields. The returned fields must equal the source at every lead time, and
   the TempestExtremes-rule tracker must find the cyclones at their initial positions.
2. ``read_weatherbench2_forecast`` reads one small slice (one lead time of mean sea-level pressure) of
   the public WeatherBench 2 Pangu-Weather and GraphCast stores over the network: the MSLP minimum near
   Typhoon Kong-rey 6 h after 2018-09-30 00 UTC (the case of Pangu-Weather's Fig. 1) must lie within
   100 km of the IBTrACS position (14.4 N, 138.0 E, NCEI IBTrACS v04r01).
"""

from __future__ import annotations

import ast
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from oracle_utils import oracle_package

from pyhazards.forecasts import great_circle_km, read_weatherbench2_forecast, run_earth2studio_forecast, tempest_tracks
from pyhazards.forecasts.synthetic import SyntheticVortex, synthetic_cyclone_fields

E2S = {"msl": "msl", "u10": "u10m", "v10": "v10m", "z300": "z300", "z500": "z500"}


class _SyntheticSource:
    """earth2studio DataSource protocol: ``__call__(time, variable) -> DataArray[time, variable, lat, lon]``."""

    def __init__(self, fields):
        self.fields = fields

    def __call__(self, time, variable):
        from earth2studio.data.utils import prep_data_inputs

        time, variable = prep_data_inputs(time, variable)
        inverse = {v: k for k, v in E2S.items()}
        data = np.stack([np.stack([self.fields[inverse[v]].values[0] for v in variable]) for _ in time])
        coords = {"time": time, "variable": variable, "lat": self.fields["lat"].values, "lon": self.fields["lon"].values}
        return xr.DataArray(data, dims=list(coords), coords=coords)


def test_earth2studio_runner_with_persistence():
    oracle_package("earth2studio", "0.19.0", "requirements-weather.txt")
    from earth2studio.models.px import Persistence

    vortices = [SyntheticVortex(15.0, 140.0), SyntheticVortex(-20.0, 80.0)]
    fields = synthetic_cyclone_fields(vortices, [0], resolution=1.0)
    domain = OrderedDict({"lat": fields["lat"].values, "lon": fields["lon"].values})
    model = Persistence(list(E2S.values()), domain)
    out = run_earth2studio_forecast(model, _SyntheticSource(fields), "2020-01-01T00", lead_hours=24, variables=tuple(E2S))
    assert list(out.data_vars) == list(E2S) and out.sizes["time"] == 5
    assert pd.to_datetime(out["time"].values)[-1] == pd.Timestamp("2020-01-02T00")
    for name in E2S:
        for t in range(out.sizes["time"]):
            np.testing.assert_array_equal(out[name].values[t], fields[name].values[0].astype(np.float32))
    tracks = tempest_tracks({k: out[k].values for k in E2S}, out["lat"].values, out["lon"].values, out["time"].values)
    assert tracks["track_id"].nunique() == 2
    starts = ast.literal_eval(fields.attrs["tracks"])
    for _, track in tracks.groupby("track_id"):
        assert min(float(great_circle_km(track["lat"].iloc[0], track["lon"].iloc[0], s[0][0], s[0][1])) for s in starts) < 80.0
        assert track[["lat", "lon"]].nunique().max() == 1  # persistence: the storm does not move
    with pytest.raises(KeyError, match="does not predict"):
        run_earth2studio_forecast(model, _SyntheticSource(fields), "2020-01-01T00", lead_hours=6, variables=("t2m",))


@pytest.mark.parametrize("model", ["pangu", "graphcast"])
def test_weatherbench2_slice_locates_kong_rey(model):
    oracle_package("zarr", "3.4.0", "requirements-weather.txt")
    oracle_package("gcsfs", "2026.10.0", "requirements-weather.txt")
    ds = read_weatherbench2_forecast(model, "2018-09-30T00", variables=("msl",), lead_hours=[6], lat_bounds=(5, 25), lon_bounds=(128, 148))
    msl = ds["msl"].values[0]
    j, i = np.unravel_index(np.nanargmin(msl), msl.shape)
    distance = float(great_circle_km(ds["lat"].values[j], ds["lon"].values[i], 14.4, 138.0))
    assert distance < 100.0, (model, ds["lat"].values[j], ds["lon"].values[i])
    assert 90000 < msl.min() < 100500
