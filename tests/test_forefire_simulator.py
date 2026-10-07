"""ForeFireSimulator without ForeFire: input conversion, validation and result helpers.

The comparison with the official ForeFire engine lives in tests/oracle/test_forefire_oracle.py.
"""

import subprocess
import sys

import numpy as np
import pytest

from pyhazards.simulators import ForeFireNotInstalledError, ForeFireResult, ForeFireSimulator
from pyhazards.simulators.forefire import _reduce_min

TABLE = "Index;vv_coeff;Kcurv;beta\n1;1.0;1.0;1.0\n2;1.0;1.0;1.0"


def _job(**kwargs):
    sim = ForeFireSimulator("WindDriven", fuels_table=TABLE)
    args = dict(
        fuel=np.ones((4, 6), dtype=np.int32),
        cell_size=10.0,
        duration=60.0,
        ignition_mask=None,
        ignition_points=[(30.0, 20.0)],
        wind=None,
        wind_u=None,
        wind_v=None,
        wind_triggers=None,
        altitude=None,
        output_shape=None,
        origin="upper",
    )
    args.update(kwargs)
    return sim._build_job(**args)


def test_importing_the_adapter_does_not_need_forefire():
    code = (
        "import sys, pyhazards.simulators, pyhazards.datasets.wrf_sfire;"
        "assert 'pyforefire' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_missing_forefire_raises_with_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyforefire", None)  # makes `import pyforefire` fail
    fuel = np.ones((4, 4), dtype=np.int32)
    for isolate in (True, False):
        sim = ForeFireSimulator("Iso", isolate=isolate)
        with pytest.raises(ForeFireNotInstalledError, match="pip install forefire"):
            sim.run(fuel, 10.0, 30.0, ignition_points=[(20.0, 20.0)])


def test_domain_and_layers_follow_forefire_conventions():
    fuel = np.arange(24, dtype=np.int32).reshape(4, 6) % 2 + 1
    job = _job(fuel=fuel, cell_size=(5.0, 10.0), ignition_points=[(30.0, 10.0)])
    assert (job["width"], job["height"]) == (60.0, 20.0)
    assert (job["atmo_nx"], job["atmo_ny"]) == (6, 4)
    # (t, z, y, x) with y growing northward: row 0 of a north-up raster becomes the last y index.
    assert job["fuel"].shape == (1, 1, 4, 6) and job["fuel"].dtype == np.int32
    np.testing.assert_array_equal(job["fuel"][0, 0], fuel[::-1])
    lower = _job(fuel=fuel, origin="lower")
    np.testing.assert_array_equal(lower["fuel"][0, 0], fuel)


def test_ignition_mask_cells_become_cell_centres():
    mask = np.zeros((4, 6), dtype=bool)
    mask[0, 1] = True  # north-west area, top row
    mask[3, 5] = True  # bottom-right
    job = _job(ignition_mask=mask, ignition_points=None)
    assert sorted(job["ignitions"]) == [(15.0, 35.0), (55.0, 5.0)]
    lower = _job(ignition_mask=mask, ignition_points=None, origin="lower")
    assert sorted(lower["ignitions"]) == [(15.0, 5.0), (55.0, 35.0)]


def test_wind_modes():
    constant = _job(wind=(3.0, -1.0))
    assert constant["wind_mode"] == "coefficients" and constant["wind_triggers"] == [(0.0, 3.0, -1.0)]
    # Unit direction fields, as in the official idealized examples.
    np.testing.assert_array_equal(constant["wind_u"][0, 0], np.ones((4, 6)))
    np.testing.assert_array_equal(constant["wind_u"][0, 1], np.zeros((4, 6)))
    np.testing.assert_array_equal(constant["wind_v"][0, 1], np.ones((4, 6)))

    field = _job(wind_u=np.full((2, 3), 2.0), wind_v=np.zeros((2, 3)))
    assert field["wind_mode"] == "field" and field["wind_u"].shape == (1, 1, 2, 3)

    coeff = _job(wind_u=np.ones((2, 2, 3)), wind_v=np.zeros((2, 2, 3)), wind_triggers=[(30.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
    assert coeff["wind_mode"] == "coefficients" and coeff["wind_u"].shape == (1, 2, 2, 3)
    assert coeff["wind_triggers"] == [(0.0, 0.0, 1.0), (30.0, 1.0, 0.0)]
    assert _job()["wind_mode"] == "none"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"fuel": np.ones((4,))}, "2-D raster"),
        ({"fuel": np.full((4, 4), 1.5)}, "integer fuel codes"),
        ({"fuel": np.full((4, 4), 3)}, "not in the Index column"),
        ({"fuel": np.full((4, 4), 2000)}, r"\[0, 1023\]"),
        ({"cell_size": -1.0}, "cell_size"),
        ({"duration": 0.0}, "duration"),
        ({"ignition_points": None}, "ignition"),
        ({"ignition_points": [(500.0, 5.0)]}, "outside the domain"),
        ({"ignition_points": [(5.0, 5.0, 1.0)]}, r"\(x, y\)"),
        ({"wind": (1.0, 0.0), "wind_triggers": [(0.0, 1.0, 0.0)]}, "not both"),
        ({"wind_triggers": [(60.0, 1.0, 0.0)]}, "0 <= t < duration"),
        ({"wind_u": np.ones((2, 2))}, "together"),
        ({"wind_u": np.ones((2, 2)), "wind_v": np.ones((2, 2)), "wind": (1.0, 0.0)}, "direction-coefficient"),
        ({"wind_u": np.ones((3, 2, 2)), "wind_v": np.ones((3, 2, 2))}, r"\(2, h, w\)"),
        ({"altitude": np.ones((2, 2, 2))}, "altitude"),
        ({"output_shape": (0, 3)}, "output_shape"),
        ({"origin": "north"}, "origin"),
    ],
)
def test_invalid_inputs_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _job(**kwargs)


def test_trigger_mode_and_parameters_are_checked():
    with pytest.raises(ValueError, match="trigger_mode"):
        _job(trigger_mode="later")
    for reserved in ("atmoNX", "propagationModel", "fuelsTable"):
        with pytest.raises(ValueError, match="managed by the adapter"):
            ForeFireSimulator("Iso", parameters={reserved: 1})
    with pytest.raises(ValueError, match="int, float or str"):
        ForeFireSimulator("Iso", parameters={"relax": [0.5]})
    sim = ForeFireSimulator("Iso", parameters={"noInitialScan": True, "relax": 0.5})
    assert sim.parameters == {"noInitialScan": 1, "relax": 0.5}
    # Unknown fuel codes can be passed through on request (ForeFire then uses zero properties).
    loose = ForeFireSimulator("WindDriven", fuels_table=TABLE, allow_unknown_fuels=True)
    loose._check_fuel_codes(np.full((2, 2), 7.0))


def test_reduce_min_takes_earliest_arrival_per_output_cell():
    native = np.arange(36, dtype=float).reshape(6, 6)
    native[0, 0] = np.inf
    out = _reduce_min(native, (2, 3))
    assert out.shape == (2, 3)
    assert out[0, 0] == 1.0  # min over rows 0-2, cols 0-1, skipping the inf
    assert out[1, 2] == native[3:6, 4:6].min()
    np.testing.assert_array_equal(_reduce_min(native, (6, 6)), native)


def test_result_masks_and_area():
    arrival = np.array([[0.0, 10.0], [np.inf, 30.0]])
    result = ForeFireResult(
        arrival_time=arrival,
        arrival_time_native=arrival,
        duration=30.0,
        cell_size=(5.0, 5.0),
        native_cell_size=(5.0, 5.0),
        origin="upper",
    )
    np.testing.assert_array_equal(result.burned_mask(), [[True, True], [False, True]])
    np.testing.assert_array_equal(result.burned_mask(10.0), [[True, True], [False, False]])
    stack = result.burned_masks([0.0, 30.0])
    assert stack.shape == (2, 2, 2) and stack.dtype == np.float32
    assert result.burned_area() == 75.0
