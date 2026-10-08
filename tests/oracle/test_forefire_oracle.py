"""ForeFireSimulator checked against the official ForeFire engine, its reference output and an example.

ForeFire 2.5.0 (https://github.com/forefireAPI/forefire, GPL-3.0) is installed from its official
wheel (tests/oracle/requirements-forefire.txt) for this test only. The pinned repository supplies the
official inputs; nothing from it is copied into PyHazards. Inputs are parsed from the official files
rather than retyped here.

1. ``tests/runff`` is ForeFire's own CI regression case (64 km Corsican landscape, Rothermel, three
   ignitions, a wind change at t = 1200 s). Fed the case's rasters, parameters (``params.ff``),
   ignitions and wind change (``real_case.ff``), the adapter must reproduce the reference
   arrival-time map ``ForeFire.0.nc.ref`` within the tolerance of the case's ``compare_nc.py``
   (rtol 1e-5, atol 1e-8), with exactly the same burned cells. The ``.ff`` script schedules the wind
   change as an event, hence ``trigger_mode="event"``.
2. ``tests/python/idealizedwind.py`` is run unchanged in a separate process; the adapter run of the
   same scenario (``trigger_mode="step"``, the script's own pattern) must give the identical burning
   map.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from oracle_utils import oracle_asset, oracle_package, oracle_repo

from pyhazards.simulators import ForeFireSimulator

FOREFIRE_VERSION = "2.5.0"


@pytest.fixture(scope="module")
def forefire():
    return oracle_package("pyforefire", FOREFIRE_VERSION, "requirements-forefire.txt", distribution="forefire")


def _ff_arguments(text: str, command: str) -> list[str]:
    return re.findall(rf"^\s*{command}\[(.*?)\](.*)$", text, flags=re.MULTILINE)


def _runff_case(repo: Path):
    """Parameters, ignitions, wind change and duration of tests/runff/real_case.ff."""
    case = repo / "tests" / "runff"
    params = {}
    for argument, _ in _ff_arguments((case / "params.ff").read_text(), "setParameter"):
        key, value = argument.split("=", 1)
        params[key.strip()] = value.strip()
    script = (case / "real_case.ff").read_text()
    points = []
    for argument, _ in _ff_arguments(script, "startFire"):
        x, y, _z = re.search(r"loc=\(([^)]*)\)", argument).group(1).split(",")
        assert re.search(r"t=0\b", argument), argument
        points.append((float(x), float(y)))
    triggers = []
    for argument, suffix in _ff_arguments(script, "trigger"):
        u, v, _w = re.search(r"vel=\(([^)]*)\)", argument).group(1).split(",")
        when = float(re.search(r"@t=([0-9.]+)", suffix).group(1)) if suffix.strip() else 0.0
        triggers.append((when, float(u), float(v)))
    (step, _), = _ff_arguments(script, "step")
    duration = float(step.split("=", 1)[1])
    table = (case / params.pop("fuelsTableFile")).read_text()
    model = params.pop("propagationModel")
    return model, table, params, points, triggers, duration


@pytest.fixture(scope="module")
def runff(forefire):
    xr = pytest.importorskip("xarray")
    repo = oracle_repo("forefire")
    model, table, params, points, triggers, duration = _runff_case(repo)
    assert (model, len(points), len(triggers), duration) == ("Rothermel", 3, 1, 2400.0)
    data = xr.open_dataset(oracle_asset("forefire_runff_data") / "data.nc")
    domain = data["domain"].attrs
    fuel = np.asarray(data["fuel"].values[0, 0])
    inputs = dict(
        fuel=fuel,
        cell_size=(float(domain["Ly"]) / fuel.shape[0], float(domain["Lx"]) / fuel.shape[1]),
        duration=duration,
        ignition_points=points,
        wind_u=np.asarray(data["wind"].values[0], dtype=np.float64),
        wind_v=np.asarray(data["wind"].values[1], dtype=np.float64),
        wind_triggers=triggers,
        altitude=np.asarray(data["altitude"].values[0, 0], dtype=np.float64),
        # loadData[] sets ForeFire's flux grid (atmoNX, atmoNY) from the file's nx, ny dimensions.
        output_shape=(data.sizes["ny"], data.sizes["nx"]),
        origin="lower",  # ForeFire landscape files store rows south to north
        trigger_mode="event",
    )
    reference = xr.open_dataset(oracle_asset("forefire_runff_reference") / "ForeFire.0.nc.ref")
    expected = np.asarray(reference["arrival_time_of_front"].values)
    return ForeFireSimulator(model, fuels_table=table, parameters=params), inputs, expected


def test_runff_reproduces_the_reference_arrival_times(runff):
    simulator, inputs, expected = runff
    result = simulator.run(**inputs)
    native = result.arrival_time_native
    assert native.shape == expected.shape == (3200, 3200)
    assert result.native_cell_size == (20.0, 20.0)
    assert result.engine_version == FOREFIRE_VERSION
    # save[] writes -9999 where the front never arrived; the adapter returns inf there.
    ours = np.where(np.isinf(native), -9999.0, native)
    np.testing.assert_array_equal(ours > -9999, expected > -9999)
    assert (expected > -9999).sum() > 900
    np.testing.assert_allclose(ours, expected, rtol=1e-5, atol=1e-8)

    # The output grid is ForeFire's 640 x 640 flux grid: earliest arrival per 5 x 5 burning-map block.
    blocks = np.where(expected > -9999, expected, np.inf).reshape(640, 5, 640, 5).min(axis=(1, 3))
    np.testing.assert_allclose(result.arrival_time, blocks, rtol=1e-5, atol=1e-8)
    assert result.cell_size == (100.0, 100.0)
    np.testing.assert_allclose(result.burned_area(), (expected > -9999).sum() * 400.0)


def test_runff_north_up_and_in_process_runs_agree(runff):
    simulator, inputs, _ = runff
    lower = simulator.run(**inputs)
    upper_inputs = dict(
        inputs,
        fuel=inputs["fuel"][::-1],
        altitude=inputs["altitude"][::-1],
        wind_u=inputs["wind_u"][:, ::-1],
        wind_v=inputs["wind_v"][:, ::-1],
        origin="upper",
    )
    upper = ForeFireSimulator(
        simulator.propagation_model, simulator.fuels_table, simulator.parameters, isolate=False
    ).run(**upper_inputs)
    np.testing.assert_array_equal(upper.arrival_time_native, lower.arrival_time_native[::-1])
    np.testing.assert_array_equal(upper.arrival_time, lower.arrival_time[::-1])


_IDEALIZED_RUNNER = """
import json, runpy, sys
import matplotlib
matplotlib.use("Agg")
import numpy as np
g = runpy.run_path(sys.argv[1])
ff = g["ff"]
np.save("official_arrival.npy", np.array(ff["arrival_time"]))
keys = sys.argv[2].split(",")
json.dump({
    "table": g["VVCoeffTable"](),
    "params": {key: ff.getString(key) for key in keys},
    "model": ff.getString("propagationModel"),
    "nb_steps_tot": g["nb_steps_tot"], "step_size": g["step_size"], "norm": g["norm"],
    "angle_deg": g["angle_deg"], "sim_shape": list(g["sim_shape"]),
    "start": [g["startx"], g["starty"]],
}, open("scenario.json", "w"))
"""

# Parameters idealizedwind.py sets with ff[...] (values are read back from the script's run).
_IDEALIZED_KEYS = (
    "defaultFuelType,spatialIncrement,minimalPropagativeFrontDepth,perimeterResolution,"
    "initialFrontDepth,relax,smoothing,minSpeed,bmapLayer,windReductionFactor"
)


def test_idealized_wind_example_is_reproduced_exactly(forefire, tmp_path):
    script = oracle_repo("forefire") / "tests" / "python" / "idealizedwind.py"
    env = dict(os.environ, MPLBACKEND="Agg")
    subprocess.run(
        [sys.executable, "-c", _IDEALIZED_RUNNER, str(script), _IDEALIZED_KEYS],
        cwd=tmp_path, env=env, check=True, capture_output=True, text=True,
    )
    official = np.load(tmp_path / "official_arrival.npy")
    scenario = json.loads((tmp_path / "scenario.json").read_text())

    width, height = scenario["sim_shape"]
    step, norm = float(scenario["step_size"]), float(scenario["norm"])
    triggers = []
    for i in range(scenario["nb_steps_tot"] + 1):  # the script triggers, then steps, each iteration
        angle = math.radians(i * scenario["angle_deg"])
        triggers.append((i * step, norm * math.cos(angle), norm * math.sin(angle)))
    unit_u = np.zeros((2, height, width))
    unit_u[0] = 1.0
    unit_v = np.zeros((2, height, width))
    unit_v[1] = 1.0
    simulator = ForeFireSimulator(scenario["model"], fuels_table=scenario["table"], parameters=scenario["params"])
    result = simulator.run(
        np.ones((height, width), dtype=np.int32),
        cell_size=1.0,
        duration=len(triggers) * step,
        ignition_points=[tuple(float(v) for v in scenario["start"])],
        wind_u=unit_u,
        wind_v=unit_v,
        wind_triggers=triggers,
        output_shape=(100, 100),  # ForeFire's default flux grid (atmoNX = atmoNY = 100), unset by the script
        origin="lower",
        trigger_mode="step",
    )
    assert np.isfinite(official).sum() > 1000
    np.testing.assert_array_equal(result.arrival_time_native, official)
