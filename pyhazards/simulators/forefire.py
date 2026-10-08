"""Adapter that runs the official ForeFire wildland-fire spread simulator on PyHazards rasters.

ForeFire is a C++ discrete-event front-tracking simulator developed at CNRS / Universite de Corse:

- Filippi, Baggio, Paugam, Bosseur, Leblanc, Alonso-Pinar, "ForeFire: A Modular, Scriptable C++
  Simulation Engine and Library for Wildland-Fire Spread", Journal of Open Source Software 10(116):
  8680 (2025), doi:10.21105/joss.08680.
- Filippi, Morandini, Balbi, Hill, "Discrete event front-tracking simulation of a physical
  fire-spread model", SIMULATION 86(10):629-646, doi:10.1177/0037549709343117.
- Official code: https://github.com/forefireAPI/forefire. Python bindings: module ``pyforefire``,
  PyPI distribution ``forefire`` (checked with 2.5.0, tag ``v2.5.0``).

Licence boundary
----------------
ForeFire is GPL-3.0 and PyHazards is MIT. This module contains no ForeFire code and does not link
against it: it imports a ``pyforefire`` that the user installed separately (``pip install forefire``)
and drives it only through its public Python API, lazily, at run time. ForeFire is not a PyHazards
dependency. Whoever redistributes PyHazards together with ForeFire must follow the GPL.

What this adapter does
----------------------
It turns rasters into ForeFire layers with the same calls as the official Python examples
(``tests/python/percolation.py``, ``idealizedwind.py`` and ``farsite_flat.py`` in the ForeFire repo):
a ``FireDomain`` covering the fuel raster, the propagation model (``addLayer("propagation", ...)``),
the fuel index map (``addIndexLayer("table", "fuel", ...)``), optional altitude and wind
(``addScalarLayer``), ignitions (``startFire[loc=...]``) and wind changes (``trigger[wind;vel=...]``),
then advances the simulation with ``step[dt=...]``: either step by step between wind changes, like the
Python examples, or with the changes scheduled as events (``trigger[...]@t=...``) like
``tests/runff/real_case.ff``. It reads back ForeFire's burning map (``ff["arrival_time"]``: arrival
time of the front in seconds, ``inf`` where the fire never arrived) and, as the only PyHazards-side
post-processing, reduces that map to the requested output grid.

Checked against the official code (``tests/oracle/test_forefire_oracle.py``): fed the rasters of the
official regression case ``tests/runff`` it reproduces the reference arrival-time map
``ForeFire.0.nc.ref`` within the tolerance of ForeFire's own ``compare_nc.py``, and it reproduces the
official ``tests/python/idealizedwind.py`` run bit for bit.

What it does not do
-------------------
It does not modify ForeFire's propagation models, fuel tables or parameters, does not calibrate
anything, and has no learned weights. Physical realism depends entirely on the fuel table, the
propagation model and the parameters the user chooses. It is not a ``torch.nn.Module`` and is not in
the model registry.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import math
import os
import pickle
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

FOREFIRE_DISTRIBUTION = "forefire"
FOREFIRE_MODULE = "pyforefire"
FOREFIRE_CHECKED_VERSION = "2.5.0"
FOREFIRE_REPO = "https://github.com/forefireAPI/forefire"

# Fuel tables shipped with pyforefire.helpers (read from the installed package at run time).
_HELPER_FUEL_TABLES = ("Rothermel", "RothermelAndrews2018")

# ForeFire stores fuel properties in a fixed table of 1024 rows (FuelDataLayer::MAXNUMFUELS); a larger
# code would index past it in the C++ core.
_MAX_FUEL_CODE = 1023

# Parameters the adapter sets itself; passing them in ``parameters`` would silently fight the inputs.
_RESERVED_PARAMETERS = {
    "atmoNX": "set from output_shape",
    "atmoNY": "set from output_shape",
    "propagationModel": "use ForeFireSimulator(propagation_model=...)",
    "fuelsTable": "use ForeFireSimulator(fuels_table=...)",
}

_ORIGINS = ("upper", "lower")

ArrayLike = Any


class ForeFireNotInstalledError(ImportError):
    """Raised when the optional, separately installed ``pyforefire`` module is missing."""


def forefire_available() -> bool:
    """Whether ``pyforefire`` can be imported (it is never installed by PyHazards)."""
    try:
        importlib.import_module(FOREFIRE_MODULE)
    except ImportError:
        return False
    return True


def forefire_version() -> Optional[str]:
    try:
        return importlib.metadata.version(FOREFIRE_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return None


def _import_pyforefire():
    try:
        return importlib.import_module(FOREFIRE_MODULE)
    except ImportError as exc:
        raise ForeFireNotInstalledError(
            "ForeFireSimulator needs the official ForeFire Python bindings, which PyHazards does not "
            "install (ForeFire is GPL-3.0). Install them yourself with `pip install forefire` "
            f"(checked with {FOREFIRE_CHECKED_VERSION}); see {FOREFIRE_REPO}."
        ) from exc


@dataclass
class ForeFireResult:
    """Output of :meth:`ForeFireSimulator.run`.

    ``arrival_time`` is on the output grid (default: the fuel grid) with the row order given by
    ``origin``: the earliest ForeFire arrival time, in seconds since ``t=0``, of any burning-map cell
    inside each output cell, ``inf`` where the front never arrived within ``duration``.
    ``arrival_time_native`` is ForeFire's own burning map at ``native_cell_size`` resolution, in the
    same row order.
    """

    arrival_time: np.ndarray
    arrival_time_native: np.ndarray
    duration: float
    cell_size: Tuple[float, float]
    native_cell_size: Tuple[float, float]
    origin: str
    engine_version: Optional[str] = None
    log: str = ""
    commands: List[str] = field(default_factory=list)

    def burned_mask(self, time: Optional[float] = None, native: bool = False) -> np.ndarray:
        """Boolean mask of cells reached by the front at ``time`` seconds (default: ``duration``)."""
        when = self.duration if time is None else float(time)
        grid = self.arrival_time_native if native else self.arrival_time
        return grid <= when

    def burned_masks(self, times: Sequence[float], native: bool = False) -> np.ndarray:
        """Stack of burned masks, shape ``(len(times), H, W)``, dtype ``float32`` (spread targets)."""
        return np.stack([self.burned_mask(t, native=native) for t in times]).astype(np.float32)

    def burned_area(self, time: Optional[float] = None) -> float:
        """Burned area in square metres at ``time``, from the native burning map."""
        dy, dx = self.native_cell_size
        return float(self.burned_mask(time, native=True).sum()) * dx * dy


@contextlib.contextmanager
def _captured_native_stdout() -> Iterator[List[str]]:
    """Capture what ForeFire's C++ core writes to ``std::cout`` (file descriptor 1).

    Same technique as the official ``tests/python/test_wheel.py``: ``sys.stdout`` never sees it.
    """
    box: List[str] = []
    try:
        fd = sys.stdout.fileno()
    except (AttributeError, OSError, ValueError):
        fd = 1
    with tempfile.TemporaryFile(mode="w+") as tmp:
        sys.stdout.flush()
        saved = os.dup(fd)
        try:
            os.dup2(tmp.fileno(), fd)
            yield box
        finally:
            sys.stdout.flush()
            os.dup2(saved, fd)
            os.close(saved)
            tmp.seek(0)
            box.append(tmp.read())


def _num(value: float) -> str:
    return repr(float(value))


def _as_array(value: ArrayLike, name: str, dtype=np.float64) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):  # torch.Tensor without importing torch
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype == object:
        raise ValueError(f"{name} must be a numeric array, got dtype=object")
    return array.astype(dtype, copy=False)


def _layer(array: np.ndarray, origin: str) -> np.ndarray:
    """(…, H, W) raster -> ForeFire layer array (t, z, y, x) with y increasing northward."""
    if origin == "upper":
        array = array[..., ::-1, :]
    if array.ndim == 2:
        array = array[None, None]
    elif array.ndim == 3:
        array = array[None]
    return np.ascontiguousarray(array)


def _reduce_min(native: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Minimum of ``native`` over the native cells whose centres fall in each output cell."""
    hn, wn = native.shape
    ho, wo = shape
    rows = np.minimum(((np.arange(hn) + 0.5) * ho / hn).astype(np.int64), ho - 1)
    cols = np.minimum(((np.arange(wn) + 0.5) * wo / wn).astype(np.int64), wo - 1)
    out = np.full(shape, np.inf, dtype=np.float64)
    np.minimum.at(out, (rows[:, None], cols[None, :]), native)
    return out


def _run_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one ForeFire run. Plain data in and out, so it can run in a separate process."""
    pyforefire = _import_pyforefire()
    commands: List[str] = []

    def execute(ff, command: str) -> str:
        commands.append(command)
        return ff.execute(command)

    with _captured_native_stdout() as captured:
        ff = pyforefire.ForeFire()
        table = job["fuels_table"]
        if table is None and job["propagation_model"] in _HELPER_FUEL_TABLES:
            table = pyforefire.helpers.get_fuels_table(job["propagation_model"])()
        if table is not None:
            ff["fuelsTable"] = table
        ff["propagationModel"] = job["propagation_model"]
        for key, value in job["parameters"].items():
            ff[key] = value
        ff["atmoNX"] = int(job["atmo_nx"])
        ff["atmoNY"] = int(job["atmo_ny"])

        width, height = float(job["width"]), float(job["height"])
        execute(ff, f"FireDomain[sw=(0.,0.,0.);ne=({_num(width)},{_num(height)},0.);t=0.]")
        ff.addLayer("propagation", job["propagation_model"], "propagationModel")
        extent = (0.0, 0.0, 0.0, width, height, 0.0)
        ff.addIndexLayer("table", "fuel", *extent, job["fuel"])
        if job["altitude"] is not None:
            ff.addScalarLayer("data", "altitude", *extent, job["altitude"])
        if job["wind_mode"] == "field":
            ff.addScalarLayer("data", "windU", *extent, job["wind_u"])
            ff.addScalarLayer("data", "windV", *extent, job["wind_v"])
        elif job["wind_mode"] == "coefficients":
            ff.addScalarLayer("windScalDir", "windU", *extent, job["wind_u"])
            ff.addScalarLayer("windScalDir", "windV", *extent, job["wind_v"])

        for x, y in job["ignitions"]:
            execute(ff, f"startFire[loc=({_num(x)},{_num(y)},0.);t=0.]")

        def trigger(u: float, v: float) -> str:
            return f"trigger[wind;loc=(0.,0.,0.);vel=({_num(u)},{_num(v)},0.)]"

        triggers, duration = job["wind_triggers"], float(job["duration"])
        if job["trigger_mode"] == "event":
            # .ff-script style (tests/runff/real_case.ff): later triggers are ForeFire events at t.
            for t, u, v in triggers:
                execute(ff, trigger(u, v) + (f"@t={_num(t)}" if t > 0 else ""))
            execute(ff, f"step[dt={_num(duration)}]")
        else:
            # Python-example style (tests/python/idealizedwind.py, farsite_flat.py): advance to each
            # trigger time with step[], then trigger the new wind.
            now = 0.0
            for boundary in sorted({t for t, _, _ in triggers} | {duration}):
                if boundary > now:
                    execute(ff, f"step[dt={_num(boundary - now)}]")
                    now = boundary
                for t, u, v in triggers:
                    if t == boundary:
                        execute(ff, trigger(u, v))
        native = np.array(ff["arrival_time"], dtype=np.float64)

    return {
        "arrival_time_native": native,
        "log": captured[0] if captured else "",
        "commands": commands,
        "engine_version": forefire_version(),
    }


def worker_main(argv: Sequence[str]) -> int:
    """Entry point of the isolated worker: ``<job.pkl> <out.pkl>`` (both written by this module)."""
    if len(argv) != 2:
        print("usage: python -m pyhazards.simulators._forefire_worker JOB.pkl OUT.pkl", file=sys.stderr)
        return 2
    job_path, out_path = Path(argv[0]), Path(argv[1])
    with job_path.open("rb") as handle:
        job = pickle.load(handle)
    try:
        output = _run_job(job)
    except Exception:  # reported to the parent, which raises it
        output = {"error": traceback.format_exc()}
    with out_path.open("wb") as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return 0


class ForeFireSimulator:
    """Run the official ForeFire simulator (user-installed ``pyforefire``) on PyHazards rasters.

    Args:
        propagation_model: a ForeFire rate-of-spread model name, e.g. ``"Rothermel"``,
            ``"RothermelAndrews2018"``, ``"Farsite"``, ``"Balbi2020"``, ``"WindDriven"`` or ``"Iso"``.
        fuels_table: ForeFire fuel table: the ``;``-separated text whose ``Index`` column holds the
            codes used in the fuel raster, or the name of a table built into ForeFire (for example
            ``"STDfarsiteFuelsTable"``). ``None`` uses the table that ``pyforefire.helpers`` ships for
            ``Rothermel`` / ``RothermelAndrews2018`` and ForeFire's own default otherwise.
        parameters: extra ForeFire parameters set with ``ff[name] = value`` before the domain is
            created (for example ``spatialIncrement``, ``perimeterResolution``,
            ``minimalPropagativeFrontDepth``, ``windReductionFactor``, ``Iso.speed``). Anything not
            given keeps ForeFire's default.
        isolate: run each simulation in a fresh process (default). ForeFire keeps its parameters in
            a process-wide singleton, so in-process runs (``isolate=False``) inherit parameters set
            by earlier runs in the same process.
        timeout: optional limit in seconds for an isolated run.
        allow_unknown_fuels: ForeFire gives fuel codes that are missing from the table all-zero fuel
            properties without a warning. By default the adapter refuses such codes (when the table
            text is given); set this to ``True`` to pass them through unchanged.
    """

    def __init__(
        self,
        propagation_model: str = "Rothermel",
        fuels_table: Optional[str] = None,
        parameters: Optional[Mapping[str, Union[int, float, str, bool]]] = None,
        isolate: bool = True,
        timeout: Optional[float] = None,
        allow_unknown_fuels: bool = False,
    ):
        if not isinstance(propagation_model, str) or not propagation_model:
            raise ValueError("propagation_model must be a non-empty ForeFire model name")
        if fuels_table is not None and not isinstance(fuels_table, str):
            raise ValueError("fuels_table must be the table text or the name of a built-in table")
        self.propagation_model = propagation_model
        self.fuels_table = fuels_table
        self.parameters = self._check_parameters(parameters or {})
        self.isolate = bool(isolate)
        self.timeout = timeout
        self.allow_unknown_fuels = bool(allow_unknown_fuels)

    @staticmethod
    def _check_parameters(parameters: Mapping[str, Any]) -> Dict[str, Union[int, float, str]]:
        checked: Dict[str, Union[int, float, str]] = {}
        for key, value in parameters.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"ForeFire parameter names must be non-empty strings, got {key!r}")
            if key in _RESERVED_PARAMETERS:
                raise ValueError(f"ForeFire parameter {key!r} is managed by the adapter: {_RESERVED_PARAMETERS[key]}")
            if isinstance(value, bool):
                value = int(value)
            if not isinstance(value, (int, float, str)):
                raise ValueError(f"ForeFire parameter {key!r} must be int, float or str, got {type(value).__name__}")
            checked[key] = value
        return checked

    def run(
        self,
        fuel: ArrayLike,
        cell_size: Union[float, Tuple[float, float]],
        duration: float,
        *,
        ignition_mask: Optional[ArrayLike] = None,
        ignition_points: Optional[Sequence[Sequence[float]]] = None,
        wind: Optional[Tuple[float, float]] = None,
        wind_u: Optional[ArrayLike] = None,
        wind_v: Optional[ArrayLike] = None,
        wind_triggers: Optional[Sequence[Tuple[float, float, float]]] = None,
        altitude: Optional[ArrayLike] = None,
        output_shape: Optional[Tuple[int, int]] = None,
        origin: str = "upper",
        trigger_mode: str = "step",
    ) -> ForeFireResult:
        """Simulate ``duration`` seconds of spread.

        Rasters are 2-D arrays ``(rows, cols)``. ``origin="upper"`` (default, GeoTIFF / image order)
        means row 0 is the northern edge; ``origin="lower"`` means row 0 is the southern edge, which is
        ForeFire's own array order. The simulation domain spans ``cols * dx`` by ``rows * dy`` metres of
        the fuel raster with its south-west corner at ``(0, 0)``; every other raster covers the same
        extent at its own resolution (ForeFire resamples it).

        Args:
            fuel: integer fuel codes of the ``fuels_table`` (``Index`` column).
            cell_size: fuel-raster cell size in metres, ``dx`` or ``(dy, dx)``.
            duration: simulated time in seconds from the ignition at ``t=0``.
            ignition_mask: boolean raster over the domain; ForeFire starts a fire
                (``startFire[loc=...]``) at the centre of every ``True`` cell at ``t=0``.
            ignition_points: ``(x, y)`` ignition locations in metres from the south-west corner,
                ignited at ``t=0``.
            wind: constant wind ``(u, v)`` in m/s (eastward, northward); shorthand for
                ``wind_triggers=[(0, u, v)]``.
            wind_u, wind_v: either ``(h, w)`` wind fields in m/s (ForeFire ``windU``/``windV`` data
                layers, held constant), or ``(2, h, w)`` ForeFire direction-coefficient fields
                (``windScalDir`` layers, as in ForeFire landscape files) that ``wind_triggers`` scale.
            wind_triggers: ``(t, u, v)`` uniform wind changes (``trigger[wind;vel=(u,v,0)]`` at
                ``t``), as in the official examples. Without ``wind_u``/``wind_v`` the adapter uses
                unit direction fields, so the wind is exactly ``(u, v)`` everywhere.
            altitude: terrain height in metres.
            output_shape: ``(rows, cols)`` of the returned ``arrival_time`` (default: fuel shape).
                It is also ForeFire's flux grid (``atmoNX``/``atmoNY``), whose cells ForeFire splits
                into burning-map cells of ``max(spatialIncrement / sqrt(2),
                minimalPropagativeFrontDepth)`` metres.
            origin: row order of every input and output raster, ``"upper"`` or ``"lower"``.
            trigger_mode: how wind changes are issued. ``"step"`` (default) advances the simulation
                with ``step[dt=...]`` to each trigger time and then triggers the new wind, as the
                official Python examples do; ``"event"`` schedules every trigger as a ForeFire event
                (``trigger[...]@t=...``) and steps once, as the official ``.ff`` scripts do. ForeFire's
                results differ slightly between the two (the extra step boundaries change its
                internal time stepping).
        """
        job = self._build_job(
            fuel=fuel,
            cell_size=cell_size,
            duration=duration,
            ignition_mask=ignition_mask,
            ignition_points=ignition_points,
            wind=wind,
            wind_u=wind_u,
            wind_v=wind_v,
            wind_triggers=wind_triggers,
            altitude=altitude,
            output_shape=output_shape,
            origin=origin,
            trigger_mode=trigger_mode,
        )
        if self.isolate:
            output = self._run_isolated(job)
        else:
            output = _run_job(job)

        native = output["arrival_time_native"]
        if native.ndim != 2 or native.size == 0:
            raise RuntimeError(f"ForeFire returned no burning map (shape {native.shape}); log:\n{output['log']}")
        out_shape = (job["atmo_ny"], job["atmo_nx"])
        reduced = _reduce_min(native, out_shape)
        if origin == "upper":
            native = native[::-1]
            reduced = reduced[::-1]
        return ForeFireResult(
            arrival_time=np.ascontiguousarray(reduced),
            arrival_time_native=np.ascontiguousarray(native),
            duration=float(job["duration"]),
            cell_size=(job["height"] / out_shape[0], job["width"] / out_shape[1]),
            native_cell_size=(job["height"] / native.shape[0], job["width"] / native.shape[1]),
            origin=origin,
            engine_version=output.get("engine_version"),
            log=output.get("log", ""),
            commands=list(output.get("commands", [])),
        )

    def _run_isolated(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Run the job in a fresh interpreter (``python -m pyhazards.simulators._forefire_worker``).

        A plain subprocess, unlike multiprocessing's spawn start method, does not re-import the
        caller's ``__main__``, so scripts need no ``if __name__ == "__main__"`` guard.
        """
        _import_pyforefire()  # fail fast, in this process, with the installation hint
        package_root = str(Path(__file__).resolve().parents[2])
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in (package_root, env.get("PYTHONPATH", "")) if p)
        with tempfile.TemporaryDirectory(prefix="pyhazards_forefire_") as tmp:
            job_path, out_path = Path(tmp) / "job.pkl", Path(tmp) / "out.pkl"
            with job_path.open("wb") as handle:
                pickle.dump(job, handle, protocol=pickle.HIGHEST_PROTOCOL)
            command = [sys.executable, "-m", "pyhazards.simulators._forefire_worker", str(job_path), str(out_path)]
            proc = subprocess.run(command, capture_output=True, text=True, timeout=self.timeout, env=env)
            if not out_path.exists():
                raise RuntimeError(
                    f"ForeFire worker exited with code {proc.returncode} without a result.\n"
                    f"stderr (tail):\n{proc.stderr[-4000:]}\nstdout (tail):\n{proc.stdout[-4000:]}"
                )
            with out_path.open("rb") as handle:
                output = pickle.load(handle)
        if "error" in output:
            raise RuntimeError(f"ForeFire run failed in the worker process:\n{output['error']}")
        return output

    def _build_job(
        self,
        fuel: ArrayLike,
        cell_size: Union[float, Tuple[float, float]],
        duration: float,
        ignition_mask: Optional[ArrayLike],
        ignition_points: Optional[Sequence[Sequence[float]]],
        wind: Optional[Tuple[float, float]],
        wind_u: Optional[ArrayLike],
        wind_v: Optional[ArrayLike],
        wind_triggers: Optional[Sequence[Tuple[float, float, float]]],
        altitude: Optional[ArrayLike],
        output_shape: Optional[Tuple[int, int]],
        origin: str,
        trigger_mode: str = "step",
    ) -> Dict[str, Any]:
        if origin not in _ORIGINS:
            raise ValueError(f"origin must be 'upper' or 'lower', got {origin!r}")
        if trigger_mode not in ("step", "event"):
            raise ValueError(f"trigger_mode must be 'step' or 'event', got {trigger_mode!r}")

        fuel_values = _as_array(fuel, "fuel")
        if fuel_values.ndim != 2 or min(fuel_values.shape) < 2:
            raise ValueError(f"fuel must be a 2-D raster (rows, cols) of at least 2x2, got shape {fuel_values.shape}")
        if not np.all(np.isfinite(fuel_values)) or not np.all(fuel_values == np.round(fuel_values)):
            raise ValueError("fuel must hold integer fuel codes (the Index column of the fuel table)")
        if fuel_values.min() < 0 or fuel_values.max() > _MAX_FUEL_CODE:
            raise ValueError(f"ForeFire fuel codes must lie in [0, {_MAX_FUEL_CODE}], got [{fuel_values.min():g}, {fuel_values.max():g}]")
        self._check_fuel_codes(fuel_values)

        if np.ndim(cell_size) == 0:
            dy = dx = float(cell_size)
        else:
            if len(cell_size) != 2:
                raise ValueError(f"cell_size must be a number or (dy, dx), got {cell_size!r}")
            dy, dx = (float(v) for v in cell_size)
        if not (math.isfinite(dx) and math.isfinite(dy) and dx > 0 and dy > 0):
            raise ValueError(f"cell_size must be positive, got {cell_size!r}")
        rows, cols = fuel_values.shape
        width, height = cols * dx, rows * dy

        duration = float(duration)
        if not (math.isfinite(duration) and duration > 0):
            raise ValueError(f"duration must be a positive number of seconds, got {duration!r}")

        if output_shape is None:
            out_rows, out_cols = rows, cols
        else:
            if len(output_shape) != 2 or any(int(v) != v or v < 1 for v in output_shape):
                raise ValueError(f"output_shape must be (rows, cols) of positive ints, got {output_shape!r}")
            out_rows, out_cols = (int(v) for v in output_shape)

        ignitions = self._ignitions(ignition_mask, ignition_points, width, height, origin)

        altitude_layer = None
        if altitude is not None:
            altitude_values = _as_array(altitude, "altitude")
            if altitude_values.ndim != 2 or not np.all(np.isfinite(altitude_values)):
                raise ValueError(f"altitude must be a finite 2-D raster, got shape {altitude_values.shape}")
            altitude_layer = _layer(altitude_values, origin)

        triggers: List[Tuple[float, float, float]] = []
        if wind is not None:
            if wind_triggers is not None:
                raise ValueError("give either wind=(u, v) or wind_triggers, not both")
            if len(wind) != 2:
                raise ValueError(f"wind must be (u, v), got {wind!r}")
            triggers = [(0.0, float(wind[0]), float(wind[1]))]
        elif wind_triggers is not None:
            for trigger in wind_triggers:
                if len(trigger) != 3:
                    raise ValueError(f"wind_triggers entries must be (t, u, v), got {trigger!r}")
                t, u, v = (float(item) for item in trigger)
                if not (math.isfinite(t) and 0.0 <= t < duration and math.isfinite(u) and math.isfinite(v)):
                    raise ValueError(f"wind trigger {trigger!r} must have 0 <= t < duration and finite (u, v)")
                triggers.append((t, u, v))
            triggers.sort(key=lambda item: item[0])

        wind_mode = "none"
        wind_u_layer = wind_v_layer = None
        if (wind_u is None) != (wind_v is None):
            raise ValueError("wind_u and wind_v must be given together")
        if wind_u is not None:
            u_values = _as_array(wind_u, "wind_u")
            v_values = _as_array(wind_v, "wind_v")
            if u_values.shape != v_values.shape:
                raise ValueError(f"wind_u and wind_v shapes differ: {u_values.shape} vs {v_values.shape}")
            if not (np.all(np.isfinite(u_values)) and np.all(np.isfinite(v_values))):
                raise ValueError("wind_u and wind_v must be finite")
            if u_values.ndim == 2:
                if triggers:
                    raise ValueError(
                        "2-D wind_u/wind_v are fixed wind fields; use (2, h, w) direction-coefficient "
                        "fields to combine spatial wind with wind/wind_triggers"
                    )
                wind_mode = "field"
            elif u_values.ndim == 3 and u_values.shape[0] == 2:
                wind_mode = "coefficients"
            else:
                raise ValueError(
                    "wind_u/wind_v must have shape (h, w) (wind field) or (2, h, w) (ForeFire "
                    f"direction coefficients), got {u_values.shape}"
                )
            wind_u_layer = _layer(u_values, origin)
            wind_v_layer = _layer(v_values, origin)
        elif triggers:
            # Unit direction fields, as in the official idealized examples: the effective wind is
            # exactly the triggered (u, v). One value per output cell keeps ForeFire's bilinear
            # interpolation defined everywhere except half an output cell along the domain edge.
            shape = (2, out_rows, out_cols)
            unit_u = np.zeros(shape)
            unit_u[0] = 1.0
            unit_v = np.zeros(shape)
            unit_v[1] = 1.0
            wind_mode = "coefficients"
            wind_u_layer = _layer(unit_u, "lower")
            wind_v_layer = _layer(unit_v, "lower")

        return {
            "propagation_model": self.propagation_model,
            "fuels_table": self.fuels_table,
            "parameters": dict(self.parameters),
            "atmo_nx": out_cols,
            "atmo_ny": out_rows,
            "width": width,
            "height": height,
            "duration": duration,
            "fuel": _layer(fuel_values.astype(np.int32), origin),
            "altitude": altitude_layer,
            "wind_mode": wind_mode,
            "wind_u": wind_u_layer,
            "wind_v": wind_v_layer,
            "ignitions": ignitions,
            "wind_triggers": triggers,
            "trigger_mode": trigger_mode,
        }

    def _check_fuel_codes(self, fuel_values: np.ndarray) -> None:
        table = self.fuels_table
        if self.allow_unknown_fuels or table is None or "\n" not in table:
            return  # helper or built-in table: resolved inside ForeFire
        lines = [line for line in table.strip().splitlines() if line.strip()]
        header = [name.strip() for name in lines[0].split(";")]
        if "Index" not in header:
            raise ValueError("fuels_table must have an 'Index' column (ForeFire table format, ';'-separated)")
        column = header.index("Index")
        known = set()
        for line in lines[1:]:
            cells = line.split(";")
            if len(cells) > column:
                try:
                    known.add(int(float(cells[column])))
                except ValueError:
                    continue
        unknown = sorted(set(np.unique(fuel_values).astype(np.int64).tolist()) - known)
        if unknown:
            raise ValueError(
                f"fuel codes {unknown[:10]} are not in the Index column of fuels_table; ForeFire would give "
                "them all-zero fuel properties (pass allow_unknown_fuels=True to accept that)"
            )

    @staticmethod
    def _ignitions(
        ignition_mask: Optional[ArrayLike],
        ignition_points: Optional[Sequence[Sequence[float]]],
        width: float,
        height: float,
        origin: str,
    ) -> List[Tuple[float, float]]:
        ignitions: List[Tuple[float, float]] = []
        if ignition_mask is not None:
            mask = _as_array(ignition_mask, "ignition_mask", dtype=np.float64)
            if mask.ndim != 2:
                raise ValueError(f"ignition_mask must be a 2-D raster, got shape {mask.shape}")
            mrows, mcols = mask.shape
            for r, c in zip(*np.nonzero(mask > 0)):
                x = (c + 0.5) * width / mcols
                row_from_south = (mrows - 1 - r) if origin == "upper" else r
                y = (row_from_south + 0.5) * height / mrows
                ignitions.append((float(x), float(y)))
        if ignition_points is not None:
            for point in ignition_points:
                if len(point) != 2:
                    raise ValueError(f"ignition points must be (x, y) in metres, got {point!r}")
                x, y = float(point[0]), float(point[1])
                if not (0.0 < x < width and 0.0 < y < height):
                    raise ValueError(f"ignition point {point!r} is outside the domain (0, {width}) x (0, {height})")
                ignitions.append((x, y))
        if not ignitions:
            raise ValueError("give an ignition_mask with at least one True cell or ignition_points")
        return ignitions


__all__ = [
    "FOREFIRE_CHECKED_VERSION",
    "ForeFireNotInstalledError",
    "ForeFireResult",
    "ForeFireSimulator",
    "forefire_available",
    "forefire_version",
]
