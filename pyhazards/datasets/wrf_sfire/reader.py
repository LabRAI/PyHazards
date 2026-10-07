"""Read WRF-SFIRE fire-grid outputs (``wrfout`` history files) into PyHazards spread rasters.

WRF-SFIRE couples the WRF atmosphere model with the SFIRE level-set fire-spread model:

- Mandel, Beezley, Kochanski, "Coupled atmosphere-wildland fire modeling with WRF 3.3 and SFIRE
  2011", Geoscientific Model Development 4:591-610 (2011), doi:10.5194/gmd-4-591-2011.
- Mandel, Amram, Beezley, Kelman, Kochanski, Kondratenko, Lynn, Regev, Vejmelka, "Recent advances
  and applications of WRF-SFIRE", Natural Hazards and Earth System Sciences 14:2829-2845 (2014),
  doi:10.5194/nhess-14-2829-2014.
- Official code: https://github.com/openwfm/WRF-SFIRE (Fortran, WRF public-domain notice), release
  ``W4.4-S0.1``.

PyHazards does not run WRF-SFIRE (an MPI Fortran model built per machine) and reimplements none of
it. This module only reads the NetCDF files that a WRF-SFIRE run writes. The layout it assumes was
taken from the official sources, not guessed:

- Variable names, descriptions and units: ``Registry/registry.fire`` at ``W4.4-S0.1`` (package
  ``fire_sfire1``, ``ifire = 1``); every variable read here has ``h`` (history output) in its I/O
  column.
- Fire-grid ("subgrid") variables have dimensions ``(Time, south_north_subgrid, west_east_subgrid)``.
  WRF sizes the fire mesh from the staggered atmospheric dimensions, so it holds
  ``(south_north + 1) * sr_y`` by ``(west_east + 1) * sr_x`` points, and the last ``sr_y`` rows and
  ``sr_x`` columns are padding. ``sr_x``/``sr_y`` (namelist ``&domains``) are read from global
  attributes when present, otherwise from the dimension sizes; the same rules are used by the
  OpenWFM post-processing tools (wrfxpy, MIT: ``src/clamp2mesh.py:get_subgrid_coordinates`` and
  ``src/vis/var_wisdom.py:strip_end``).
- ``TIGN_G`` is the fire arrival ("ignition") time in seconds since the simulation start. In cells
  that are not burning SFIRE stores a time one fire step in the future (``prop_ls`` in
  ``phys/module_fr_sfire_core.F``), so ``TIGN_G`` is only an arrival time where the cell has burned.
- ``LFN`` is the level-set function; the fire region is ``LFN < 0``.
- ``XTIME`` is minutes since the simulation start; the fire-mesh spacing is ``DX / sr_x`` by
  ``DY / sr_y`` metres.

Checked on a real history file of the official ideal case ``test/em_fire/hill`` (WRF-SFIRE
``W4.4-S0.1`` built locally): its header is recorded in
``tests/fixtures/wrf_sfire_hill_wrfout_header.json``. In that file the subgrid is 420 x 420 for a
41 x 41 atmospheric grid with ``sr = 10`` (no ``sr_x``/``sr_y`` attributes), ``TIGN_G`` is the frame
time plus one fire step (0.25 s) wherever ``LFN >= 0``, and ``LFN < 0`` coincides with
``TIGN_G <= frame time`` in every frame.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

# name -> (description, units), from Registry/registry.fire (WRF-SFIRE W4.4-S0.1, package fire_sfire1).
WRF_SFIRE_FIRE_GRID_VARIABLES: Dict[str, Tuple[str, str]] = {
    "TIGN_G": ("ignition time on ground", "s"),
    "LFN": ("level function", "1"),
    "FIRE_AREA": ("fraction of cell area on fire", "1"),
    "FUEL_FRAC": ("fuel remaining", "1"),
    "FGRNHFX": ("heat flux from ground fire", "W/m^2"),
    "FGRNQFX": ("moisture flux from ground fire", "W/m^2"),
    "FCANHFX": ("heat flux from crown fire", "W/m^2"),
    "ROS": ("rate of spread", "m/s"),
    "F_ROS": ("max spread rate in any direction", "m/s"),
    "F_INT": ("fire reaction intensity for risk rating, without fire", "J/m^2/s"),
    "FLINEINT": ("fireline intensity", "W/m"),
    "NFUEL_CAT": ("fuel data", "-"),
    "ZSF": ("height of surface above sea level", "m"),
    "FMC_G": ("ground fuel moisture contents", "1"),
    "UF": ("fire wind", "m/s"),
    "VF": ("fire wind", "m/s"),
    "FXLONG": ("longitude of midpoints of fire cells", "degrees"),
    "FXLAT": ("latitude of midpoints of fire cells", "degrees"),
}

DEFAULT_VARIABLES: Tuple[str, ...] = ("TIGN_G", "LFN", "FIRE_AREA", "FGRNHFX", "ROS")
SUBGRID_DIMS = ("south_north_subgrid", "west_east_subgrid")
_ORIGINS = ("upper", "lower")

PathLike = Union[str, os.PathLike]


def _expand_paths(paths: Union[PathLike, Sequence[PathLike]]) -> List[Path]:
    items = [paths] if isinstance(paths, (str, os.PathLike)) else list(paths)
    found: List[Path] = []
    for item in items:
        text = os.fspath(item)
        pattern = any(char in text for char in "*?[")
        matches = sorted(glob.glob(os.path.expanduser(text))) if pattern else [os.path.expanduser(text)]
        if not matches:
            raise FileNotFoundError(f"no WRF-SFIRE output matches {text!r}")
        for match in matches:
            path = Path(match)
            if not path.is_file():
                raise FileNotFoundError(f"WRF-SFIRE output not found: {path}")
            found.append(path)
    if not found:
        raise ValueError("give at least one wrfout path")
    return found


def _time_strings(ds) -> List[str]:
    if "Times" not in ds.variables:
        return []
    raw = np.asarray(ds["Times"].values)
    if raw.dtype.kind == "S" and raw.ndim == 2:  # undecoded (Time, DateStrLen) characters
        raw = np.array([b"".join(row) for row in raw])
    return [item.decode() if isinstance(item, bytes) else str(item) for item in raw.reshape(-1)]


def _refinement(ds) -> Tuple[int, int]:
    attrs = ds.attrs
    if "sr_x" in attrs and "sr_y" in attrs and int(attrs["sr_x"]) > 0 and int(attrs["sr_y"]) > 0:
        return int(attrs["sr_x"]), int(attrs["sr_y"])
    for name in ("west_east", "south_north", *SUBGRID_DIMS):
        if name not in ds.sizes:
            raise ValueError(f"wrfout file has no {name!r} dimension and no sr_x/sr_y attributes")
    srx, rx = divmod(ds.sizes["west_east_subgrid"], ds.sizes["west_east"] + 1)
    sry, ry = divmod(ds.sizes["south_north_subgrid"], ds.sizes["south_north"] + 1)
    if rx or ry or srx < 1 or sry < 1:
        raise ValueError(
            "fire subgrid dimensions are not multiples of the staggered atmospheric dimensions: "
            f"{dict(ds.sizes)}"
        )
    return srx, sry


@dataclass
class WRFSFireFireGrid:
    """Fire-grid fields of one WRF-SFIRE run, padding removed.

    ``variables[name]`` has shape ``(time, rows, cols)``; with ``origin="upper"`` (default) row 0 is the
    northern edge, with ``origin="lower"`` it is the southern edge (WRF's own order). ``times`` are
    seconds since the simulation start.
    """

    times: np.ndarray
    variables: Dict[str, np.ndarray]
    fire_dx: float
    fire_dy: float
    sr_x: int
    sr_y: int
    origin: str = "upper"
    time_strings: List[str] = field(default_factory=list)
    source_files: List[str] = field(default_factory=list)

    @property
    def shape(self) -> Tuple[int, int]:
        first = next(iter(self.variables.values()))
        return int(first.shape[1]), int(first.shape[2])

    def burned_masks(self) -> np.ndarray:
        """``(time, rows, cols)`` boolean fire region: ``LFN < 0``, else ``TIGN_G <= frame time``."""
        if "LFN" in self.variables:
            return self.variables["LFN"] < 0
        if "TIGN_G" in self.variables:
            tign = self.variables["TIGN_G"]
            # Before the fire model has run (t = 0) the file can hold zeros; nothing has burned yet.
            return (tign <= self.times[:, None, None]) & (self.times[:, None, None] > 0)
        raise KeyError("burned masks need LFN or TIGN_G; read them with variables=(..., 'LFN', 'TIGN_G')")

    def arrival_time(self, frame: int = -1) -> np.ndarray:
        """``TIGN_G`` (s since simulation start) where the cell has burned by ``frame``, ``inf`` elsewhere."""
        if "TIGN_G" not in self.variables:
            raise KeyError("arrival_time needs TIGN_G; read it with variables=(..., 'TIGN_G')")
        burned = self.burned_masks()[frame]
        return np.where(burned, self.variables["TIGN_G"][frame].astype(np.float64), np.inf)

    def spread_pairs(
        self,
        horizon: int = 1,
        features: Sequence[str] = (),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Next-mask pairs for the ``wildfire.spread`` task.

        Returns ``inputs`` ``(N, 1 + len(features), rows, cols)`` (burned mask at frame ``t``, then the
        requested fields at ``t``), ``targets`` ``(N, 1, rows, cols)`` (burned mask at ``t + horizon``)
        and the input frame times ``(N,)``, all ``float32`` except the times.
        """
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1 output frames, got {horizon}")
        missing = [name for name in features if name not in self.variables]
        if missing:
            raise KeyError(f"features {missing} were not read; available: {sorted(self.variables)}")
        masks = self.burned_masks().astype(np.float32)
        steps = masks.shape[0] - horizon
        if steps < 1:
            raise ValueError(f"need more than {horizon} output frames to build spread pairs, got {masks.shape[0]}")
        channels = [masks[:steps, None]] + [self.variables[name][:steps, None].astype(np.float32) for name in features]
        inputs = np.concatenate(channels, axis=1)
        targets = masks[horizon:, None]
        return inputs, targets, self.times[:steps].copy()


def read_wrf_sfire_fire_grid(
    paths: Union[PathLike, Sequence[PathLike]],
    variables: Iterable[str] = DEFAULT_VARIABLES,
    origin: str = "upper",
    strip_padding: bool = True,
) -> WRFSFireFireGrid:
    """Read fire-grid variables from one or more ``wrfout`` files of one WRF-SFIRE domain.

    Args:
        paths: a file, a glob pattern (e.g. ``"run/wrfout_d01_*"``) or a list of them. Frames from all
            files are concatenated and sorted by ``XTIME``.
        variables: fire-grid variable names (see :data:`WRF_SFIRE_FIRE_GRID_VARIABLES`); any other
            variable on the fire subgrid is accepted too.
        origin: ``"upper"`` (row 0 = north, PyHazards / GeoTIFF order) or ``"lower"`` (WRF order).
        strip_padding: drop the ``sr_y`` padding rows and ``sr_x`` padding columns (default).
    """
    import xarray as xr

    if origin not in _ORIGINS:
        raise ValueError(f"origin must be 'upper' or 'lower', got {origin!r}")
    names = list(dict.fromkeys(variables))
    if not names:
        raise ValueError("give at least one fire-grid variable to read")
    files = _expand_paths(paths)

    times: List[np.ndarray] = []
    strings: List[str] = []
    fields: Dict[str, List[np.ndarray]] = {name: [] for name in names}
    geometry: Optional[Tuple[int, int, float, float, Tuple[int, int]]] = None
    for path in files:
        with xr.open_dataset(path, decode_times=False, mask_and_scale=False) as ds:
            if "XTIME" not in ds.variables:
                raise ValueError(f"{path} has no XTIME variable; is it a WRF history (wrfout) file?")
            srx, sry = _refinement(ds)
            for key in ("DX", "DY"):
                if key not in ds.attrs:
                    raise ValueError(f"{path} has no {key} global attribute")
            dx, dy = float(ds.attrs["DX"]) / srx, float(ds.attrs["DY"]) / sry
            sub_shape = (int(ds.sizes.get(SUBGRID_DIMS[0], 0)), int(ds.sizes.get(SUBGRID_DIMS[1], 0)))
            current = (srx, sry, dx, dy, sub_shape)
            if geometry is None:
                geometry = current
            elif geometry != current:
                raise ValueError(f"{path} has a different fire grid ({current}) than earlier files ({geometry})")
            for name in names:
                if name not in ds.variables:
                    available = sorted(v for v in ds.variables if ds[v].dims[-2:] == SUBGRID_DIMS)
                    raise KeyError(f"{name!r} is not in {path}; fire-grid variables in the file: {available}")
                var = ds[name]
                if var.dims != ("Time", *SUBGRID_DIMS):
                    raise ValueError(
                        f"{name} has dimensions {var.dims}, expected ('Time', {SUBGRID_DIMS[0]!r}, "
                        f"{SUBGRID_DIMS[1]!r}); only fire-grid variables are read here"
                    )
                values = np.asarray(var.values, dtype=np.float32)
                if strip_padding:
                    values = values[:, :-sry, :-srx]
                fields[name].append(values)
            times.append(np.asarray(ds["XTIME"].values, dtype=np.float64).reshape(-1) * 60.0)
            strings.extend(_time_strings(ds))

    assert geometry is not None
    seconds = np.concatenate(times)
    order = np.argsort(seconds, kind="stable")
    arrays = {name: np.concatenate(parts, axis=0)[order] for name, parts in fields.items()}
    if origin == "upper":
        arrays = {name: np.ascontiguousarray(values[:, ::-1, :]) for name, values in arrays.items()}
    srx, sry, dx, dy, _ = geometry
    return WRFSFireFireGrid(
        times=seconds[order],
        variables=arrays,
        fire_dx=dx,
        fire_dy=dy,
        sr_x=srx,
        sr_y=sry,
        origin=origin,
        time_strings=[strings[i] for i in order] if len(strings) == len(seconds) else [],
        source_files=[str(path) for path in files],
    )


__all__ = [
    "DEFAULT_VARIABLES",
    "WRF_SFIRE_FIRE_GRID_VARIABLES",
    "WRFSFireFireGrid",
    "read_wrf_sfire_fire_grid",
]
