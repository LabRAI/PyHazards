"""TempestExtremes-compatible cyclone detection (DetectNodes) and stitching (StitchNodes) in Python.

TCBench (Gomez et al. 2026, https://github.com/msgomez06/TCBench_Alpha) extracts the cyclone tracks
of Pangu-Weather, FourCastNet v2 and AIFS from their global forecast fields with TempestExtremes
(Ullrich and Zarzycki, GMD 10:1069-1090, 2017, doi:10.5194/gmd-10-1069-2017; Ullrich et al., GMD
14:5023-5048, 2021, doi:10.5194/gmd-14-5023-2021), using the commands of
``dev/TempestExtremes_example.sh``::

    DetectNodes --searchbymin msl --mergedist 6.0
        --closedcontourcmd "msl,200.0,5.5,0;_DIFF(z300,z500),-58.8,6.5,1.0"
        --outputcmd "msl,min,0;_VECMAG(u10,v10),max,2"
    StitchNodes --in_fmt "lon,lat,slp,wind10" --range 8.0 --mintime 12h
        --threshold "wind10,>=,10.0,2;lat,<=,50.0,1;lat,>=,-50.0,1"

This module implements the subset of DetectNodes and StitchNodes those commands use, on regular
latitude-longitude grids, following the published algorithm descriptions and the TempestExtremes
user guide. TempestExtremes itself (C++, BSD-2-Clause per its repository LICENSE; some source files
still carry an older GPL header) is not copied, linked or required: the official binaries are used
only as the test oracle. ``tests/oracle/test_tempest_oracle.py`` checks that on synthetic fields and
on TCBench's released Pangu-Weather forecast fields this module writes the same tracks, byte for
byte, as DetectNodes and StitchNodes v2.4.2 (and as TCBench's released ``unmatched_tracks`` file).

Algorithm (DetectNodes, per time step)
--------------------------------------
1. Candidates: grid points where the search variable is a local minimum (no 4-connected neighbour
   strictly smaller; 8-connected with ``diagonal_connectivity``). Longitude wraps around unless
   ``regional``; rows at the poles are not connected across the pole.
2. Merge: a candidate is dropped when another candidate within ``merge_dist`` degrees (great-circle,
   compared as chord length) has a strictly smaller value.
3. Closed contours, in order: starting from the candidate (or from the extremum of the field within
   ``minmax_dist`` degrees of it), a flood fill over connected grid points stops at points whose value
   has risen by ``delta`` (fallen, for negative ``delta``) relative to the start value; the criterion
   fails when the fill reaches any point farther than ``distance`` degrees from the start point.
4. Outputs: for each surviving node, the minimum or maximum of a field within a great-circle radius.

Algorithm (StitchNodes)
-----------------------
Each node is linked to its nearest node at the next time (``max_gap`` extra steps allowed) when that
node lies within ``range_deg``; paths are assembled first-come-first-served in time and grid order,
then filtered by duration (``min_time``) and threshold counts. Thresholds and output columns use the
node values as printed by DetectNodes (``%3.6e``), exactly like the C++ tools, which pass them through
a text file.

Numerics follow the C++ code paths that affect results: fields are single precision, differences of
field values are formed in single precision, great-circle distances use the spherical law of cosines
in double precision, and coordinates are converted with ``deg * (pi / 180)``.

Only operators used by common cyclone configurations are supported in variable expressions: a field
name, ``_DIFF(a,b)`` and ``_VECMAG(a,b)``. Threshold commands (``--thresholdcmd``), ``--searchbymax``
and unstructured grids raise ``NotImplementedError``.
"""

from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "ClosedContourCriterion",
    "DetectNodesConfig",
    "NodeOutput",
    "StitchNodesConfig",
    "StitchThreshold",
    "TCBENCH_DETECT_NODES",
    "TCBENCH_STITCH_NODES",
    "detect_nodes",
    "read_stitchnodes_csv",
    "stitch_nodes",
    "tempest_tracks",
    "write_stitchnodes_csv",
]

# --------------------------------------------------------------------------------------------------
# Configuration


@dataclass(frozen=True)
class ClosedContourCriterion:
    """``--closedcontourcmd var,delta,distance,minmaxdist`` (distances in great-circle degrees)."""

    variable: str
    delta: float
    distance: float
    minmax_distance: float = 0.0

    def __post_init__(self) -> None:
        if self.delta == 0.0:
            raise ValueError("closed contour delta must be non-zero")
        if self.distance <= 0.0:
            raise ValueError("closed contour distance must be positive")
        if self.minmax_distance < 0.0:
            raise ValueError("closed contour min/max distance must be non-negative")


@dataclass(frozen=True)
class NodeOutput:
    """``--outputcmd var,op,distance`` with ``op`` in ``min`` / ``max``."""

    variable: str
    op: str
    distance: float
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.op not in {"min", "max"}:
            raise NotImplementedError(f"output operation {self.op!r} is not supported (use 'min' or 'max')")
        if self.distance < 0.0:
            raise ValueError("output distance must be non-negative")


@dataclass(frozen=True)
class DetectNodesConfig:
    """Options of ``DetectNodes`` (``search_by_min`` is the variable searched for minima)."""

    search_by_min: str = "msl"
    closed_contours: Tuple[ClosedContourCriterion, ...] = ()
    no_closed_contours: Tuple[ClosedContourCriterion, ...] = ()
    merge_dist: float = 0.0
    outputs: Tuple[NodeOutput, ...] = ()
    min_lat: float = 0.0
    max_lat: float = 0.0
    min_abs_lat: float = 0.0
    regional: bool = False
    diagonal_connectivity: bool = False


@dataclass(frozen=True)
class StitchThreshold:
    """``--threshold column,op,value,count``; ``count`` is an integer or ``all`` / ``first`` / ``last``."""

    column: str
    op: str
    value: float
    count: Union[int, str] = 1

    _OPS = (">", "<", ">=", "<=", "=", "!=", "|>=", "|<=")

    def __post_init__(self) -> None:
        if self.op not in self._OPS:
            raise ValueError(f"threshold operation {self.op!r} not in {self._OPS}")
        if isinstance(self.count, str) and self.count not in {"all", "first", "last"}:
            raise ValueError("threshold count must be an integer or 'all', 'first', 'last'")

    def satisfied(self, value: float) -> bool:
        op, ref = self.op, self.value
        if op == ">":
            return value > ref
        if op == "<":
            return value < ref
        if op == ">=":
            return value >= ref
        if op == "<=":
            return value <= ref
        if op == "=":
            return value == ref
        if op == "!=":
            return value != ref
        if op == "|>=":
            return abs(value) >= ref
        return abs(value) <= ref


@dataclass(frozen=True)
class StitchNodesConfig:
    """Options of ``StitchNodes``.

    ``min_time`` is a number of time steps (int) or a duration in hours (``"12h"`` / float hours);
    ``max_gap`` is the number of missing time steps allowed inside a path.
    """

    columns: Tuple[str, ...] = ("lon", "lat")
    range_deg: float = 5.0
    min_time: Union[int, str] = 3
    max_gap: int = 0
    thresholds: Tuple[StitchThreshold, ...] = ()
    min_endpoint_dist: float = 0.0
    min_path_dist: float = 0.0

    def __post_init__(self) -> None:
        if "lat" not in self.columns or "lon" not in self.columns:
            raise ValueError("columns must contain 'lat' and 'lon'")
        if int(self.max_gap) < 0:
            raise ValueError("max_gap must be non-negative")
        for threshold in self.thresholds:
            if threshold.column not in self.columns:
                raise ValueError(f"threshold column {threshold.column!r} not in columns {self.columns}")


TCBENCH_DETECT_NODES = DetectNodesConfig(
    search_by_min="msl",
    closed_contours=(
        ClosedContourCriterion("msl", 200.0, 5.5, 0.0),
        ClosedContourCriterion("_DIFF(z300,z500)", -58.8, 6.5, 1.0),
    ),
    merge_dist=6.0,
    outputs=(NodeOutput("msl", "min", 0.0, "slp"), NodeOutput("_VECMAG(u10,v10)", "max", 2.0, "wind10")),
)
"""DetectNodes options of TCBench's ``dev/TempestExtremes_example.sh`` (msl in Pa, z in m2 s-2)."""

TCBENCH_STITCH_NODES = StitchNodesConfig(
    columns=("lon", "lat", "slp", "wind10"),
    range_deg=8.0,
    min_time="12h",
    max_gap=0,
    thresholds=(
        StitchThreshold("wind10", ">=", 10.0, 2),
        StitchThreshold("lat", "<=", 50.0, 1),
        StitchThreshold("lat", ">=", -50.0, 1),
    ),
)
"""StitchNodes options of TCBench's ``dev/TempestExtremes_example.sh``."""


# --------------------------------------------------------------------------------------------------
# Grid and fields


class _Grid:
    """Regular lat-lon grid in TempestExtremes' flattened (lat-major) order."""

    def __init__(self, lat: np.ndarray, lon: np.ndarray, regional: bool, diagonal: bool):
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        if lat.ndim != 1 or lon.ndim != 1 or lat.size < 2 or lon.size < 2:
            raise ValueError("lat and lon must be 1-D coordinate vectors with at least two values")
        if not regional and np.abs(lat).max() > 90.0 + 1e-10:
            raise ValueError("latitudes must lie in [-90, 90] (use regional=True otherwise)")
        self.nlat, self.nlon = lat.size, lon.size
        self.regional = regional
        self.diagonal = diagonal
        # TempestExtremes: vecLat[j] *= M_PI / 180.0
        self.lat_rad = lat * (math.pi / 180.0)
        self.lon_rad = lon * (math.pi / 180.0)
        self._lat = [float(v) for v in self.lat_rad]
        self._lon = [float(v) for v in self.lon_rad]
        self._sin_lat = [math.sin(v) for v in self._lat]
        self._cos_lat = [math.cos(v) for v in self._lat]

    def neighbours(self, ix: int) -> List[int]:
        nlat, nlon = self.nlat, self.nlon
        j, i = divmod(ix, nlon)
        out: List[int] = []
        if self.diagonal:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    inew, jnew = i + di, j + dj
                    if jnew < 0 or jnew >= nlat:
                        continue
                    if self.regional:
                        if inew < 0 or inew >= nlon:
                            continue
                    else:
                        inew %= nlon
                    out.append(jnew * nlon + inew)
            return out
        if j != 0:
            out.append((j - 1) * nlon + i)
        if j != nlat - 1:
            out.append((j + 1) * nlon + i)
        if (not self.regional) or (i != 0 and i != nlon - 1):
            out.append(j * nlon + (i + 1) % nlon)
            out.append(j * nlon + (i + nlon - 1) % nlon)
        return out

    def latlon(self, ix: int) -> Tuple[float, float]:
        j, i = divmod(ix, self.nlon)
        return self._lat[j], self._lon[i]

    def distance_deg(self, ix1: int, ix2: int) -> float:
        """GreatCircleDistance_Deg(point 1, point 2) (law of cosines, ``RadToDeg``), same operation order."""
        j1, i1 = divmod(ix1, self.nlon)
        j2, i2 = divmod(ix2, self.nlon)
        r = self._sin_lat[j1] * self._sin_lat[j2] + self._cos_lat[j1] * self._cos_lat[j2] * math.cos(self._lon[i2] - self._lon[i1])
        if r >= 1.0:
            r = 0.0
        elif r <= -1.0:
            r = math.pi
        else:
            r = math.acos(r)
        return r * 180.0 / math.pi

    def contour_distance_deg(self, ix0: int, ix: int) -> float:
        """Distance as written in HasClosedContour (``180 / pi * acos``; 180 when antipodal)."""
        j0, i0 = divmod(ix0, self.nlon)
        j, i = divmod(ix, self.nlon)
        r = math.sin(self._lat[j0]) * math.sin(self._lat[j]) + math.cos(self._lat[j0]) * math.cos(self._lat[j]) * math.cos(self._lon[i] - self._lon[i0])
        if r >= 1.0:
            return 0.0
        if r <= -1.0:
            return 180.0
        return 180.0 / math.pi * math.acos(r)


_EXPR = re.compile(r"^\s*(_[A-Z]+)\((.*)\)\s*$")


def _evaluate(expression: str, fields: Mapping[str, np.ndarray], cache: Dict[str, np.ndarray]) -> np.ndarray:
    """Single-precision field for a variable name or a supported TempestExtremes operator."""
    expression = expression.strip()
    if expression in cache:
        return cache[expression]
    match = _EXPR.match(expression)
    if match is None:
        if expression not in fields:
            raise KeyError(f"field {expression!r} not provided (have {sorted(fields)})")
        value = np.asarray(fields[expression], dtype=np.float32)
    else:
        op, inner = match.groups()
        if op not in {"_DIFF", "_VECMAG"}:
            raise NotImplementedError(f"operator {op} is not supported (only _DIFF and _VECMAG)")
        args = _split_args(inner)
        if len(args) != 2:
            raise ValueError(f"{op} expects two arguments, got {args}")
        left, right = (_evaluate(arg, fields, cache) for arg in args)
        if op == "_DIFF":
            value = (left - right).astype(np.float32)
        else:
            value = np.sqrt(left * left + right * right).astype(np.float32)
    cache[expression] = value
    return value


def _split_args(inner: str) -> List[str]:
    args, depth, start = [], 0, 0
    for k, ch in enumerate(inner):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(inner[start:k])
            start = k + 1
    args.append(inner[start:])
    return [arg.strip() for arg in args]


# --------------------------------------------------------------------------------------------------
# DetectNodes


def _local_minima(grid: _Grid, data: np.ndarray) -> np.ndarray:
    """Flat indices of points with no connected neighbour strictly smaller (NaN = fill, ignored)."""
    values = data.reshape(grid.nlat, grid.nlon)
    valid = ~np.isnan(values)
    is_min = valid.copy()

    def shifted(dj: int, di: int):
        out = np.full_like(values, np.nan)
        rows_src = slice(max(0, dj), grid.nlat + min(0, dj))
        rows_dst = slice(max(0, -dj), grid.nlat + min(0, -dj))
        moved = np.roll(values, -di, axis=1) if di else values
        out[rows_dst] = moved[rows_src]
        if grid.regional and di:
            if di > 0:
                out[:, grid.nlon - di :] = np.nan
            else:
                out[:, : -di] = np.nan
        return out

    if grid.diagonal:
        offsets = [(dj, di) for dj in (-1, 0, 1) for di in (-1, 0, 1) if (dj, di) != (0, 0)]
    else:
        offsets = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    for dj, di in offsets:
        neighbour = shifted(dj, di)
        if grid.regional and not grid.diagonal and di:
            # 4-connected regional grids drop both zonal links at the first and last column.
            neighbour[:, 0] = np.nan
            neighbour[:, -1] = np.nan
        with np.errstate(invalid="ignore"):
            is_min &= ~(neighbour < values)
    return np.flatnonzero(is_min)


def _xyz(lat: float, lon: float) -> Tuple[float, float, float]:
    return math.cos(lon) * math.cos(lat), math.sin(lon) * math.cos(lat), math.sin(lat)


def _merge(grid: _Grid, data: np.ndarray, candidates: Sequence[int], merge_dist: float) -> List[int]:
    if not candidates:
        return []
    chord = 2.0 * math.sin(0.5 * merge_dist / 180.0 * math.pi)
    chord_sq = chord * chord
    points = np.array([_xyz(*grid.latlon(ix)) for ix in candidates])
    values = data[np.asarray(candidates)]
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    neighbours = tree.query_ball_point(points, r=chord * (1.0 + 1e-9) + 1e-15)
    kept = []
    for k, ix in enumerate(candidates):
        x0, y0, z0 = points[k]
        value = float(values[k])
        extremum = True
        for m in neighbours[k]:
            x, y, z = points[m]
            dist_sq = 0.0
            dist_sq += (x - x0) * (x - x0)
            dist_sq += (y - y0) * (y - y0)
            dist_sq += (z - z0) * (z - z0)
            if dist_sq <= chord_sq and float(values[m]) < value:
                extremum = False
                break
        if extremum:
            kept.append(ix)
    return kept


def _find_local_minmax(grid: _Grid, data: np.ndarray, minimum: bool, ix0: int, max_dist: float) -> Tuple[int, float]:
    """FindLocalMinMax: extremum (first found on ties) over the connected points within ``max_dist``."""
    best_ix = ix0
    best = data[ix0]
    best_is_fill = bool(np.isnan(best))
    seen = set()
    queue = deque([ix0])
    while queue:
        ix = queue.popleft()
        if ix in seen:
            continue
        seen.add(ix)
        if grid.distance_deg(ix, ix0) > max_dist:
            continue
        value = data[ix]
        if not np.isnan(value):
            if best_is_fill or (value < best if minimum else value > best):
                best_ix, best, best_is_fill = ix, value, False
        queue.extend(grid.neighbours(ix))
    return best_ix, float(best)


def _has_closed_contour(grid: _Grid, data: np.ndarray, ix0: int, criterion: ClosedContourCriterion) -> bool:
    delta = float(criterion.delta)
    origin = ix0
    if criterion.minmax_distance != 0.0:
        origin, _ = _find_local_minmax(grid, data, delta > 0.0, ix0, criterion.minmax_distance)
    ref = data[origin]
    seen = set()
    queue = deque([origin])
    while queue:
        ix = queue.popleft()
        if ix in seen:
            continue
        seen.add(ix)
        if grid.contour_distance_deg(origin, ix) > criterion.distance:
            return False
        value = data[ix]
        # Single-precision difference compared with the double-precision delta, as in the C++ code.
        if delta > 0.0:
            if float(value - ref) >= delta:
                continue
        elif float(ref - value) >= -delta:
            continue
        queue.extend(grid.neighbours(ix))
    return True


def _output_value(grid: _Grid, data: np.ndarray, ix: int, output: NodeOutput) -> float:
    _, value = _find_local_minmax(grid, data, output.op == "min", ix, output.distance)
    return value


def _format_e(value: float) -> str:
    return "%3.6e" % value


def _format_f(value: float) -> str:
    return "%3.6f" % value


def detect_nodes(
    fields: Mapping[str, np.ndarray],
    lat: Sequence[float],
    lon: Sequence[float],
    times: Sequence,
    config: DetectNodesConfig = TCBENCH_DETECT_NODES,
):
    """Run DetectNodes on fields shaped ``(time, lat, lon)``.

    ``fields`` maps variable names to arrays (cast to float32 like TempestExtremes). NaN marks missing
    values: such points are never candidates, but inside closed-contour tests they never stop the
    flood fill (TempestExtremes compares its numeric fill value instead), so fill gaps before tracking
    fields that have them. Returns a DataFrame with one row per node: ``time``, ``i`` (longitude index), ``j``
    (latitude index), ``lon``, ``lat`` (degrees, as printed by DetectNodes), one float column per output
    (named ``output.name`` or ``out<k>``) and ``text_<name>`` columns holding the printed strings.
    """
    import pandas as pd

    grid = _Grid(np.asarray(lat), np.asarray(lon), config.regional, config.diagonal_connectivity)
    times = pd.to_datetime(list(times))
    arrays = {name: np.asarray(value) for name, value in fields.items()}
    for name, value in arrays.items():
        if value.ndim != 3 or value.shape[1:] != (grid.nlat, grid.nlon) or value.shape[0] != len(times):
            raise ValueError(
                f"field {name!r} must be shaped (time={len(times)}, lat={grid.nlat}, lon={grid.nlon}), got {value.shape}"
            )
    if config.min_lat > config.max_lat:
        raise ValueError("min_lat must not exceed max_lat")
    names = [out.name or f"out{k}" for k, out in enumerate(config.outputs)]
    rows: List[Dict[str, object]] = []
    for t, time in enumerate(times):
        step = {name: value[t] for name, value in arrays.items()}
        cache: Dict[str, np.ndarray] = {}
        search = _evaluate(config.search_by_min, step, cache).reshape(-1)
        candidates = [int(ix) for ix in _local_minima(grid, search)]
        if config.min_lat != config.max_lat or config.min_abs_lat != 0.0:
            kept = []
            # TempestExtremes: dMinLatitude *= M_PI / 180.0
            lo, hi, absmin = (value * (math.pi / 180.0) for value in (config.min_lat, config.max_lat, config.min_abs_lat))
            for ix in candidates:
                la, _ = grid.latlon(ix)
                if config.min_lat != config.max_lat and (la < lo or la > hi):
                    continue
                if config.min_abs_lat != 0.0 and abs(la) < absmin:
                    continue
                kept.append(ix)
            candidates = kept
        if config.merge_dist != 0.0:
            candidates = _merge(grid, search, candidates, config.merge_dist)
        for criterion in config.closed_contours:
            data = _evaluate(criterion.variable, step, cache).reshape(-1)
            candidates = [ix for ix in candidates if _has_closed_contour(grid, data, ix, criterion)]
        for criterion in config.no_closed_contours:
            data = _evaluate(criterion.variable, step, cache).reshape(-1)
            candidates = [ix for ix in candidates if not _has_closed_contour(grid, data, ix, criterion)]
        output_data = [_evaluate(out.variable, step, cache).reshape(-1) for out in config.outputs]
        for ix in sorted(candidates):
            j, i = divmod(ix, grid.nlon)
            row: Dict[str, object] = {
                "time": time,
                "i": i,
                "j": j,
                "text_lon": _format_f(grid._lon[i] * 180.0 / math.pi),
                "text_lat": _format_f(grid._lat[j] * 180.0 / math.pi),
            }
            for name, out, data in zip(names, config.outputs, output_data):
                row["text_" + name] = _format_e(_output_value(grid, data, ix, out))
            rows.append(row)
    columns = ["time", "i", "j", "text_lon", "text_lat"] + ["text_" + name for name in names]
    frame = pd.DataFrame(rows, columns=columns)
    frame["lon"] = frame["text_lon"].astype(float)
    frame["lat"] = frame["text_lat"].astype(float)
    for name in names:
        frame[name] = frame["text_" + name].astype(float)
    return frame



# --------------------------------------------------------------------------------------------------
# StitchNodes


def _min_time_seconds(min_time: Union[int, str, float]) -> Tuple[int, float]:
    """(minimum number of steps, minimum duration in seconds)."""
    if isinstance(min_time, (int, np.integer)) and not isinstance(min_time, bool):
        if int(min_time) < 1:
            raise ValueError("min_time as a step count must be positive")
        return int(min_time), 0.0
    if isinstance(min_time, float):
        hours = min_time
    else:
        text = str(min_time).strip()
        if text.isdigit():
            return _min_time_seconds(int(text))
        match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([hd])", text)
        if match is None:
            raise ValueError(f"min_time must be a step count, '<n>h' or '<n>d', got {min_time!r}")
        hours = float(match.group(1)) * (24.0 if match.group(2) == "d" else 1.0)
    if hours <= 0:
        raise ValueError("min_time must be positive")
    return 1, hours * 3600.0


def stitch_nodes(nodes, config: StitchNodesConfig = TCBENCH_STITCH_NODES):
    """Run StitchNodes on the output of :func:`detect_nodes` (or a table with the same columns).

    Returns a DataFrame with one row per track point: ``track_id`` (0-based, in path order),
    ``time``, ``i``, ``j`` and the configured columns as floats, plus ``text_<column>`` strings.
    The time steps are the distinct node times together with ``nodes.attrs['times']`` (set by
    :func:`tempest_tracks`, so that a time without any node still counts as a step, as in a
    DetectNodes file). Nodes of one time are taken in table order (DetectNodes: ascending grid index).
    """
    import pandas as pd

    min_steps, min_seconds = _min_time_seconds(config.min_time)
    times = sorted(set(pd.to_datetime(nodes["time"]).tolist()) | set(pd.to_datetime(nodes.attrs.get("times", [])).tolist()))
    if not times:
        raise ValueError("no candidate nodes to stitch")
    text_cols = []
    for column in config.columns:
        name = "text_" + column
        if name not in nodes.columns:
            if column not in nodes.columns:
                raise KeyError(f"nodes have no column {column!r}")
            nodes = nodes.assign(**{name: nodes[column].map(_format_e if column not in {"lon", "lat"} else _format_f)})
        text_cols.append(name)
    by_time: List[List[Dict[str, object]]] = []
    stamps = pd.to_datetime(nodes["time"])
    for time in times:
        rows = nodes[stamps == time]
        # Candidates keep the DetectNodes order (ascending flat grid index).
        by_time.append(rows.to_dict("records"))
    lon_col = text_cols[config.columns.index("lon")]
    lat_col = text_cols[config.columns.index("lat")]
    geo: List[List[Tuple[float, float, float, float, float]]] = []
    for rows in by_time:
        level = []
        for row in rows:
            lon_r = float(row[lon_col]) * math.pi / 180.0
            lat_r = float(row[lat_col]) * math.pi / 180.0
            level.append((lon_r, lat_r) + _xyz(lat_r, lon_r))
        geo.append(level)

    def gc_deg(a, b) -> float:
        r = math.sin(a[1]) * math.sin(b[1]) + math.cos(a[1]) * math.cos(b[1]) * math.cos(b[0] - a[0])
        if r >= 1.0:
            r = 0.0
        elif r <= -1.0:
            r = math.pi
        else:
            r = math.acos(r)
        return r * 180.0 / math.pi

    n_times = len(times)
    segments: List[Dict[int, Tuple[int, int]]] = [dict() for _ in range(max(n_times - 1, 0))]
    for t in range(n_times - 1):
        for i, node in enumerate(geo[t]):
            for g in range(1, int(config.max_gap) + 2):
                if t + g >= n_times:
                    break
                if not geo[t + g]:
                    continue
                best, best_d = None, None
                for k, other in enumerate(geo[t + g]):
                    d = (other[2] - node[2]) ** 2 + (other[3] - node[3]) ** 2 + (other[4] - node[4]) ** 2
                    if best_d is None or d < best_d:
                        best, best_d = k, d
                if gc_deg(node, geo[t + g][best]) <= config.range_deg:
                    segments[t][i] = (t + g, best)
                    break

    paths: List[List[Tuple[int, int]]] = []
    for t in range(n_times - 1):
        while segments[t]:
            start = min(segments[t])
            path = [(t, start)]
            tx, cand = t, start
            while True:
                nxt = segments[tx].pop(cand)
                path.append(nxt)
                tnext, cnext = nxt
                if tnext >= n_times - 1 or cnext not in segments[tnext]:
                    break
                tx, cand = tnext, cnext
            paths.append(path)

    kept_paths = []
    for path in paths:
        if len(path) < min_steps:
            continue
        if min_seconds > 0.0:
            duration = (times[path[-1][0]] - times[path[0][0]]).total_seconds()
            if duration < min_seconds:
                continue
        if config.min_endpoint_dist > 0.0:
            if gc_deg(geo[path[0][0]][path[0][1]], geo[path[-1][0]][path[-1][1]]) < config.min_endpoint_dist:
                continue
        if config.min_path_dist > 0.0:
            total = sum(gc_deg(geo[a[0]][a[1]], geo[b[0]][b[1]]) for a, b in zip(path[:-1], path[1:]))
            if total < config.min_path_dist:
                continue
        ok = True
        for threshold in config.thresholds:
            col = text_cols[config.columns.index(threshold.column)]
            points = path
            if threshold.count == "first":
                points = path[:1]
            elif threshold.count == "last":
                points = path[-1:]
            count = sum(threshold.satisfied(float(by_time[t][c][col])) for t, c in points)
            if threshold.count == "all":
                ok = count == len(path)
            elif threshold.count in {"first", "last"}:
                ok = count == 1
            else:
                ok = count >= int(threshold.count)
            if not ok:
                break
        if ok:
            kept_paths.append(path)

    rows = []
    for track_id, path in enumerate(kept_paths):
        for t, c in path:
            node = by_time[t][c]
            row = {"track_id": track_id, "time": times[t], "i": int(node["i"]), "j": int(node["j"])}
            for column, text in zip(config.columns, text_cols):
                row["text_" + column] = node[text]
                row[column] = float(node[text])
            rows.append(row)
    columns = ["track_id", "time", "i", "j"] + list(config.columns) + ["text_" + c for c in config.columns]
    return pd.DataFrame(rows, columns=columns)


def tempest_tracks(
    fields: Mapping[str, np.ndarray],
    lat: Sequence[float],
    lon: Sequence[float],
    times: Sequence,
    detect: DetectNodesConfig = TCBENCH_DETECT_NODES,
    stitch: StitchNodesConfig = TCBENCH_STITCH_NODES,
):
    """DetectNodes followed by StitchNodes (defaults: the TCBench configuration)."""
    nodes = detect_nodes(fields, lat, lon, times, detect)
    nodes.attrs["times"] = list(times)
    return stitch_nodes(nodes, stitch)


def write_stitchnodes_csv(tracks, path) -> None:
    """Write tracks in StitchNodes' ``--out_file_format csv`` layout (the TCBench ``unmatched_tracks`` files)."""
    import pandas as pd

    columns = [c[len("text_"):] for c in tracks.columns if c.startswith("text_")]
    lines = ["track_id, year, month, day, hour, i, j" + "".join(f", {c}" for c in columns)]
    for row in tracks.itertuples(index=False):
        record = row._asdict()
        stamp = pd.Timestamp(record["time"])
        seconds = stamp.hour * 3600 + stamp.minute * 60 + stamp.second
        values = [str(record["track_id"]), str(stamp.year), str(stamp.month), str(stamp.day), str(seconds // 3600), str(record["i"]), str(record["j"])]
        values += [str(record["text_" + c]) for c in columns]
        lines.append(", ".join(values))
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def read_stitchnodes_csv(path, columns: Optional[Iterable[str]] = None):
    """Read a StitchNodes CSV (e.g. a TCBench ``unmatched_tracks`` file) into the :func:`stitch_nodes` layout."""
    import pandas as pd

    frame = pd.read_csv(path, skipinitialspace=True, dtype=str)
    frame.columns = [c.strip() for c in frame.columns]
    required = ["track_id", "year", "month", "day", "hour", "i", "j", "lon", "lat"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} is not a StitchNodes CSV: missing columns {missing}")
    value_cols = [c for c in frame.columns if c not in {"track_id", "year", "month", "day", "hour", "i", "j"}]
    if columns is not None:
        value_cols = [c for c in value_cols if c in set(columns) | {"lon", "lat"}]
    out = pd.DataFrame(
        {
            "track_id": frame["track_id"].astype(int),
            "time": pd.to_datetime(
                {"year": frame["year"].astype(int), "month": frame["month"].astype(int), "day": frame["day"].astype(int), "hour": frame["hour"].astype(int)}
            ),
            "i": frame["i"].astype(int),
            "j": frame["j"].astype(int),
        }
    )
    for column in value_cols:
        out[column] = frame[column].astype(float)
    for column in value_cols:
        out["text_" + column] = frame[column].str.strip()
    return out
