"""Reader for seismic waveform benchmark datasets in the SeisBench format and in STEAD's own layout.

PyHazards reads these files itself (``h5py`` and the ``csv`` module); SeisBench (GPL-3.0) is neither
imported nor required. The datasets are not redistributed: download them from their providers and
point ``path`` at the directory.

SeisBench format (``layout="seisbench"``), as documented in SeisBench's "The SeisBench Data Format":
``metadata.csv`` + ``waveforms.hdf5`` in one directory, or chunks ``metadata<chunk>.csv`` +
``waveforms<chunk>.hdf5`` (listed in a ``chunks`` file when present). The HDF5 file has a ``data`` group
with one array per trace at ``data/<trace_name>`` and a ``data_format`` group with
``dimension_order`` (e.g. ``"CW"``), ``component_order`` (e.g. ``"ZNE"``) and optionally
``sampling_rate``. Trace names of the form ``<block>$<slice>`` (e.g. ``bucket3$12,:3,:6000``) address a
trace inside a block array ``data/<block>``. The metadata column ``split`` (train/dev/test) defines the
splits; sampling rates come from ``trace_sampling_rate_hz`` / ``trace_dt_s`` or ``data_format``.

STEAD layout (``layout="stead"``): the files distributed by the STEAD authors
(https://github.com/smousavi05/STEAD; ``merged.csv`` / ``merged.hdf5``, the six ``chunk<k>`` pairs, or
any ``<name>.csv`` + ``<name>.hdf5`` pair such as EQTransformer's ``100samples``). Each trace is
``data/<trace_name>``, shape ``(6000, 3)``, components E, N, Z, 100 Hz; P and S arrivals are the
``p_arrival_sample`` / ``s_arrival_sample`` columns. STEAD has no split column: ``test_trace_names``
(e.g. EQTransformer's ``ModelsAndSampleData/test.npy``) selects the test traces and the others go to
``train``; without it every trace is in ``test``.

Known datasets (``preset``): ``"stead"`` (Mousavi et al. 2019, CC BY 4.0) and ``"instance"`` (Michelini
et al. 2021, INGV, CC BY 4.0; 120-s traces at 100 Hz stored as ``(3, 12000)`` in E, N, Z order per the
INGV notebooks). A preset only fills in values that the files do not provide.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ..base import DataBundle, DataSplit, Dataset, FeatureSpec, LabelSpec

# Arrival-sample columns, in order of preference, used when ``phase_columns`` is not given.
DEFAULT_PHASE_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "P": ("trace_p_arrival_sample", "trace_P_arrival_sample", "trace_P1_arrival_sample",
          "trace_Pg_arrival_sample", "trace_Pn_arrival_sample", "p_arrival_sample"),
    "S": ("trace_s_arrival_sample", "trace_S_arrival_sample", "trace_S1_arrival_sample",
          "trace_Sg_arrival_sample", "trace_Sn_arrival_sample", "s_arrival_sample"),
}

PRESETS: Dict[str, Dict[str, object]] = {
    "stead": {"layout": "stead", "component_order": "ENZ", "dimension_order": "WC", "sampling_rate": 100.0},
    "instance": {"layout": "seisbench", "component_order": "ENZ", "dimension_order": "CW", "sampling_rate": 100.0},
}

_SPLIT_NAMES = {"train": "train", "dev": "val", "val": "val", "validation": "val", "test": "test"}
_SLICE_ITEM = re.compile(r"^\s*(-?\d*)\s*(?::\s*(-?\d*)\s*(?::\s*(-?\d*)\s*)?)?$")


def parse_trace_name(trace_name: str) -> Tuple[str, Optional[Tuple[Union[int, slice], ...]]]:
    """``"block$1,:3,:6000"`` -> ``("block", (1, slice(None, 3), slice(None, 6000)))``."""
    if "$" not in trace_name:
        return trace_name, None
    block, _, location = trace_name.partition("$")
    items: List[Union[int, slice]] = []
    for item in location.split(","):
        match = _SLICE_ITEM.match(item)
        if match is None or item.strip() == "":
            raise ValueError(f"Cannot parse the trace location {location!r} of {trace_name!r}.")
        start, stop, step = match.groups()
        if ":" not in item:
            items.append(int(start))
        else:
            to_int = lambda value: int(value) if value not in (None, "") else None  # noqa: E731
            items.append(slice(to_int(start), to_int(stop), to_int(step)))
    return block, tuple(items)


def _decode(value) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind in ("S", "O"):
        return "".join(_decode(v) for v in value.ravel())
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_data_format(handle) -> Dict[str, object]:
    """The ``data_format`` group of a SeisBench waveform file (empty when absent)."""
    if "data_format" not in handle:
        return {}
    group = handle["data_format"]
    return {key: _decode(group[key][()]) for key in group.keys()}


def _float(value: Optional[str]) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if text in ("", "nan", "NaN", "None", "none"):
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _seisbench_chunks(root: Path) -> List[Tuple[Path, Path]]:
    if (root / "chunks").is_file():
        names = [line.strip() for line in (root / "chunks").read_text(encoding="utf-8").splitlines() if line.strip()]
    elif (root / "metadata.csv").is_file():
        names = [""]
    else:
        names = sorted(path.stem[len("metadata"):] for path in root.glob("metadata*.csv"))
    pairs = [(root / f"metadata{name}.csv", root / f"waveforms{name}.hdf5") for name in names]
    if not pairs:
        raise FileNotFoundError(f"No SeisBench metadata*.csv / waveforms*.hdf5 files in {root}.")
    for metadata, waveforms in pairs:
        if not metadata.is_file() or not waveforms.is_file():
            raise FileNotFoundError(f"Incomplete SeisBench chunk in {root}: need {metadata.name} and {waveforms.name}.")
    return pairs


def _stead_pairs(root: Path) -> List[Tuple[Path, Path]]:
    pairs = [(csv_path, csv_path.with_suffix(".hdf5")) for csv_path in sorted(root.glob("*.csv"))]
    pairs = [pair for pair in pairs if pair[1].is_file()]
    if not pairs:
        raise FileNotFoundError(f"No STEAD <name>.csv + <name>.hdf5 pair in {root}.")
    return pairs


class SeisBenchWaveformDataset(Dataset):
    """Labelled three-component waveforms from a SeisBench-format (or STEAD-layout) dataset.

    Inputs are ``(n, 3, window)`` float32 waveforms in the dataset's component order (reported in
    ``metadata["component_order"]``; the picking benchmark reorders channels for each model), targets
    ``(n, 2)`` P and S arrival samples relative to the window start (NaN when absent or outside the
    window). Splits ``train``/``val``/``test`` follow the dataset's ``split`` column (``dev`` -> ``val``).

    ``window_samples`` / ``window_start`` cut a fixed window from every trace (zero-padded when a
    trace is shorter); without ``window_samples`` all traces must have the same length.
    ``max_traces`` keeps the first rows of each split (``micro=True`` keeps 8), ``trace_category``
    keeps rows with that ``trace_category`` value (e.g. ``"earthquake_local"`` or ``"noise"`` in STEAD).
    """

    name = "seisbench_waveforms"

    def __init__(
        self,
        path: Union[str, Path, None] = None,
        cache_dir: Optional[str] = None,
        layout: Optional[str] = None,
        preset: Optional[str] = None,
        component_order: Optional[str] = None,
        dimension_order: Optional[str] = None,
        sampling_rate: Optional[float] = None,
        window_samples: Optional[int] = None,
        window_start: int = 0,
        phase_columns: Optional[Dict[str, Sequence[str]]] = None,
        test_trace_names: Union[str, Path, Sequence[str], None] = None,
        trace_category: Optional[str] = None,
        max_traces: Optional[int] = None,
        micro: bool = False,
    ):
        super().__init__(cache_dir=cache_dir)
        if path is None:
            raise ValueError(
                "seisbench_waveforms reads a downloaded dataset: pass path=<directory with metadata.csv and "
                "waveforms.hdf5 (SeisBench format) or <name>.csv and <name>.hdf5 (STEAD layout)>. "
                "For synthetic smoke data use 'earthquake_waveforms_synthetic'."
            )
        self.path = Path(path).expanduser()
        if not self.path.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.path}")
        if preset is not None and preset not in PRESETS:
            raise ValueError(f"Unknown preset {preset!r}; expected one of {sorted(PRESETS)}.")
        defaults = dict(PRESETS[preset]) if preset is not None else {}
        self.preset = preset
        self.layout = (layout or defaults.get("layout") or "seisbench").lower()
        if self.layout not in ("seisbench", "stead"):
            raise ValueError(f"layout must be 'seisbench' or 'stead', got {self.layout!r}.")
        if self.layout == "stead":
            defaults = {**PRESETS["stead"], **defaults}
        self.defaults = defaults
        self.component_order = component_order
        self.dimension_order = dimension_order
        self.sampling_rate = sampling_rate
        self.window_samples = None if window_samples is None else int(window_samples)
        self.window_start = int(window_start)
        if self.window_samples is not None and self.window_samples < 1:
            raise ValueError(f"window_samples must be positive, got {window_samples}.")
        if self.window_start < 0:
            raise ValueError(f"window_start must be >= 0, got {window_start}.")
        self.phase_columns = {
            phase: tuple(columns) for phase, columns in (phase_columns or DEFAULT_PHASE_COLUMNS).items()
        }
        if set(self.phase_columns) != {"P", "S"}:
            raise ValueError("phase_columns needs exactly the phases 'P' and 'S'.")
        self.test_trace_names = test_trace_names
        self.trace_category = trace_category
        self.max_traces = 8 if micro and max_traces is None else max_traces

    # -- metadata -------------------------------------------------------------------------------
    def _resolve(self, key: str, file_value: object) -> object:
        explicit = getattr(self, key)
        if explicit is not None:
            return explicit
        if file_value not in (None, ""):
            return file_value
        return self.defaults.get(key)

    def _test_names(self) -> Optional[set]:
        names = self.test_trace_names
        if names is None:
            return None
        if isinstance(names, (str, Path)):
            path = Path(names).expanduser()
            if path.suffix == ".npy":
                return {str(_decode(name)) for name in np.load(path, allow_pickle=True).tolist()}
            return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
        return {str(name) for name in names}

    def _split_of(self, row: Dict[str, str], test_names: Optional[set]) -> Optional[str]:
        if self.layout == "stead":
            if test_names is None:
                return "test"
            return "test" if row["trace_name"] in test_names else "train"
        if "split" not in row or row["split"] in (None, ""):
            return "test"
        return _SPLIT_NAMES.get(row["split"].strip().lower())

    def _arrival(self, row: Dict[str, str], phase: str) -> float:
        for column in self.phase_columns[phase]:
            value = _float(row.get(column))
            if not math.isnan(value):
                return value
        return math.nan

    # -- waveforms ------------------------------------------------------------------------------
    def _trace(self, handle, trace_name: str, dimension_order: str) -> np.ndarray:
        block, location = parse_trace_name(trace_name)
        if block not in handle["data"]:
            raise KeyError(f"Trace {trace_name!r} not found in {handle.filename}.")
        dataset = handle["data"][block]
        array = np.asarray(dataset[location] if location is not None else dataset[()], dtype=np.float32)
        order = dimension_order.upper()
        if len(order) == 3 and order[0] == "N":  # block arrays carry a leading trace axis
            order = order[1:]
        if array.ndim != 2 or sorted(order) != ["C", "W"]:
            raise ValueError(f"Trace {trace_name!r} has shape {array.shape}; expected 2-D data in order {dimension_order!r}.")
        return array if order == "CW" else array.T

    def _window(self, trace: np.ndarray) -> np.ndarray:
        if self.window_samples is None:
            return trace
        window = np.zeros((trace.shape[0], self.window_samples), dtype=np.float32)
        piece = trace[:, self.window_start : self.window_start + self.window_samples]
        window[:, : piece.shape[1]] = piece
        return window

    def _load(self) -> DataBundle:
        import h5py

        pairs = _stead_pairs(self.path) if self.layout == "stead" else _seisbench_chunks(self.path)
        test_names = self._test_names()
        collected: Dict[str, Dict[str, list]] = {
            split: {"x": [], "y": [], "names": []} for split in ("train", "val", "test")
        }
        component_order = sampling_rate = None
        for metadata_path, waveform_path in pairs:
            rows = _read_rows(metadata_path)
            with h5py.File(waveform_path, "r") as handle:
                data_format = read_data_format(handle) if self.layout == "seisbench" else {}
                chunk_components = self._resolve("component_order", data_format.get("component_order"))
                chunk_dimensions = self._resolve("dimension_order", data_format.get("dimension_order")) or "CW"
                chunk_rate = self._resolve("sampling_rate", data_format.get("sampling_rate"))
                if chunk_components is None:
                    raise ValueError(
                        f"{waveform_path} has no data_format/component_order; pass component_order (e.g. 'ZNE')."
                    )
                chunk_components = str(chunk_components).upper()
                if len(chunk_components) != 3:
                    raise ValueError(f"Expected three components, got {chunk_components!r}.")
                if component_order not in (None, chunk_components):
                    raise ValueError(f"Chunks disagree on the component order ({component_order} vs {chunk_components}).")
                component_order = chunk_components
                for row in rows:
                    if self.trace_category is not None and row.get("trace_category") != self.trace_category:
                        continue
                    split = self._split_of(row, test_names)
                    if split is None:
                        continue
                    bucket = collected[split]
                    if self.max_traces is not None and len(bucket["names"]) >= self.max_traces:
                        continue
                    rate = _float(row.get("trace_sampling_rate_hz"))
                    if math.isnan(rate) and not math.isnan(_float(row.get("trace_dt_s"))):
                        rate = 1.0 / _float(row.get("trace_dt_s"))
                    if math.isnan(rate):
                        rate = float(chunk_rate) if chunk_rate is not None else math.nan
                    if math.isnan(rate):
                        raise ValueError("No sampling rate in the metadata or data_format; pass sampling_rate.")
                    if sampling_rate is not None and not math.isclose(rate, sampling_rate):
                        raise ValueError(f"Mixed sampling rates ({sampling_rate} and {rate} Hz) are not supported.")
                    sampling_rate = rate
                    trace = self._window(self._trace(handle, row["trace_name"], str(chunk_dimensions)))
                    arrivals = []
                    for phase in ("P", "S"):
                        sample = self._arrival(row, phase) - self.window_start
                        inside = 0 <= sample < trace.shape[-1]
                        arrivals.append(sample if inside else math.nan)
                    bucket["x"].append(trace)
                    bucket["y"].append(arrivals)
                    bucket["names"].append(row["trace_name"])

        lengths = {trace.shape[-1] for split in collected.values() for trace in split["x"]}
        if len(lengths) > 1:
            raise ValueError(f"Traces have different lengths {sorted(lengths)}; pass window_samples.")
        length = lengths.pop() if lengths else int(self.window_samples or 0)
        splits = {}
        for split, bucket in collected.items():
            inputs = torch.from_numpy(np.stack(bucket["x"])) if bucket["x"] else torch.zeros(0, 3, length)
            targets = torch.tensor(bucket["y"], dtype=torch.float32).reshape(-1, 2)
            splits[split] = DataSplit(inputs, targets, metadata={"trace_names": bucket["names"]})
        if not any(len(bucket["names"]) for bucket in collected.values()):
            raise ValueError(f"No traces selected from {self.path}.")
        return DataBundle(
            splits=splits,
            feature_spec=FeatureSpec(
                channels=3,
                description=f"Three-component waveforms ({component_order}) read from {self.path.name}.",
                extra={"length": length, "sampling_rate": sampling_rate, "component_order": component_order},
            ),
            label_spec=LabelSpec(
                num_targets=2,
                task_type="picking",
                description="P and S arrival samples relative to the window start (NaN = no arrival).",
            ),
            metadata={
                "dataset": self.name,
                "source_dataset": self.preset or self.path.name,
                "hazard_task": "earthquake.picking",
                "layout": self.layout,
                "path": str(self.path),
                "sampling_rate": sampling_rate,
                "component_order": component_order,
                "synthetic": False,
            },
        )


__all__ = ["DEFAULT_PHASE_COLUMNS", "PRESETS", "SeisBenchWaveformDataset", "parse_trace_name", "read_data_format"]
