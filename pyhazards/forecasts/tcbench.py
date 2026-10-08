"""Readers for TCBench's released weather-model cyclone tracks and forecast fields.

TCBench (Gomez et al., "TCBench: A Benchmark for Tropical Cyclone Track and Intensity Forecasting at
the Global Scale", arXiv:2601.23268, 2026; code https://github.com/msgomez06/TCBench_Alpha, MIT) ran
Pangu-Weather, FourCastNet v2 (SFNO, "small") and AIFS for 2023, tracked cyclones in their outputs
with TempestExtremes and matched the tracks to IBTrACS. The Hugging Face dataset ``TCBench/TCBench``
(MIT) holds, per model:

* ``matched_tracks/2023_<model>.csv``: tracks matched to IBTrACS storms (``SID``, ``Initial Time``,
  ``Valid Time``, ``wind max`` in m/s, ``pressure min`` in Pa, ``lat``, ``lon``), a few MB;
* ``unmatched_tracks/<dir>/<file>.csv``: TempestExtremes StitchNodes output per initial time;
* ``neural_weather_models/<dir>/<file>.nc``: the raw forecast fields (about 3 GB per initial time,
  1.1-2.7 TB per model). :func:`read_tcbench_fields` reads only the variables a tracker needs with
  HTTP range requests (about 200 MB for the TempestExtremes variables), never the whole file.

Everything is read at run time from revision :data:`TCBENCH_REVISION`; nothing is redistributed.
The Pangu-Weather forecasts derive from weights licensed CC BY-NC-SA 4.0 (non-commercial); TCBench's
FourCastNet v2 is NVIDIA's ``sfno_73ch_small``, not the FourCastNet (AFNO) model of Pathak et al. 2022.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

import numpy as np

__all__ = [
    "TCBENCH_FILES",
    "TCBENCH_MODELS",
    "TCBENCH_REPO",
    "TCBENCH_REVISION",
    "HTTPRangeFile",
    "download_tcbench_file",
    "read_tcbench_fields",
    "read_tcbench_matched_tracks",
    "read_tcbench_unmatched_tracks",
    "tcbench_ibtracs",
    "tcbench_path",
    "tcbench_url",
]

TCBENCH_REPO = "TCBench/TCBench"
TCBENCH_REVISION = "0124d14d7f1f468096f46aec1e79696ab9c880c1"

TCBENCH_FILES: Dict[str, str] = {
    "matched_tracks/2023_PANGU.csv": "0b5f7a51531a34c70b190a9e9bbec10bae3a7a009badfd9995d61b882e34e348",
    "matched_tracks/2023_fcnet.csv": "fe4197793b072ce94de6ad0bce4abefa512ba89784f38eb9068e6475fed47323",
    "matched_tracks/2023_aifs.csv": "ec93a38582f970043ddf497d73bba836d3ccb6c1f8dc7483f0f00494461056f9",
}
"""sha256 of the pinned small files (checked on download)."""

TCBENCH_MODELS: Dict[str, Dict[str, str]] = {
    "pangu": {
        "label": "Pangu-Weather",
        "matched": "matched_tracks/2023_PANGU.csv",
        "unmatched_dir": "unmatched_tracks/2023_pangu",
        "fields_dir": "neural_weather_models/panguweather",
        "stem": "panguweather_{init:%Y.%m.%d-%Hh%M}_maxltd-120_timeres-6",
    },
    "fourcastnet_v2": {
        "label": "FourCastNet v2 (SFNO small)",
        "matched": "matched_tracks/2023_fcnet.csv",
        "unmatched_dir": "unmatched_tracks/2023_fcnet",
        "fields_dir": "neural_weather_models/fourcastnetv2_small",
        "stem": "fcnet_{init:%Y.%m.%d-%Hh%Mm}_maxltd-120_timeres-6",
    },
    "aifs": {
        "label": "AIFS v1.0",
        "matched": "matched_tracks/2023_aifs.csv",
        "unmatched_dir": "unmatched_tracks/2023_aifs",
        "fields_dir": "neural_weather_models/AIFS",
        "stem": "AIFS_init-{init:%Y.%m.%d-%Hh%M}_max-lead-120",
    },
}


def tcbench_url(path: str, revision: str = TCBENCH_REVISION) -> str:
    return f"https://huggingface.co/datasets/{TCBENCH_REPO}/resolve/{revision}/{path}"


def _model(model: str) -> Dict[str, str]:
    if model not in TCBENCH_MODELS:
        raise ValueError(f"unknown TCBench model {model!r}; choose from {sorted(TCBENCH_MODELS)}")
    return TCBENCH_MODELS[model]


def tcbench_path(model: str, kind: str, init_time=None) -> str:
    """Repository path of a model's ``matched`` file, or of its ``unmatched`` / ``fields`` file for ``init_time``."""
    import pandas as pd

    spec = _model(model)
    if kind == "matched":
        return spec["matched"]
    if init_time is None:
        raise ValueError(f"kind={kind!r} needs init_time")
    stem = spec["stem"].format(init=pd.Timestamp(init_time))
    if kind == "unmatched":
        return f"{spec['unmatched_dir']}/{stem}.csv"
    if kind == "fields":
        return f"{spec['fields_dir']}/{stem}.nc"
    raise ValueError("kind must be 'matched', 'unmatched' or 'fields'")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_tcbench_file(path: str, cache_dir: Union[str, Path], overwrite: bool = False) -> Path:
    """Download a (small) TCBench file at the pinned revision into ``cache_dir``.

    Files listed in :data:`TCBENCH_FILES` are checked against their sha256. Raw forecast files are
    refused (use :func:`read_tcbench_fields`, which reads only the needed variables).
    """
    if path.startswith("neural_weather_models/"):
        raise ValueError("raw forecast files are ~3 GB each; read slices with read_tcbench_fields()")
    target = Path(cache_dir).expanduser() / TCBENCH_REVISION / path
    if overwrite or not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(tcbench_url(path), partial)
        partial.replace(target)
    expected = TCBENCH_FILES.get(path)
    if expected is not None:
        found = _sha256(target)
        if found != expected:
            raise ValueError(f"{target} has sha256 {found}, expected {expected}")
    return target


def read_tcbench_matched_tracks(path: Union[str, Path], units: str = "si"):
    """Read a TCBench ``matched_tracks`` CSV into a forecast-track table (see :mod:`.scoring`).

    The released files hold winds in m/s and pressures in Pa (``units="si"``); TCBench's own
    ``track_matcher.py`` writes knots and hPa (``units="kt_hpa"``), which are converted back with its
    factors. Ensemble files keep their ``ensemble_idx`` column as ``member``.
    """
    import pandas as pd

    from .scoring import TCBENCH_KT_PER_MS

    frame = pd.read_csv(path)
    required = ["SID", "Initial Time", "Valid Time", "wind max", "pressure min", "lat", "lon"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} is not a TCBench matched-tracks file: missing {missing}")
    if units not in {"si", "kt_hpa"}:
        raise ValueError("units must be 'si' or 'kt_hpa'")
    init = pd.to_datetime(frame["Initial Time"])
    valid = pd.to_datetime(frame["Valid Time"])
    wind = frame["wind max"].astype(float)
    pres = frame["pressure min"].astype(float)
    if units == "kt_hpa":
        wind, pres = wind / TCBENCH_KT_PER_MS, pres * 100.0
    out = pd.DataFrame(
        {
            "SID": frame["SID"].astype(str),
            "init_time": init,
            "valid_time": valid,
            "lead_hours": (valid - init).dt.total_seconds() / 3600.0,
            "lat": frame["lat"].astype(float),
            "lon": frame["lon"].astype(float),
            "wind_ms": wind,
            "pres_pa": pres,
        }
    )
    ensemble = next((c for c in frame.columns if "ensemble" in str(c).lower()), None)
    if ensemble is not None:
        out["member"] = frame[ensemble]
    return out


def read_tcbench_unmatched_tracks(path: Union[str, Path]):
    """Read a TCBench ``unmatched_tracks`` CSV (TempestExtremes StitchNodes output)."""
    from .tempest import read_stitchnodes_csv

    return read_stitchnodes_csv(path)


def tcbench_ibtracs(table, year: int = 2023):
    """IBTrACS rows TCBench evaluates against: records whose ``ISO_TIME`` falls in ``year``."""
    import pandas as pd

    return table[pd.to_datetime(table["ISO_TIME"]).dt.year == year].reset_index(drop=True)


class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable file over HTTP range requests (for opening remote HDF5 files with h5py)."""

    def __init__(self, url: str, block_size: int = 1 << 22, max_blocks: int = 256, session=None):
        import requests

        self._session = session or requests.Session()
        response = self._session.head(url, allow_redirects=True, timeout=60)
        response.raise_for_status()
        self.url = url
        self.size = int(response.headers["Content-Length"])
        self.block_size = int(block_size)
        self.max_blocks = int(max_blocks)
        self._blocks: Dict[int, bytes] = {}
        self._pos = 0
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        else:
            self._pos = self.size + offset
        return self._pos

    def _block(self, index: int) -> bytes:
        if index not in self._blocks:
            start = index * self.block_size
            end = min(self.size, start + self.block_size) - 1
            response = self._session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, allow_redirects=True, timeout=300)
            response.raise_for_status()
            data = response.content
            if len(data) != end - start + 1:
                raise IOError(f"range request returned {len(data)} bytes, expected {end - start + 1}")
            self._blocks[index] = data
            self.bytes_read += len(data)
            while len(self._blocks) > self.max_blocks:
                self._blocks.pop(next(iter(self._blocks)))
        return self._blocks[index]

    def readinto(self, buffer) -> int:
        end = min(self.size, self._pos + len(buffer))
        out = bytearray()
        position = self._pos
        while position < end:
            index = position // self.block_size
            block = self._block(index)
            offset = position - index * self.block_size
            take = min(len(block) - offset, end - position)
            out += block[offset : offset + take]
            position += take
        buffer[: len(out)] = out
        self._pos = end
        return len(out)


_SURFACE = {"msl": "msl", "u10": "u10", "v10": "v10", "t2m": "t2m"}
_LEVEL_PREFIX = ("u", "v", "z", "t", "q", "r")


def _split_variable(name: str):
    if name in _SURFACE:
        return _SURFACE[name], None
    for prefix in _LEVEL_PREFIX:
        rest = name[len(prefix):]
        if name.startswith(prefix) and rest.isdigit():
            return prefix, int(rest)
    raise ValueError(f"cannot map variable {name!r} to a TCBench field (use msl, u10, v10, t2m or u/v/z/t/q/r<hPa>)")


def read_tcbench_fields(
    model: str,
    init_time,
    variables: Sequence[str] = ("msl", "u10", "v10", "z300", "z500"),
    path: Optional[Union[str, Path]] = None,
    url: Optional[str] = None,
):
    """Read selected variables of a TCBench raw forecast as an ``xarray.Dataset`` (tracker conventions).

    Reads ``path`` (a local copy) or, by default, the pinned file on the Hugging Face Hub with HTTP
    range requests, fetching only the requested variables and pressure levels. Values are unpacked
    from int16 with the files' ``scale_factor`` / ``add_offset`` in double precision and returned as
    float32 (this is the input that reproduces TCBench's TempestExtremes tracks exactly). ``time``
    holds the valid times (lead times 6-120 h).
    """
    import h5py
    import pandas as pd
    import xarray as xr

    init = pd.Timestamp(init_time)
    if path is not None:
        handle = h5py.File(Path(path), "r")
        source = str(path)
    else:
        source = url or tcbench_url(tcbench_path(model, "fields", init))
        handle = h5py.File(HTTPRangeFile(source), "r")
    try:
        lat = handle["latitude"][...].astype(np.float64)
        lon = handle["longitude"][...].astype(np.float64)
        leads = handle["leadtime_hours"][...].astype(np.int64)
        levels = list(handle["level"][...].astype(int)) if "level" in handle else []
        data = {}
        for name in variables:
            source_name, level = _split_variable(name)
            if source_name not in handle:
                raise KeyError(f"{source} has no variable {source_name!r}")
            dataset = handle[source_name]
            dims = [dim[0].name.rsplit("/", 1)[-1] if len(dim) else "" for dim in dataset.dims]
            index = []
            for dim in dims:
                if dim == "time":
                    index.append(0)
                elif dim == "level":
                    if level not in levels:
                        raise KeyError(f"level {level} hPa not in {levels}")
                    index.append(levels.index(level))
                else:
                    index.append(slice(None))
            raw = dataset[tuple(index)]
            kept = [dim for dim, idx in zip(dims, index) if isinstance(idx, slice)]
            order = [kept.index(d) for d in ("leadtime_hours", "latitude", "longitude")]
            raw = np.transpose(raw, order)
            scale = float(np.asarray(dataset.attrs.get("scale_factor", 1.0)).ravel()[0])
            offset = float(np.asarray(dataset.attrs.get("add_offset", 0.0)).ravel()[0])
            fill = dataset.attrs.get("_FillValue")
            values = raw.astype(np.float64) * scale + offset
            if fill is not None:
                values[raw == np.asarray(fill).ravel()[0]] = np.nan
            data[name] = (("time", "lat", "lon"), values.astype(np.float32))
    finally:
        handle.close()
    times = [init + pd.Timedelta(hours=int(h)) for h in leads]
    ds = xr.Dataset(data, coords={"time": times, "lat": lat, "lon": lon})
    ds.attrs.update({"source": source, "model": model, "init_time": str(init), "tcbench_revision": TCBENCH_REVISION})
    return ds
