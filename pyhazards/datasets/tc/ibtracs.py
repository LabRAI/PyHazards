"""Reader for NOAA NCEI's International Best Track Archive for Climate Stewardship (IBTrACS) v04.

IBTrACS (Knapp et al. 2010, BAMS 91:363-376; Gahtan et al. 2024, doi:10.25921/82ty-9e16) merges the
best tracks of the WMO Regional Specialized Meteorological Centres and other agencies. NCEI
distributes it as CSV (``ibtracs.<subset>.list.v04r01.csv``) and netCDF
(``IBTrACS.<subset>.v04r01.nc``) files, where ``<subset>`` is ``ALL``, ``since1980``,
``last3years``, ``ACTIVE`` or a basin (``NA``, ``EP``, ``WP``, ``NI``, ``SI``, ``SP``, ``SA``).
Access follows the World Data Center for Meteorology policy (full and open access; WMO
Resolution 40 guides commercial use); PyHazards downloads the files from NCEI on request and never
redistributes them.

:func:`read_ibtracs` returns one row per track point with the CSV column names (``SID``,
``SEASON``, ``BASIN``, ``NAME``, ``ISO_TIME``, ``NATURE``, ``LAT``, ``LON``, ``WMO_WIND``,
``WMO_PRES``, ``WMO_AGENCY``, ``TRACK_TYPE``, ``USA_WIND``, ``TOKYO_WIND``, ``CMA_WIND``, ...) from
either format. CSV specifics handled here: the second line holds units, missing values are a single
space, and ``NA`` is the North Atlantic basin code, not a missing value. Positions: ``LAT``/``LON``
are IBTrACS' merged position (``LON`` stays continuous across the dateline, so it can exceed 180);
agency positions (``USA_LON``, ...) are wrapped to [-180, 180).

:class:`IBTrACSTropicalCycloneDataset` (registry name ``ibtracs_tracks``) turns the tracks into
forecasting samples: ``history`` six-hourly observations of the chosen variables before a forecast
time and the same variables at ``lead_hours`` after it. Winds are in knots (agency averaging periods
differ: 1 minute for the US agencies, 10 minutes for most others, 2 minutes for CMA; IBTrACS does
not convert them), pressures in hPa (mb).
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ..base import DataBundle, DataSplit, Dataset, FeatureSpec, LabelSpec

IBTRACS_VERSION = "v04r01"
IBTRACS_BASE_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access"
)
IBTRACS_SUBSETS = ("ALL", "since1980", "last3years", "ACTIVE", "NA", "EP", "WP", "NI", "SI", "SP", "SA")
BASINS = ("NA", "SA", "EP", "WP", "NI", "SI", "SP")
# Agencies with <AGENCY>_LAT/_LON/_WIND/_PRES columns (WMO has only WIND/PRES).
AGENCIES = ("wmo", "usa", "tokyo", "cma", "hko", "kma", "newdelhi", "reunion", "bom", "nadi", "wellington", "ds824", "td9636", "td9635", "neumann", "mlc")
REQUIRED_COLUMNS = ("SID", "SEASON", "BASIN", "NAME", "ISO_TIME", "LAT", "LON")
STRING_COLUMNS = (
    "SID", "BASIN", "SUBBASIN", "NAME", "ISO_TIME", "NATURE", "WMO_AGENCY", "TRACK_TYPE", "MAIN_TRACK_SID",
    "IFLAG", "USA_AGENCY", "USA_ATCF_ID", "USA_RECORD", "USA_STATUS",
)
VARIABLE_UNITS = {"lat": "degrees_north", "lon": "degrees_east", "wind": "kt", "pres": "hPa"}


def ibtracs_url(subset: str = "ALL", fmt: str = "csv") -> str:
    """NCEI download URL of an IBTrACS v04r01 file."""
    if subset not in IBTRACS_SUBSETS:
        raise ValueError(f"unknown IBTrACS subset {subset!r}; choose from {IBTRACS_SUBSETS}")
    if fmt == "csv":
        return f"{IBTRACS_BASE_URL}/csv/ibtracs.{subset}.list.{IBTRACS_VERSION}.csv"
    if fmt in {"netcdf", "nc"}:
        return f"{IBTRACS_BASE_URL}/netcdf/IBTrACS.{subset}.{IBTRACS_VERSION}.nc"
    raise ValueError("fmt must be 'csv' or 'netcdf'")


def download_ibtracs(dest: Union[str, Path], subset: str = "ALL", fmt: str = "csv", overwrite: bool = False) -> Path:
    """Download an IBTrACS file from NCEI into ``dest`` (the ALL CSV is about 300 MB).

    IBTrACS is updated several times a week, so files cannot be pinned by checksum; the dataset
    records the sha256 of the file it read in its metadata.
    """
    url = ibtracs_url(subset, fmt)
    dest = Path(dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / url.rsplit("/", 1)[1]
    if overwrite or not target.exists():
        partial = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(url, partial)
        partial.replace(target)
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finish_table(frame):
    import pandas as pd

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"not an IBTrACS v04 table: missing columns {missing}")
    for column in frame.columns:
        if column == "ISO_TIME":
            continue
        if column in STRING_COLUMNS:
            frame[column] = frame[column].fillna("").astype(str).str.strip()
            continue
        # Numeric columns become floats; category columns (e.g. NEWDELHI_GRADE, DS824_STAGE) stay text.
        converted = pd.to_numeric(frame[column], errors="coerce")
        if int(converted.notna().sum()) == int(frame[column].notna().sum()):
            frame[column] = converted
    frame["ISO_TIME"] = pd.to_datetime(frame["ISO_TIME"], format="%Y-%m-%d %H:%M:%S")
    frame = frame.sort_values(["SID", "ISO_TIME"], kind="stable").reset_index(drop=True)
    return frame


def read_ibtracs_csv(path: Union[str, Path], usecols: Optional[Iterable[str]] = None):
    """Read an IBTrACS v04 CSV file into a pandas DataFrame (one row per track point)."""
    import pandas as pd

    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
        second = handle.readline().rstrip("\r\n").split(",")
    if "SID" not in header:
        raise ValueError(f"{path} does not start with the IBTrACS header (SID, SEASON, ...)")
    # Line 2 is the units row (" ,Year, , , , , , ,degrees_north,degrees_east,kts,mb,...").
    units_row = len(second) == len(header) and second[0].strip() == "" and "degrees_north" in second
    if usecols is not None:
        usecols = [column for column in header if column in set(usecols) | set(REQUIRED_COLUMNS)]
    frame = pd.read_csv(
        path,
        skiprows=[1] if units_row else None,
        usecols=usecols,
        dtype=str,
        keep_default_na=False,  # "NA" is the North Atlantic basin
        na_values=[" ", ""],
    )
    return _finish_table(frame)


def _decode(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind == "S":
        return np.char.strip(np.char.decode(values, "utf-8", errors="replace"))
    return values


def read_ibtracs_netcdf(path: Union[str, Path], variables: Optional[Iterable[str]] = None):
    """Read an IBTrACS v04 netCDF file (dimensions ``storm`` x ``date_time``) into the CSV table layout."""
    import pandas as pd
    import xarray as xr

    with xr.open_dataset(Path(path), decode_times=False, mask_and_scale=True) as ds:
        if "sid" not in ds or "iso_time" not in ds or "numobs" not in ds:
            raise ValueError(f"{path} is not an IBTrACS v04 netCDF file (needs sid, iso_time, numobs)")
        wanted = list(ds.variables) if variables is None else sorted(set(v.lower() for v in variables) | {c.lower() for c in REQUIRED_COLUMNS})
        storm_count, slot_count = ds.sizes["storm"], ds.sizes["date_time"]
        numobs = np.nan_to_num(ds["numobs"].values).astype(int)
        valid = np.arange(slot_count)[None, :] < numobs[:, None]
        columns: Dict[str, np.ndarray] = {}
        for name in wanted:
            if name not in ds or name in {"time", "numobs"}:
                continue
            var = ds[name]
            dims = var.dims
            if dims == ("storm",):
                values = np.repeat(_decode(var.values)[:, None], slot_count, axis=1)
            elif dims == ("storm", "date_time"):
                values = _decode(var.values)
            else:
                continue  # wind radii and other per-quadrant variables are not tabulated
            columns[name.upper()] = values[valid]
    frame = pd.DataFrame(columns)
    for column in frame.columns:
        if frame[column].dtype.kind in "OUS":
            frame[column] = frame[column].replace("", np.nan)
    return _finish_table(frame)


def read_ibtracs(path: Union[str, Path], columns: Optional[Iterable[str]] = None):
    """Read an IBTrACS v04 CSV or netCDF file (chosen by suffix) into a DataFrame."""
    path = Path(path)
    if path.suffix.lower() in {".nc", ".nc4", ".cdf", ".netcdf"}:
        return read_ibtracs_netcdf(path, columns)
    return read_ibtracs_csv(path, columns)


def _variable_columns(agency: str, positions: str) -> Dict[str, str]:
    agency = agency.lower()
    if agency not in AGENCIES:
        raise ValueError(f"unknown agency {agency!r}; choose from {AGENCIES}")
    if positions not in {"merged", "agency"}:
        raise ValueError("positions must be 'merged' (LAT/LON) or 'agency' (<AGENCY>_LAT/_LON)")
    prefix = agency.upper()
    if positions == "agency" and agency == "wmo":
        raise ValueError("WMO has no position columns; use positions='merged'")
    lat, lon = ("LAT", "LON") if positions == "merged" else (f"{prefix}_LAT", f"{prefix}_LON")
    return {"lat": lat, "lon": lon, "wind": f"{prefix}_WIND", "pres": f"{prefix}_PRES"}


def _unwrap_degrees(lon: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.unwrap(np.deg2rad(lon)))


def build_track_samples(
    table,
    variables: Sequence[str] = ("lat", "lon", "wind", "pres"),
    history: int = 4,
    lead_hours: Sequence[int] = (6, 12, 18, 24),
    agency: str = "usa",
    positions: str = "merged",
    basins: Optional[Sequence[str]] = None,
    include_spur: bool = False,
    min_wind: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[Any]]]:
    """Forecast windows from an IBTrACS table.

    Keeps synoptic times (00, 06, 12, 18 UTC). A window ending at time ``t`` needs observations at
    ``t - 6 h * (history - 1) ... t`` and at ``t + lead`` for every lead, all with the requested
    variables present. Longitudes are unwrapped within each window so they are continuous.
    Returns inputs ``(N, history, V)``, targets ``(N, len(lead_hours), V)`` and per-sample
    metadata lists.
    """
    variables = [v.lower() for v in variables]
    unknown = sorted(set(variables) - set(VARIABLE_UNITS))
    if unknown or not variables:
        raise ValueError(f"unknown variables {unknown}; choose from {sorted(VARIABLE_UNITS)}")
    lead_steps = []
    for hours in lead_hours:
        if int(hours) != hours or int(hours) <= 0 or int(hours) % 6:
            raise ValueError("lead_hours must be positive multiples of 6")
        lead_steps.append(int(hours) // 6)
    history = int(history)
    if history < 1:
        raise ValueError("history must be at least 1")
    columns = _variable_columns(agency, positions)
    missing = [columns[v] for v in variables if columns[v] not in table.columns]
    if missing:
        raise ValueError(f"IBTrACS table has no column(s) {missing} for agency={agency!r}")

    frame = table
    times = frame["ISO_TIME"]
    frame = frame[(times.dt.hour % 6 == 0) & (times.dt.minute == 0) & (times.dt.second == 0)]
    if not include_spur and "TRACK_TYPE" in frame.columns:
        frame = frame[~frame["TRACK_TYPE"].str.contains("spur", case=False, na=False)]

    inputs, targets = [], []
    meta: Dict[str, List[Any]] = {"sid": [], "name": [], "basin": [], "season": [], "iso_time": []}
    six_hours = np.timedelta64(6, "h")
    max_lead = max(lead_steps)
    for sid, storm in frame.groupby("SID", sort=True):
        stamps = storm["ISO_TIME"].to_numpy(dtype="datetime64[ns]")
        values = np.stack([storm[columns[v]].to_numpy(dtype=np.float64) for v in variables], axis=1)
        index_of = {stamp: i for i, stamp in enumerate(stamps)}
        basin_col = storm["BASIN"].to_numpy()
        name, season = storm["NAME"].iloc[0], storm["SEASON"].iloc[0]
        for i in range(history - 1, len(stamps)):
            origin = stamps[i]
            wanted = [origin - six_hours * k for k in range(history - 1, -1, -1)]
            wanted += [origin + six_hours * step for step in lead_steps]
            if origin + six_hours * max_lead > stamps[-1]:
                break
            rows = [index_of.get(stamp) for stamp in wanted]
            if any(row is None for row in rows):
                continue
            window = values[rows]
            if np.isnan(window).any():
                continue
            if basins is not None and basin_col[i] not in basins:
                continue
            if min_wind is not None and "wind" in variables and window[history - 1, variables.index("wind")] < min_wind:
                continue
            if "lon" in variables:
                col = variables.index("lon")
                window[:, col] = _unwrap_degrees(window[:, col])
            inputs.append(window[:history])
            targets.append(window[history:])
            meta["sid"].append(sid)
            meta["name"].append(name)
            meta["basin"].append(basin_col[i])
            meta["season"].append(int(season))
            meta["iso_time"].append(str(np.datetime_as_string(origin, unit="s")))
    width = len(variables)
    x = np.asarray(inputs, dtype=np.float32).reshape(-1, history, width)
    y = np.asarray(targets, dtype=np.float32).reshape(-1, len(lead_steps), width)
    return x, y, meta


def _season_split(seasons: Sequence[int], sids: Sequence[str], splits: Mapping[str, Optional[Sequence[int]]]) -> Dict[str, np.ndarray]:
    seasons = np.asarray(seasons)
    given = {name: years for name, years in splits.items() if years is not None}
    if given:
        masks = {name: np.isin(seasons, list(years)) for name, years in given.items()}
        if "train" not in given:
            other = np.zeros(len(seasons), dtype=bool)
            for mask in masks.values():
                other |= mask
            masks["train"] = ~other
        return masks
    distinct = sorted(set(seasons.tolist()))
    if len(distinct) >= 3:
        return {
            "train": np.isin(seasons, distinct[:-2]),
            "val": seasons == distinct[-2],
            "test": seasons == distinct[-1],
        }
    # Fewer than three seasons: split whole storms (sorted by SID) 70 / 15 / 15.
    storms = sorted(set(sids))
    n = len(storms)
    train_end = max(1, int(round(0.7 * n)))
    val_end = min(n, max(train_end, int(round(0.85 * n))))
    groups = {"train": set(storms[:train_end]), "val": set(storms[train_end:val_end]), "test": set(storms[val_end:])}
    return {name: np.asarray([sid in members for sid in sids], dtype=bool) for name, members in groups.items()}


class IBTrACSTropicalCycloneDataset(Dataset):
    """Best-track forecasting samples read from an IBTrACS v04 CSV or netCDF file.

    Inputs ``(samples, history, V)`` and targets ``(samples, len(lead_hours), V)`` hold the
    ``variables`` (default latitude, longitude, maximum sustained wind in knots and central
    pressure in hPa) in physical units. Splits are by season: pass ``train_seasons``,
    ``val_seasons``, ``test_seasons`` (seasons not listed go to train), otherwise the last season
    is the test split, the one before validation and the rest training.
    """

    name = "ibtracs_tracks"

    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        cache_dir: Optional[str] = None,
        subset: str = "ALL",
        file_format: str = "csv",
        download: bool = False,
        variables: Sequence[str] = ("lat", "lon", "wind", "pres"),
        history: int = 4,
        lead_hours: Sequence[int] = (6, 12, 18, 24),
        agency: str = "usa",
        positions: str = "merged",
        basins: Optional[Sequence[str]] = None,
        include_spur: bool = False,
        min_wind: Optional[float] = None,
        train_seasons: Optional[Sequence[int]] = None,
        val_seasons: Optional[Sequence[int]] = None,
        test_seasons: Optional[Sequence[int]] = None,
    ):
        super().__init__(cache_dir=cache_dir)
        if path is None:
            if cache_dir is None:
                raise ValueError(
                    "ibtracs_tracks needs path= (an IBTrACS v04 CSV or netCDF file) or cache_dir= with download=True"
                )
            name = ibtracs_url(subset, file_format).rsplit("/", 1)[1]
            candidate = Path(cache_dir).expanduser() / name
            if not candidate.exists():
                if not download:
                    raise FileNotFoundError(f"{candidate} not found; pass download=True to fetch it from NCEI")
                candidate = download_ibtracs(cache_dir, subset=subset, fmt=file_format)
            path = candidate
        self.path = Path(path)
        if basins is not None:
            basins = [b.upper() for b in basins]
            unknown = sorted(set(basins) - set(BASINS))
            if unknown:
                raise ValueError(f"unknown basins {unknown}; choose from {BASINS}")
        self.variables = tuple(v.lower() for v in variables)
        self.history = int(history)
        self.lead_hours = tuple(int(h) for h in lead_hours)
        self.agency = agency.lower()
        self.positions = positions
        self.basins = basins
        self.include_spur = include_spur
        self.min_wind = min_wind
        self.splits = {"train": train_seasons, "val": val_seasons, "test": test_seasons}

    def _load(self) -> DataBundle:
        columns = _variable_columns(self.agency, self.positions)
        needed = {"TRACK_TYPE", *(columns[v] for v in self.variables)}
        table = read_ibtracs(self.path, needed)
        x, y, meta = build_track_samples(
            table,
            variables=self.variables,
            history=self.history,
            lead_hours=self.lead_hours,
            agency=self.agency,
            positions=self.positions,
            basins=self.basins,
            include_spur=self.include_spur,
            min_wind=self.min_wind,
        )
        if len(x) == 0:
            raise ValueError(f"no complete {self.history}-step + {self.lead_hours} h windows in {self.path}")
        masks = _season_split(meta["season"], meta["sid"], self.splits)
        splits = {}
        for split_name in ("train", "val", "test"):
            mask = masks.get(split_name, np.zeros(len(x), dtype=bool))
            index = np.flatnonzero(mask)
            splits[split_name] = DataSplit(
                torch.from_numpy(x[index]),
                torch.from_numpy(y[index]),
                metadata={key: [values[i] for i in index] for key, values in meta.items()},
            )
        units = {v: VARIABLE_UNITS[v] for v in self.variables}
        return DataBundle(
            splits=splits,
            feature_spec=FeatureSpec(
                input_dim=len(self.variables),
                description=f"{self.history} six-hourly IBTrACS observations of {', '.join(self.variables)}.",
                extra={"history": self.history, "variables": list(self.variables)},
            ),
            label_spec=LabelSpec(
                num_targets=len(self.variables),
                task_type="regression",
                description=f"Best-track {', '.join(self.variables)} at {list(self.lead_hours)} h.",
            ),
            metadata={
                "dataset": self.name,
                "source_dataset": "IBTrACS " + IBTRACS_VERSION,
                "hazard_task": "tc.track_intensity",
                "lead_hours": list(self.lead_hours),
                "target_variables": list(self.variables),
                "units": units,
                "agency": self.agency,
                "positions": self.positions,
                "source_file": str(self.path),
                "source_sha256": _sha256(self.path),
                "synthetic": False,
            },
        )


__all__ = [
    "AGENCIES",
    "BASINS",
    "IBTRACS_SUBSETS",
    "IBTrACSTropicalCycloneDataset",
    "build_track_samples",
    "download_ibtracs",
    "ibtracs_url",
    "read_ibtracs",
    "read_ibtracs_csv",
    "read_ibtracs_netcdf",
]
