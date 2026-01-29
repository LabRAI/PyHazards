# pyhazards/datasets/inspection.py
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from datetime import date, timezone

import numpy as np
import pandas as pd
import xarray as xr
import h5py
import matplotlib.pyplot as plt
import requests


# ---------------------------------------------------------------------
# Optional: notebook-style display, but safe for pure python
# ---------------------------------------------------------------------
try:
    from IPython.display import display  # type: ignore
except Exception:
    def display(x):
        if hasattr(x, "to_string"):
            print(x.to_string(index=False))
        else:
            print(x)


# ---------------------------------------------------------------------
# Constants / Defaults
# ---------------------------------------------------------------------
RAW_DATASETS = [
    "M2I1NXASM",   # inst1_2d_asm_Nx
    "M2INVASM",    # alias folder name for inst3_3d_asm_Nv (normally M2I3NVASM)
    "M2T1NXFLX",   # tavg1_2d_flx_Nx
    "M2T1NXLND",   # tavg1_2d_lnd_Nx
    "M2T1NXRAD",   # tavg1_2d_rad_Nx
    "M2C0NXCTM",   # const_2d_ctm_Nx (static)
]

PATTERN_SFC_OUT  = "MERRA2_sfc_{yyyymmdd}.nc"
PATTERN_PRES_OUT = "MERRA_pres_{yyyymmdd}.nc"

# Merge SFC variable subsets (from mergesfc.py)
ASM1_VARS   = ["QV2M", "T2M", "TQI", "TQL", "TQV", "TS", "U10M", "V10M"]
FLX_VARS    = ["EFLUX", "HFLUX", "Z0M"]
RAD_VARS    = ["SWGNT", "SWTNT", "LWGAB", "LWGEM", "LWTUP"]
NV_VARS     = ["SLP", "PS"]
LND_VARS    = ["GWETROOT", "LAI"]
STATIC_VARS = ["FRACI", "FRLAND", "FROCEAN", "PHIS"]

TARGET_N_FRAMES = 8
TARGET_LAT = 361
TARGET_LON = 576
HOURS8 = [0, 3, 6, 9, 12, 15, 18, 21]

# Merge PRES (from mergepres.py)
PRES_VARS = ["CLOUD", "H", "OMEGA", "PL", "QI", "QL", "QV", "T", "U", "V"]
PRES_LEVELS = [
    34.0, 39.0, 41.0, 43.0, 44.0, 45.0, 48.0,
    51.0, 53.0, 56.0, 63.0, 68.0, 71.0, 72.0,
]

# Try to import NAN_VALS (mergesfc depends on it)
try:
    from PrithviWxC.definitions import NAN_VALS  # type: ignore
except Exception as e:
    NAN_VALS = {}
    print("[WARN] Cannot import PrithviWxC.definitions.NAN_VALS. "
          "NaN filling will be skipped. Error:", e)


# ---------------------------------------------------------------------
# Helpers: repo root inference + date formatting
# ---------------------------------------------------------------------
def infer_repo_root() -> Path:
    """
    Infer REPO_ROOT from this file location.
    We want outputs to land at:
      REPO_ROOT/Prithvi-WxC/data/merra-2
      REPO_ROOT/M2I1NXASM, REPO_ROOT/M2INVASM, ...
    Works no matter where the repo is cloned.

    Heuristic:
      - Prefer a parent directory that contains BOTH 'Prithvi-WxC/' and 'pyhazards/'.
      - Fallback: any parent that contains 'Prithvi-WxC/'.
    """
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    for p in [here] + list(here.parents):
        if (p / "Prithvi-WxC").is_dir():
            candidates.append(p)

    if not candidates:
        raise RuntimeError(
            f"Cannot infer repo root from {here}. "
            f"Expected a parent dir containing 'Prithvi-WxC/'. "
            f"Please pass --repo-root."
        )

    for p in candidates:
        if (p / "pyhazards").is_dir():
            return p

    return candidates[0]


def yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")


def get_stream_number(d: date) -> str:
    """Same logic as merra2.py."""
    y = d.year
    if 1980 <= y <= 1991:
        return "100"
    elif 1992 <= y <= 2000:
        return "200"
    elif 2001 <= y <= 2010:
        return "300"
    else:
        return "400"


# ---------------------------------------------------------------------
# PART 1) Download raw datasets (adapted from merra2.py, but NO hardcoded creds)
# ---------------------------------------------------------------------
PRODUCT_INFO = {
    "M2I1NXASM": {
        "host": "https://goldsmr4.gesdisc.eosdis.nasa.gov",
        "collection": "M2I1NXASM.5.12.4",
        "prefix": "inst1_2d_asm_Nx",
        "has_date": True,
        "data_root": "MERRA2",
    },
    # Official code is M2I3NVASM, but user wants folder name M2INVASM.
    "M2I3NVASM": {
        "host": "https://goldsmr5.gesdisc.eosdis.nasa.gov",
        "collection": "M2I3NVASM.5.12.4",
        "prefix": "inst3_3d_asm_Nv",
        "has_date": True,
        "data_root": "MERRA2",
    },
    "M2T1NXFLX": {
        "host": "https://goldsmr4.gesdisc.eosdis.nasa.gov",
        "collection": "M2T1NXFLX.5.12.4",
        "prefix": "tavg1_2d_flx_Nx",
        "has_date": True,
        "data_root": "MERRA2",
    },
    "M2T1NXLND": {
        "host": "https://goldsmr4.gesdisc.eosdis.nasa.gov",
        "collection": "M2T1NXLND.5.12.4",
        "prefix": "tavg1_2d_lnd_Nx",
        "has_date": True,
        "data_root": "MERRA2",
    },
    "M2T1NXRAD": {
        "host": "https://goldsmr4.gesdisc.eosdis.nasa.gov",
        "collection": "M2T1NXRAD.5.12.4",
        "prefix": "tavg1_2d_rad_Nx",
        "has_date": True,
        "data_root": "MERRA2",
    },
    "M2C0NXCTM": {
        "host": "https://goldsmr4.gesdisc.eosdis.nasa.gov",
        "collection": "M2C0NXCTM.5.12.4",
        "prefix": "const_2d_ctm_Nx",
        "has_date": False,
        "filename": "MERRA2_101.const_2d_ctm_Nx.00000000.nc4",
        # ✅ 常量集合通常在 MONTHLY 根目录
        "data_root": "MERRA2_MONTHLY",
        # ✅ 很常见的子目录布局（你的 404 就是少了它）
        "subdir": "1980",
    },
}


def build_file_url(product_code: str, d: date | None) -> tuple[str, str]:
    """
    Build a direct HTTPS URL to the granule.
    - Dated products: .../data/<root>/<collection>/<YYYY>/<MM>/<filename>
    - Const products: .../data/<root>/<collection>/<subdir?>/<filename>
    """
    info = PRODUCT_INFO[product_code]
    host = info["host"].rstrip("/")
    collection = info["collection"].strip("/")
    data_root = info.get("data_root", "MERRA2").strip("/")
    base = f"{host}/data/{data_root}/{collection}"

    # const / no-date
    if not info.get("has_date", True):
        filename = info["filename"]
        subdir = (info.get("subdir") or "").strip("/")
        if subdir:
            return f"{base}/{subdir}/{filename}", filename
        return f"{base}/{filename}", filename

    if d is None:
        raise ValueError(f"date must be provided for product {product_code}")

    stream = get_stream_number(d)
    yyyy = f"{d.year:04d}"
    mm = f"{d.month:02d}"
    datestr = f"{d.year:04d}{d.month:02d}{d.day:02d}"

    prefix = info["prefix"]
    filename = f"MERRA2_{stream}.{prefix}.{datestr}.nc4"
    # ✅ MERRA2 dated granules are under /YYYY/MM/
    url = f"{base}/{yyyy}/{mm}/{filename}"
    return url, filename


def _looks_like_html_login(resp: requests.Response) -> bool:
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "text/html" in ctype:
        return True
    # sometimes ctype is octet-stream but body is still HTML; cheap check:
    head = (resp.text[:200] if resp.encoding else "")
    if "<html" in head.lower() or "earthdata login" in head.lower():
        return True
    return False


def download_file(session: requests.Session, url: str, out_path: Path, *, force: bool = False):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        print(f"[SKIP] {out_path} exists")
        return

    print(f"[GET ] {url}")
    r = session.get(url, stream=True, timeout=120)
    if not r.ok:
        raise RuntimeError(f"Download failed HTTP {r.status_code}: {url}")

    # Guard: accidental HTML login page
    try:
        if _looks_like_html_login(r):
            raise RuntimeError(f"Got HTML instead of data (auth redirect?) for URL: {url}")
    except Exception:
        # If streaming/encoding prevents text check, ignore.
        pass

    tmp = out_path.with_suffix(out_path.suffix + ".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)
    tmp.replace(out_path)
    print(f"[OK  ] {out_path}")


# -------------------- CMR fallback helpers (for robust URL resolution) --------------------
_CMR_BASE = "https://cmr.earthdata.nasa.gov/search"


def _cmr_get_collection_concept_id(short_name: str, version: str) -> str:
    r = requests.get(
        f"{_CMR_BASE}/collections.json",
        params={"short_name": short_name, "version": version},
        timeout=30,
    )
    r.raise_for_status()
    entries = r.json().get("feed", {}).get("entry", []) or []
    if not entries:
        raise RuntimeError(f"CMR: collection not found for {short_name} v{version}")
    return entries[0]["id"]


def _cmr_pick_direct_data_href(collection_concept_id: str, temporal: str | None = None) -> str:
    params = {"collection_concept_id": collection_concept_id, "page_size": 200}
    if temporal:
        params["temporal"] = temporal

    r = requests.get(
        f"{_CMR_BASE}/granules.json",
        params=params,
        headers={"Accept": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    entries = r.json().get("feed", {}).get("entry", []) or []
    if not entries:
        raise RuntimeError("CMR: granule not found")

    def ok(h: str) -> bool:
        hl = h.lower()
        if not h.startswith("http"):
            return False
        if any(x in hl for x in ["opendap", "thredds", ".html", ".xml"]):
            return False
        return hl.endswith((".nc4", ".nc", ".h5"))

    # prefer links that are "data#" rel if present
    for e in entries:
        for link in e.get("links", []) or []:
            href = link.get("href") or ""
            rel = (link.get("rel") or "").lower()
            if ok(href) and ("data#" in rel or "data" in rel):
                return href

    # fallback: any direct-looking href
    for e in entries:
        for link in e.get("links", []) or []:
            href = link.get("href") or ""
            if ok(href):
                return href

    raise RuntimeError("CMR: no direct data href")


def _resolve_const_ctm_urls() -> list[str]:
    """
    Try multiple plausible layouts for M2C0NXCTM.
    Some servers expose it as:
      .../MERRA2_MONTHLY/M2C0NXCTM.../1980/<file>
    others may differ. We'll try a short list before CMR fallback.
    """
    info = PRODUCT_INFO["M2C0NXCTM"]
    host = info["host"].rstrip("/")
    collection = info["collection"].strip("/")
    filename = info["filename"]
    data_root_primary = info.get("data_root", "MERRA2").strip("/")
    subdir = (info.get("subdir") or "").strip("/")

    candidates: list[str] = []

    def add(root: str, suffix: str):
        candidates.append(f"{host}/data/{root}/{collection}/{suffix}")

    # common layouts
    if subdir:
        add(data_root_primary, f"{subdir}/{filename}")
        add(data_root_primary, f"{subdir}/01/{filename}")
    add(data_root_primary, filename)

    # fallback other root
    other_root = "MERRA2" if data_root_primary == "MERRA2_MONTHLY" else "MERRA2_MONTHLY"
    if subdir:
        add(other_root, f"{subdir}/{filename}")
        add(other_root, f"{subdir}/01/{filename}")
    add(other_root, filename)

    # dedupe
    out: list[str] = []
    seen = set()
    for u in candidates:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def _resolve_const_ctm_url_via_cmr() -> str:
    """
    Resolve M2C0NXCTM direct granule URL via CMR as a robust fallback.
    """
    cid = _cmr_get_collection_concept_id("M2C0NXCTM", "5.12.4")
    # const file exists in 1980 year range for sure
    return _cmr_pick_direct_data_href(cid, temporal="1980-01-01T00:00:00Z,1980-12-31T23:59:59Z")


def download_raw_all(raw_base: Path, d: date, *, force: bool = False):
    """
    Download required raw files for a single day + static file.
    Folder layout matches user's requirement:
      raw_base/M2I1NXASM/...
      raw_base/M2INVASM/...
      ...
      raw_base/M2C0NXCTM/...
    """
    # Session setup: support env creds and/or ~/.netrc
    session = requests.Session()
    session.trust_env = True
    session.headers.update({"User-Agent": "pyhazards-merra2-inspection/1.0"})

    username = os.getenv("EARTHDATA_USERNAME")
    password = os.getenv("EARTHDATA_PASSWORD")
    if username and password:
        session.auth = (username, password)
    else:
        print("[INFO] EARTHDATA_USERNAME/PASSWORD not set. Will try ~/.netrc (machine urs.earthdata.nasa.gov).")

    # ---------- const file ----------
    const_dir = raw_base / "M2C0NXCTM"
    const_dir.mkdir(parents=True, exist_ok=True)
    # We always store with the canonical filename
    _, const_filename = build_file_url("M2C0NXCTM", None)
    const_out = const_dir / const_filename

    # First try a few deterministic URLs
    last_err: Exception | None = None
    for url in _resolve_const_ctm_urls():
        try:
            download_file(session, url, const_out, force=force)
            last_err = None
            break
        except Exception as e:
            last_err = e
            # only continue if it's a 404; otherwise fail fast
            if "HTTP 404" in str(e):
                continue
            raise

    # If still failing, use CMR to resolve a direct href
    if last_err is not None:
        cmr_url = _resolve_const_ctm_url_via_cmr()
        print(f"[CMR] resolved M2C0NXCTM URL: {cmr_url}")
        download_file(session, cmr_url, const_out, force=force)

    # ---------- daily files ----------
    day_products = ["M2I1NXASM", "M2I3NVASM", "M2T1NXFLX", "M2T1NXLND", "M2T1NXRAD"]

    # user wants NV folder named M2INVASM
    nv_folder = raw_base / "M2INVASM"
    nv_folder.mkdir(parents=True, exist_ok=True)

    for prod in day_products:
        url, filename = build_file_url(prod, d)
        if prod == "M2I3NVASM":
            out_dir = nv_folder
        else:
            out_dir = raw_base / prod
        download_file(session, url, out_dir / filename, force=force)


# ---------------------------------------------------------------------
# PART 2) timefix (shared, derived from mergesfc/mergepres timefix sections)
# ---------------------------------------------------------------------
def parse_ymd_from_name(path: Path) -> tuple[int, str]:
    m = re.search(r"(\d{8})", path.name)
    if not m:
        raise ValueError(f"Cannot find YYYYMMDD in filename: {path.name}")
    return int(m.group(1)), m.group(1)


def timefix_one_file(path: Path):
    """
    Keep the same behavior: rewrite time to int32 [0, step, 2*step,...] (minutes),
    and set begin_date/begin_time/units/calendar.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"timefix target missing: {path}")

    ymd_int, ymd_str = parse_ymd_from_name(path)

    with h5py.File(path, "r+") as f:
        if "time" not in f:
            raise RuntimeError(f"{path}: missing 'time' variable for timefix")

        t = f["time"]
        data = np.array(t[...])
        if data.size == 0:
            raise RuntimeError(f"{path}: time length is 0")

        # step from old time
        if data.size > 1:
            step = int(data[1] - data[0])
        else:
            step = 180
        if step <= 0:
            step = 180

        new_time = np.arange(0, step * data.size, step, dtype="int32")
        t[...] = new_time.astype("int32")

        t.attrs["begin_date"] = np.array([ymd_int], dtype="int32")
        t.attrs["begin_time"] = np.array([0], dtype="int32")
        t.attrs["units"] = f"minutes since {ymd_str[0:4]}-{ymd_str[4:6]}-{ymd_str[6:8]} 00:00:00"
        t.attrs["calendar"] = "proleptic_gregorian"


# ---------------------------------------------------------------------
# PART 3) Merge SFC (adapted from mergesfc.py; same merge logic, path parameterized)
# ---------------------------------------------------------------------
def _rename_latlon(ds: xr.Dataset) -> xr.Dataset:
    rename_map = {}
    if "latitude" in ds.coords and "lat" not in ds.coords:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.coords and "lon" not in ds.coords:
        rename_map["longitude"] = "lon"
    if rename_map:
        ds = ds.rename(rename_map)
    return ds


def _ensure_order(ds: xr.Dataset) -> xr.Dataset:
    if "lat" in ds.coords and ds["lat"].size > 1:
        lat = ds["lat"].values
        if np.nanmean(np.diff(lat)) < 0:
            ds = ds.sortby("lat")
    if "lon" in ds.coords and ds["lon"].size > 1:
        lon = ds["lon"].values
        if np.nanmean(np.diff(lon)) < 0:
            ds = ds.sortby("lon")
    return ds


def _coerce_to_shape(ds: xr.Dataset, target_lat=TARGET_LAT, target_lon=TARGET_LON) -> xr.Dataset:
    if "lat" in ds.dims and ds.dims["lat"] != target_lat:
        raise ValueError(f"lat size mismatch: {ds.dims['lat']} != {target_lat}")
    if "lon" in ds.dims and ds.dims["lon"] != target_lon:
        raise ValueError(f"lon size mismatch: {ds.dims['lon']} != {target_lon}")
    return ds


def _select_8_from_24(da: xr.DataArray):
    return da.isel(time=[0, 3, 6, 9, 12, 15, 18, 21])


def _repeat_to_n(da: xr.DataArray, n: int = TARGET_N_FRAMES) -> xr.DataArray:
    if "time" not in da.dims:
        tiles = [da.expand_dims(time=[i]) for i in range(n)]
        return xr.concat(tiles, dim="time")
    T = da.sizes["time"]
    rep = int(np.ceil(n / T))
    out = xr.concat([da] * rep, dim="time").isel(time=slice(0, n))
    return out


def _average_into_n(da: xr.DataArray, n: int = TARGET_N_FRAMES) -> xr.DataArray:
    T = da.sizes["time"]
    splits = np.array_split(np.arange(T), n)
    tiles = [da.isel(time=idx).mean(dim="time") for idx in splits]
    return xr.concat(tiles, dim="time")


def _to_8_frames_data(da: xr.DataArray) -> xr.DataArray:
    if "time" not in da.dims:
        return _repeat_to_n(da, TARGET_N_FRAMES)
    T = da.sizes["time"]
    if T == TARGET_N_FRAMES:
        return da
    if T == 24:
        return _select_8_from_24(da)
    if T == 1:
        return _repeat_to_n(da, TARGET_N_FRAMES)
    if T in (2, 4):
        return _repeat_to_n(da, TARGET_N_FRAMES)
    if T > TARGET_N_FRAMES:
        return _average_into_n(da, TARGET_N_FRAMES)
    return _repeat_to_n(da, TARGET_N_FRAMES)


def _make_time8_from_base_time(base_time: xr.DataArray) -> xr.DataArray:
    t = pd.to_datetime(base_time.values)
    if t.size >= 24:
        idx = [0, 3, 6, 9, 12, 15, 18, 21]
        t8 = t[idx]
    elif t.size == 8:
        t8 = t
    else:
        t0 = pd.Timestamp(t[0]).normalize() if t.size > 0 else pd.Timestamp("2000-01-01")
        t8 = [t0 + pd.Timedelta(hours=h) for h in HOURS8]
    return xr.DataArray(np.array(t8, dtype="datetime64[ns]"), dims=["time"], name="time")


def _align_and_put(ds_out: xr.Dataset, name: str, da: xr.DataArray):
    if "lev" in da.dims:
        da = da.isel(lev=0)

    da = da.transpose(*[d for d in da.dims if d in ["time", "lat", "lon"]])
    da8 = _to_8_frames_data(da)

    if set(["lat", "lon"]).issubset(set(da8.dims)):
        order = ("time", "lat", "lon") if "time" in da8.dims else ("lat", "lon")
        da8 = da8.transpose(*order)
        ds_out[name] = xr.DataArray(da8.values, dims=order, attrs=da.attrs)
    else:
        ds_out[name] = xr.DataArray(da8.values, dims=da8.dims, attrs=da.attrs)


def _load_and_prepare(path: Path) -> xr.Dataset:
    print(f"[DEBUG] open_dataset: {path}")
    last_err = None
    for engine in [None, "netcdf4", "h5netcdf"]:
        try:
            print(f"  try engine={engine}")
            ds = xr.open_dataset(path, engine=engine) if engine else xr.open_dataset(path)
            ds = _rename_latlon(ds)
            ds = _ensure_order(ds)
            ds = _coerce_to_shape(ds, TARGET_LAT, TARGET_LON)
            print(f"  OK with engine={engine}")
            return ds
        except Exception as e:
            print(f"  engine={engine} FAILED:", e)
            last_err = e
    raise last_err


def _squeeze_static_2d(da: xr.DataArray, name: str) -> xr.DataArray:
    if "time" in da.dims:
        print(f"[STATIC] {name}: has time dim={da.sizes['time']}, taking time=0")
        da = da.isel(time=0, drop=True)
    if "lev" in da.dims:
        print(f"[STATIC] {name}: has lev dim={da.sizes['lev']}, taking lev=0")
        da = da.isel(lev=0, drop=True)

    if not {"lat", "lon"}.issubset(da.dims):
        raise ValueError(f"STATIC {name} missing lat/lon dims: dims={da.dims}")

    da2 = da.transpose("lat", "lon")
    if da2.sizes["lat"] != TARGET_LAT or da2.sizes["lon"] != TARGET_LON:
        raise ValueError(f"STATIC {name} size not (361,576): {da2.sizes}")
    return da2


def _merge_global_attrs(*datasets) -> dict:
    merged = {}
    for ds in datasets:
        if ds is None:
            continue
        for k, v in ds.attrs.items():
            if k not in merged:
                merged[k] = v
    return merged


def _raw_paths_for_day(raw_base: Path, d: date) -> dict[str, Path]:
    datestr = yyyymmdd(d)
    stream = get_stream_number(d)

    # note: NV folder is M2INVASM (user requirement)
    return {
        "ASM1":   raw_base / "M2I1NXASM" / f"MERRA2_{stream}.inst1_2d_asm_Nx.{datestr}.nc4",
        "FLX":    raw_base / "M2T1NXFLX" / f"MERRA2_{stream}.tavg1_2d_flx_Nx.{datestr}.nc4",
        "RAD":    raw_base / "M2T1NXRAD" / f"MERRA2_{stream}.tavg1_2d_rad_Nx.{datestr}.nc4",
        "NV":     raw_base / "M2INVASM"  / f"MERRA2_{stream}.inst3_3d_asm_Nv.{datestr}.nc4",
        "LND":    raw_base / "M2T1NXLND" / f"MERRA2_{stream}.tavg1_2d_lnd_Nx.{datestr}.nc4",
        "STATIC": raw_base / "M2C0NXCTM" / "MERRA2_101.const_2d_ctm_Nx.00000000.nc4",
    }


def merge_sfc(raw_base: Path, merged_dir: Path, d: date) -> Path:
    paths = _raw_paths_for_day(raw_base, d)
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing raw file for {k}: {p}")

    ds_asm1   = _load_and_prepare(paths["ASM1"])
    ds_flx    = _load_and_prepare(paths["FLX"])
    ds_rad    = _load_and_prepare(paths["RAD"])
    ds_nv     = _load_and_prepare(paths["NV"])
    ds_lnd    = _load_and_prepare(paths["LND"])
    ds_static = _load_and_prepare(paths["STATIC"])

    keep = [v for v in ASM1_VARS if v in ds_asm1]
    if not keep:
        raise RuntimeError("ASM1 has none of requested variables.")
    asm1_vars_8 = {v: _to_8_frames_data(ds_asm1[v]) for v in keep}

    if "time" in ds_asm1.coords and ds_asm1.sizes.get("time", 0) > 0:
        time8 = _make_time8_from_base_time(ds_asm1["time"])
    else:
        t0 = pd.Timestamp("2000-01-01")
        time8 = xr.DataArray(
            np.array([t0 + pd.Timedelta(hours=h) for h in HOURS8], dtype="datetime64[ns]"),
            dims=["time"], name="time"
        )

    base = xr.DataArray(
        np.empty((TARGET_N_FRAMES, TARGET_LAT, TARGET_LON), dtype="float32"),
        dims=("time", "lat", "lon")
    ).assign_coords(
        time=time8,
        lat=ds_asm1["lat"] if "lat" in ds_asm1.coords else np.arange(TARGET_LAT),
        lon=ds_asm1["lon"] if "lon" in ds_asm1.coords else np.arange(TARGET_LON),
    )

    ds_out = xr.Dataset(coords={"time": base["time"], "lat": base["lat"], "lon": base["lon"]})

    for v, da8 in asm1_vars_8.items():
        ds_out[v] = xr.DataArray(da8.values, dims=("time", "lat", "lon"), attrs=ds_asm1[v].attrs)

    for v in FLX_VARS:
        if v not in ds_flx:
            raise RuntimeError(f"FLX missing var: {v}")
        _align_and_put(ds_out, v, ds_flx[v])

    for v in RAD_VARS:
        if v not in ds_rad:
            raise RuntimeError(f"RAD missing var: {v}")
        _align_and_put(ds_out, v, ds_rad[v])

    for v in NV_VARS:
        if v not in ds_nv:
            print(f"[WARN] NV missing var: {v}; skip")
            continue
        _align_and_put(ds_out, v, ds_nv[v])

    for v in LND_VARS:
        if v not in ds_lnd:
            print(f"[WARN] LND missing var: {v}; skip")
            continue
        _align_and_put(ds_out, v, ds_lnd[v])

    for v in STATIC_VARS:
        if v not in ds_static:
            raise RuntimeError(f"STATIC missing var: {v}")
        da2 = _squeeze_static_2d(ds_static[v], v)
        ds_out[v] = xr.DataArray(da2.values, dims=("lat", "lon"), attrs=ds_static[v].attrs)

    ds_out.attrs = _merge_global_attrs(ds_asm1, ds_flx, ds_rad, ds_nv, ds_lnd, ds_static)

    # NaN fill (same intent as mergesfc.py)
    if NAN_VALS:
        for var in ds_out.data_vars:
            if var in NAN_VALS:
                nan_val = NAN_VALS[var]
                ds_out[var].data[:] = np.nan_to_num(ds_out[var].data, nan=nan_val)

    # time encoding + timefix afterwards (same structure as mergesfc.py)
    t0 = ds_out["time"].values[0]
    ds_out["time"] = ((ds_out["time"].values - t0).astype("timedelta64[m]").astype("int32"))
    ds_out.time.attrs = {"begin_time": 0, "begin_date": int(yyyymmdd(d))}

    encoding = {name: {"zlib": True} for name in ds_out.data_vars}
    encoding["time"] = {"dtype": "int32"}

    merged_dir.mkdir(parents=True, exist_ok=True)
    out_path = merged_dir / PATTERN_SFC_OUT.format(yyyymmdd=yyyymmdd(d))

    try:
        print("[INFO] Writing SFC with engine=h5netcdf...")
        ds_out.to_netcdf(out_path, encoding=encoding, engine="h5netcdf")
    except Exception as e:
        print("[WARN] h5netcdf failed:", e)
        print("[INFO] Falling back to engine=netcdf4")
        ds_out.to_netcdf(out_path, encoding=encoding, engine="netcdf4")

    timefix_one_file(out_path)
    print(f"[OK] Wrote SFC: {out_path}")
    return out_path


# ---------------------------------------------------------------------
# PART 4) Merge PRES (adapted from mergepres.py; same merge logic, path parameterized)
# ---------------------------------------------------------------------
def _fmt_date_any(ts):
    ts_pd = pd.to_datetime(ts)
    if ts_pd.tzinfo is None:
        ts_pd = ts_pd.tz_localize(timezone.utc)
    else:
        ts_pd = ts_pd.tz_convert(timezone.utc)
    return ts_pd.strftime("%Y-%m-%d"), ts_pd.strftime("%H:%M:%S.%f")


def _open_nv_any_engine(path: Path) -> xr.Dataset:
    print(f"[DEBUG] open_dataset: {path}")
    last_err = None
    for engine in [None, "netcdf4", "h5netcdf"]:
        try:
            print(f"  try engine={engine}")
            ds = xr.open_dataset(path, engine=engine) if engine else xr.open_dataset(path)
            print(f"  OK with engine={engine}")
            return ds
        except Exception as e:
            print(f"  engine={engine} FAILED:", e)
            last_err = e
    raise last_err


def _write_any_engine(ds: xr.Dataset, out_path: Path):
    engines = [("h5netcdf", True), ("netcdf4", True), ("scipy", False)]
    last_err = None
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for engine, can_compress in engines:
        try:
            print(f"[INFO] Writing to {out_path} (engine={engine}, compression={'on' if can_compress else 'off'})")
            if can_compress:
                comp = dict(zlib=True, complevel=1, shuffle=True)
                encoding = {v: comp for v in ds.data_vars}
                ds.to_netcdf(out_path, engine=engine, mode="w", encoding=encoding)
            else:
                ds.to_netcdf(out_path, engine=engine, mode="w")
            print(f"[INFO] Write OK with engine={engine}")
            return
        except Exception as e:
            print(f"[WARN] engine={engine} FAILED:", e)
            last_err = e
    raise last_err


def merge_pres(raw_base: Path, merged_dir: Path, d: date) -> Path:
    datestr = yyyymmdd(d)
    stream = get_stream_number(d)
    in_vert = raw_base / "M2INVASM" / f"MERRA2_{stream}.inst3_3d_asm_Nv.{datestr}.nc4"
    if not in_vert.exists():
        raise FileNotFoundError(f"Missing NV raw file for PRES merge: {in_vert}")

    merged_dir.mkdir(parents=True, exist_ok=True)
    out_vert = merged_dir / PATTERN_PRES_OUT.format(yyyymmdd=datestr)

    ds = _open_nv_any_engine(in_vert)

    keep_vars = [v for v in PRES_VARS if v in ds.data_vars]
    if not keep_vars:
        raise ValueError(f"No target vars found. Wanted: {PRES_VARS}")
    ds = ds[keep_vars]

    if "lev" not in ds.coords:
        raise KeyError("Missing 'lev' coordinate in NV file.")

    lev_vals = np.array(ds["lev"].values, dtype=float).tolist()
    missing = [lv for lv in PRES_LEVELS if lv not in lev_vals]
    if missing:
        raise ValueError(f"Missing levels: {missing}\nAvailable lev: {lev_vals}")

    ds = ds.sel(lev=xr.DataArray(PRES_LEVELS, dims="lev"))
    ds = ds.sortby("lev", ascending=False)

    attrs = dict(ds.attrs) if ds.attrs else {}
    if "time" in ds.coords and ds.sizes.get("time", 0) > 0:
        tvals = ds["time"].values
        tmin, tmax = tvals.min(), tvals.max()
        beg_date, beg_time = _fmt_date_any(tmin)
        end_date, end_time = _fmt_date_any(tmax)
        attrs.update({
            "RangeBeginningDate": beg_date,
            "RangeBeginningTime": beg_time,
            "RangeEndingDate": end_date,
            "RangeEndingTime": end_time,
        })
    ds = ds.assign_attrs(attrs)

    _write_any_engine(ds, out_vert)
    ds.close()

    timefix_one_file(out_vert)

    # quick check
    try:
        ds2 = xr.open_dataset(out_vert, engine="h5netcdf")
        print(f"[OK] Wrote PRES: {out_vert}")
        print("vars:", list(ds2.data_vars))
        print("shape:", {k: int(v) for k, v in ds2.sizes.items()})
        ds2.close()
    except Exception as e:
        print("[WARN] PRES written & timefixed, but reopen check failed:", e)

    return out_vert


# ---------------------------------------------------------------------
# PART 5) Inspection (SFC/PRES merged products)
# ---------------------------------------------------------------------
def list_vars(ds: xr.Dataset, max_show: int = 60) -> pd.DataFrame:
    rows = []
    for name, da in ds.data_vars.items():
        rows.append({
            "var": name,
            "dims": str(da.dims),
            "shape": str(tuple(da.shape)),
            "dtype": str(da.dtype),
        })
    df = pd.DataFrame(rows).sort_values("var").reset_index(drop=True)
    return df.head(max_show) if len(df) > max_show else df


def inspect_ds(ds: xr.Dataset, name: str, max_vars: int = 60):
    print(f"\n=== {name} ===")
    print("dims:", dict(ds.sizes))
    print("coords:", list(ds.coords))
    print("n_vars:", len(ds.data_vars))
    display(list_vars(ds, max_show=max_vars))


def summarize_da(da: xr.DataArray) -> pd.Series:
    s = xr.Dataset({
        "min": da.min(skipna=True),
        "max": da.max(skipna=True),
        "mean": da.mean(skipna=True),
        "std": da.std(skipna=True),
    }).compute()
    return pd.Series({k: float(s[k].values) for k in s.data_vars})


def run_inspection(merged_dir: Path, outdir: Path, d: date, var: str = "T2M"):
    sfc_path  = merged_dir / PATTERN_SFC_OUT.format(yyyymmdd=yyyymmdd(d))
    pres_path = merged_dir / PATTERN_PRES_OUT.format(yyyymmdd=yyyymmdd(d))

    ds_sfc  = xr.open_dataset(sfc_path, engine="h5netcdf")
    ds_pres = xr.open_dataset(pres_path, engine="h5netcdf")

    inspect_ds(ds_sfc,  "SFC (one day)")
    inspect_ds(ds_pres, "PRES (one day)")

    # save var tables
    outdir.mkdir(parents=True, exist_ok=True)
    list_vars(ds_sfc).to_csv(outdir / f"sfc_vars_{d.isoformat()}.csv", index=False)
    list_vars(ds_pres).to_csv(outdir / f"pres_vars_{d.isoformat()}.csv", index=False)

    if var not in ds_sfc:
        raise KeyError(f"{var} not found in SFC merged file. See vars csv in {outdir}")

    da = ds_sfc[var]
    print(f"\n{var} dims :", da.dims)
    print(f"{var} shape:", da.shape)
    print(f"\n{var} summary:")
    print(summarize_da(da))

    # plot one timestep
    t = 0
    Z = da.isel(time=t).compute()
    plt.figure()
    plt.contourf(ds_sfc["lon"], ds_sfc["lat"], Z, 100)
    plt.gca().set_aspect("equal")
    plt.title(f"{var} (t={t})")

    out_pdf = outdir / f"{var.lower()}_{d.isoformat()}.pdf"
    plt.savefig(out_pdf)
    print(f"\nSaved: {out_pdf}")

    ds_sfc.close()
    ds_pres.close()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pyhazards.datasets.inspection",
        description="One-shot pipeline: download raw MERRA-2 -> merge SFC+PRES -> inspection.",
    )

    # ✅ positional date
    p.add_argument(
        "date",
        help="Date to run. Accepts YYYYMMDD (e.g., 20251111) or YYYY-MM-DD (e.g., 2025-11-11).",
    )

    # ✅ default outputs (relative to repo root unless absolute)
    p.add_argument("--outdir", default="outputs", help="Output directory (default: outputs under repo root)")

    # ✅ auto infer repo root if omitted
    p.add_argument("--repo-root", default=None, help="Repo root (auto-infer if omitted)")
    p.add_argument("--raw-base", default=None, help="Raw datasets base dir (default: REPO_ROOT)")
    p.add_argument("--merged-dir", default=None, help="Merged dir (default: REPO_ROOT/Prithvi-WxC/data/merra-2)")

    p.add_argument("--skip-download", action="store_true", help="Assume raw files already exist")
    p.add_argument("--skip-merge", action="store_true", help="Assume merged nc already exist")
    p.add_argument("--force-download", action="store_true", help="Force re-download even if file exists")
    p.add_argument("--var", default="T2M", help="Variable to summarize/plot from SFC")

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    s = args.date.strip()
    if len(s) == 8 and s.isdigit():
        d = date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    else:
        d = date.fromisoformat(s)

    repo_root = Path(args.repo_root).expanduser().resolve() if args.repo_root else infer_repo_root()
    raw_base = Path(args.raw_base).expanduser().resolve() if args.raw_base else repo_root
    merged_dir = Path(args.merged_dir).expanduser().resolve() if args.merged_dir else (repo_root / "Prithvi-WxC" / "data" / "merra-2")

    # outdir: relative -> repo_root/outdir; absolute -> keep
    outdir = (repo_root / args.outdir) if not Path(args.outdir).is_absolute() else Path(args.outdir)
    outdir = outdir.expanduser().resolve()

    print("repo_root :", repo_root)
    print("raw_base  :", raw_base)
    print("merged_dir:", merged_dir)
    print("outdir    :", outdir)
    print("date      :", d)

    raw_base.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        print("\n=== STEP 1: Download raw datasets ===")
        download_raw_all(raw_base, d, force=args.force_download)

    if not args.skip_merge:
        print("\n=== STEP 2: Merge SFC ===")
        merge_sfc(raw_base, merged_dir, d)

        print("\n=== STEP 3: Merge PRES ===")
        merge_pres(raw_base, merged_dir, d)

    print("\n=== STEP 4: Inspection ===")
    run_inspection(merged_dir, outdir, d, var=args.var)

    print("\n[DONE] Full pipeline finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
