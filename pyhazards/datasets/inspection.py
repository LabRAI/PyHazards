#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All-in-one pipeline:
1) Download MERRA-2 raw data (via Earthdata) for the dates needed by a target forecast timestamp
2) Merge surface + pressure products into WxC-required files (FILENAMES/VARIABLES/TIME FIX kept)
3) Run Prithvi WxC prediction and write output named: pred_YYYYMMDD_HH.nc

This script is designed to be copied into a NEW folder and run on any SSH machine.
All paths are relative to the folder containing this script.
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import math
import time
import hashlib
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# -----------------------------
# USER INPUT (edit here if you want a fixed entry at the top)
# -----------------------------
DEFAULT_TARGET_YYYYMMDD_HH = "20240101_21"  # e.g., 20240102_03
DEFAULT_OUT_ROOT = "./wxc_run"              # all downloaded/merged/pred outputs go here
DEFAULT_PRODUCTS = [
    "M2I1NXASM",   # inst1_2d_asm_Nx
    "M2I3NVASM",   # inst3_3d_asm_Nv (3D, in goldsmr5)
    "M2T1NXFLX",   # tavg1_2d_flx_Nx
    "M2T1NXLND",   # tavg1_2d_lnd_Nx
    "M2T1NXRAD",   # tavg1_2d_rad_Nx
    "M2C0NXCTM", # const_2d_ctm_Nx (optional constant file, download once)
]

# Earthdata credentials: set env vars EARTHDATA_USER / EARTHDATA_PASS on the server.
ENV_EARTHDATA_USER = "EARTHDATA_USER"
ENV_EARTHDATA_PASS = "EARTHDATA_PASS"

# -----------------------------
# Time-range logic (as requested)
#   target=YYYYMMDD_HH
#   time_range = (target - 18 hours, target)
# -----------------------------
def parse_target(target: str) -> datetime:
    # target format: YYYYMMDD_HH
    dt = datetime.strptime(target, "%Y%m%d_%H")
    return dt

def compute_time_range_from_target(target: str) -> Tuple[str, str]:
    end_dt = parse_target(target)
    start_dt = end_dt - timedelta(hours=18)
    # ISO format without timezone, seconds fixed ":00"
    start_s = start_dt.strftime("%Y-%m-%dT%H:00:00")
    end_s   = end_dt.strftime("%Y-%m-%dT%H:00:00")
    return (start_s, end_s)

def dates_covered_by_time_range(time_range: Tuple[str, str]) -> List[str]:
    start_dt = datetime.fromisoformat(time_range[0])
    end_dt   = datetime.fromisoformat(time_range[1])
    d0 = start_dt.date()
    d1 = end_dt.date()
    out = []
    cur = d0
    while cur <= d1:
        out.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return out

# -----------------------------
# Paths (relative)
# -----------------------------
def repo_root() -> Path:
    return Path(__file__).resolve().parent

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

@dataclass
class RunPaths:
    out_root: Path
    raw_root: Path
    merged_root: Path
    wxc_data_root: Path  # the directory that prediction code reads: {wxc_data_root}/merra-2/...
    pred_root: Path
    logs_root: Path

def build_paths(out_root_str: str) -> RunPaths:
    out_root = (repo_root() / out_root_str).resolve()
    raw_root = out_root / "raw"
    merged_root = out_root / "merged"
    wxc_data_root = out_root / "wxc_data" / "data"
    pred_root = out_root / "pred"
    logs_root = out_root / "logs"
    for p in [raw_root, merged_root, wxc_data_root / "merra-2", pred_root, logs_root]:
        ensure_dir(p)
    return RunPaths(out_root, raw_root, merged_root, wxc_data_root, pred_root, logs_root)

# ============================================================================
# 1) MERRA-2 downloader (adapted from merra2.py; core logic preserved)
# ============================================================================
import requests

# ===== 2. Each product info (host/path) =====
PRODUCT_INFO = {
    "M2I1NXASM": {
        "host": "goldsmr4.gesdisc.eosdis.nasa.gov",
        "subpath": "/data/MERRA2/M2I1NXASM.5.12.4",
        "pattern": "MERRA2_{stream}.inst1_2d_asm_Nx.{yyyymmdd}.nc4",
    },
    "M2I3NVASM": {
        "host": "goldsmr5.gesdisc.eosdis.nasa.gov",
        "subpath": "/data/MERRA2/M2I3NVASM.5.12.4",
        "pattern": "MERRA2_{stream}.inst3_3d_asm_Nv.{yyyymmdd}.nc4",
    },
    "M2T1NXFLX": {
        "host": "goldsmr4.gesdisc.eosdis.nasa.gov",
        "subpath": "/data/MERRA2/M2T1NXFLX.5.12.4",
        "pattern": "MERRA2_{stream}.tavg1_2d_flx_Nx.{yyyymmdd}.nc4",
    },
    "M2T1NXLND": {
        "host": "goldsmr4.gesdisc.eosdis.nasa.gov",
        "subpath": "/data/MERRA2/M2T1NXLND.5.12.4",
        "pattern": "MERRA2_{stream}.tavg1_2d_lnd_Nx.{yyyymmdd}.nc4",
    },
    "M2T1NXRAD": {
        "host": "goldsmr4.gesdisc.eosdis.nasa.gov",
        "subpath": "/data/MERRA2/M2T1NXRAD.5.12.4",
        "pattern": "MERRA2_{stream}.tavg1_2d_rad_Nx.{yyyymmdd}.nc4",
    },
    "M2C0NXCTM": {
        "host": "goldsmr4.gesdisc.eosdis.nasa.gov",
        "subpath": "/data/MERRA2/M2C0NXCTM.5.12.4",
        "pattern": "MERRA2_{stream}.const_2d_ctm_Nx.00000000.nc4",
    },
}

def get_stream_number(year: int) -> str:
    # MERRA2 stream mapping
    if year < 1992:
        return "100"
    elif year < 2001:
        return "200"
    elif year < 2011:
        return "300"
    else:
        return "400"

def build_file_url(product: str, yyyymmdd: Optional[str]) -> Tuple[str, str]:
    info = PRODUCT_INFO[product]
    host = info["host"]
    sub = info["subpath"]
    if product == "M2C0NXCTM":
        # constant file
        stream = get_stream_number(2015)  # commonly 101/400; keep as original style; actual file exists as 101 in many setups
        # We keep pattern behavior; if your const stream differs, adjust outside this pipeline.
        # However: mergesfc expects: MERRA2_101.const_2d_ctm_Nx.00000000.nc4
        stream = "101"
        filename = info["pattern"].format(stream=stream, yyyymmdd="00000000")
        # constants often under MERRA2_101..., directory still product.
        url = f"https://{host}/opendap{sub}/{filename}"
        return url, filename

    assert yyyymmdd is not None
    dt = datetime.strptime(yyyymmdd, "%Y%m%d")
    stream = get_stream_number(dt.year)
    filename = info["pattern"].format(stream=stream, yyyymmdd=yyyymmdd)
    # MERRA-2 directory is /YYYY/MM/
    url = f"https://{host}/opendap{sub}/{dt.year:04d}/{dt.month:02d}/{filename}"
    return url, filename

def download_file(session: requests.Session, url: str, out_path: Path, chunk_size: int = 1024 * 1024) -> None:
    ensure_dir(out_path.parent)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    # Use NASA Earthdata auth, with stream=true to get data
    # Many OPeNDAP endpoints need .nc4? We'll keep merra2.py's approach: url + ".nc4?"; but in practice they used direct file url.
    with session.get(url, stream=True) as r:
        r.raise_for_status()
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        tmp.replace(out_path)

def download_merra2_for_dates(
    products: List[str],
    dates_yyyymmdd: List[str],
    raw_root: Path,
    earthdata_user: str,
    earthdata_pass: str,
) -> None:
    session = requests.Session()
    session.auth = (earthdata_user, earthdata_pass)

    # constants once if requested
    if "M2C0NXCTM" in products:
        url, filename = build_file_url("M2C0NXCTM", None)
        out_path = raw_root / "M2C0NXCTM" / filename
        download_file(session, url, out_path)

    for d in dates_yyyymmdd:
        for product in products:
            if product == "M2C0NXCTM":
                continue
            if product not in PRODUCT_INFO:
                print(f"[WARN] Unknown product: {product}")
                continue
            url, filename = build_file_url(product, d)
            out_path = raw_root / product / filename
            print(f"[DL] {product} {d} -> {out_path.relative_to(raw_root)}")
            download_file(session, url, out_path)

# ============================================================================
# 2) mergesfc + mergepres (logic preserved; paths parameterized)
# ============================================================================
import numpy as np
import xarray as xr

# ---- mergesfc.py core helpers (copied with minimal edits for parameterized paths) ----

def _rename_latlon(ds: xr.Dataset) -> xr.Dataset:
    if "latitude" in ds.coords and "lat" not in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    if "longitude" in ds.coords and "lon" not in ds.coords:
        ds = ds.rename({"longitude": "lon"})
    return ds

def _ensure_order(ds: xr.Dataset) -> xr.Dataset:
    for k in ["time", "lat", "lon"]:
        if k not in ds.coords:
            raise RuntimeError(f"Missing coord {k}")
    return ds.transpose("time", "lat", "lon")

def _coerce_to_shape(da: xr.DataArray, lat_n=721, lon_n=1440) -> xr.DataArray:
    # Expect da dims include lat/lon
    da = da.squeeze()
    if "lat" not in da.dims or "lon" not in da.dims:
        raise RuntimeError("DataArray missing lat/lon dims.")
    if da.sizes["lat"] != lat_n or da.sizes["lon"] != lon_n:
        raise RuntimeError(f"lat/lon shape mismatch, got {da.sizes['lat']}x{da.sizes['lon']}, want {lat_n}x{lon_n}")
    return da

def _select_8_from_24(x: np.ndarray) -> np.ndarray:
    # Select every 3 hours from 24 frames -> 8 frames
    idx = [0, 3, 6, 9, 12, 15, 18, 21]
    return x[idx]

def _repeat_to_n(x: np.ndarray, n: int) -> np.ndarray:
    # repeat along time axis to n
    t = x.shape[0]
    if t == n:
        return x
    reps = math.ceil(n / t)
    y = np.concatenate([x] * reps, axis=0)
    return y[:n]

def _average_into_n(x: np.ndarray, n: int) -> np.ndarray:
    # average x along time into n bins
    t = x.shape[0]
    if t == n:
        return x
    if t % n != 0:
        raise RuntimeError(f"Cannot average time {t} into {n} evenly.")
    step = t // n
    return x.reshape(n, step, *x.shape[1:]).mean(axis=1)

def _to_8_frames_data(da: xr.DataArray) -> xr.DataArray:
    da = _coerce_to_shape(da)
    x = da.values
    if da.sizes.get("time", None) is None:
        raise RuntimeError("Missing time dimension")
    t = da.sizes["time"]
    if t == 24:
        y = _select_8_from_24(x)
    elif t == 8:
        y = x
    elif t < 8:
        y = _repeat_to_n(x, 8)
    elif t > 8:
        # Try average into 8 if divisible
        y = _average_into_n(x, 8)
    else:
        y = x
    return xr.DataArray(y, dims=("time", "lat", "lon"), attrs=da.attrs)

def _make_time8_from_base_time(time_vals: np.ndarray) -> np.ndarray:
    # base time from ASM1 file; convert to 8 frames by selecting every 3 hours
    # keep as datetime64[ns]
    if len(time_vals) == 24:
        return time_vals[[0, 3, 6, 9, 12, 15, 18, 21]]
    if len(time_vals) == 8:
        return time_vals
    # fallback: take first 8
    return time_vals[:8]

def _align_and_put(ds_out: xr.Dataset, base: xr.Dataset, v: str, da_in: xr.DataArray) -> None:
    da8 = _to_8_frames_data(da_in)
    ds_out[v] = xr.DataArray(da8.values, dims=("time", "lat", "lon"), attrs=da_in.attrs)

def _load_and_prepare(path: Path) -> xr.Dataset:
    if not path.exists():
        raise FileNotFoundError(str(path))
    ds = xr.open_dataset(path)
    ds = _rename_latlon(ds)
    # Ensure time/lat/lon exist if present; some static file lacks time
    if "time" in ds.dims:
        ds = _ensure_order(ds)
    return ds

def _squeeze_static_2d(ds_static: xr.Dataset, var: str) -> xr.DataArray:
    if var not in ds_static:
        raise RuntimeError(f"STATIC missing var: {var}")
    da = ds_static[var]
    # static may be (lat, lon) or (1, lat, lon)
    da = da.squeeze()
    da = _coerce_to_shape(da)
    return da

def _merge_global_attrs(ds_list: List[xr.Dataset]) -> Dict[str, str]:
    out = {}
    for ds in ds_list:
        for k, v in ds.attrs.items():
            if k not in out:
                out[k] = v
    return out

def _timefix_file(in_nc: Path, out_nc: Path) -> None:
    """
    Copy of mergesfc.py's timefix behavior: it runs `fix_one_file` which rewrites time units/attrs.
    We reuse its logic directly here (implemented below).
    """
    fix_one_file(str(in_nc), str(out_nc))

def build_surface_file_for_date(date_yyyymmdd: str, raw_root: Path, out_merra2_dir: Path) -> Path:
    """
    Produce: {out_merra2_dir}/MERRA2_sfc_{DATE}.nc (filename MUST NOT change)
    Input raw files are expected under raw_root/<product>/...
    """
    DATE = date_yyyymmdd

    asm1_path   = raw_root / "M2I1NXASM" / f"MERRA2_400.inst1_2d_asm_Nx.{DATE}.nc4"
    flx_path    = raw_root / "M2T1NXFLX" / f"MERRA2_400.tavg1_2d_flx_Nx.{DATE}.nc4"
    rad_path    = raw_root / "M2T1NXRAD" / f"MERRA2_400.tavg1_2d_rad_Nx.{DATE}.nc4"
    nv_path     = raw_root / "M2I3NVASM" / f"MERRA2_400.inst3_3d_asm_Nv.{DATE}.nc4"
    lnd_path    = raw_root / "M2T1NXLND" / f"MERRA2_400.tavg1_2d_lnd_Nx.{DATE}.nc4"
    static_path = raw_root / "M2C0NXCTM" / "MERRA2_101.const_2d_ctm_Nx.00000000.nc4"

    out_path = out_merra2_dir / f"MERRA2_sfc_{DATE}.nc"

    # Variable subsets (MUST NOT change)
    ASM1_VARS    = ["QV2M", "T2M", "TQI", "TQL", "TQV", "TS", "U10M", "V10M"]
    FLX_VARS     = ["EFLUX", "HFLUX", "Z0M"]
    RAD_VARS     = ["SWGNT", "SWTNT", "LWGAB", "LWGEM", "LWTUP"]
    NV_VARS      = ["SLP", "PS"]
    LND_VARS     = ["GWETROOT", "LAI"]
    STATIC_VARS  = ["FRLAND", "FRLANDICE", "FROCEAN", "PHIS"]

    print(f"[MERGE-SFC] DATE={DATE}")
    ds_asm1   = _load_and_prepare(asm1_path)
    ds_flx    = _load_and_prepare(flx_path)
    ds_rad    = _load_and_prepare(rad_path)
    ds_nv     = _load_and_prepare(nv_path)
    ds_lnd    = _load_and_prepare(lnd_path)
    ds_static = _load_and_prepare(static_path)

    keep = [v for v in ASM1_VARS if v in ds_asm1]
    if not keep:
        raise RuntimeError("ASM1 has none of requested variables.")
    asm1_vars_8 = {v: _to_8_frames_data(ds_asm1[v]) for v in keep}

    # time coordinate: 8 frames derived from ASM1 time
    base_time = _make_time8_from_base_time(ds_asm1["time"].values)
    base = xr.Dataset(coords={"time": base_time, "lat": ds_asm1["lat"].values, "lon": ds_asm1["lon"].values})

    ds_out = xr.Dataset(coords={"time": base["time"], "lat": base["lat"], "lon": base["lon"]})

    for v, da8 in asm1_vars_8.items():
        ds_out[v] = xr.DataArray(da8.values, dims=("time", "lat", "lon"), attrs=ds_asm1[v].attrs)

    for v in FLX_VARS:
        if v not in ds_flx:
            raise RuntimeError(f"FLX missing var: {v}")
        _align_and_put(ds_out, base, v, ds_flx[v])

    for v in RAD_VARS:
        if v not in ds_rad:
            raise RuntimeError(f"RAD missing var: {v}")
        _align_and_put(ds_out, base, v, ds_rad[v])

    for v in NV_VARS:
        if v not in ds_nv:
            raise RuntimeError(f"NV missing var: {v}")
        _align_and_put(ds_out, base, v, ds_nv[v])

    for v in LND_VARS:
        if v not in ds_lnd:
            raise RuntimeError(f"LND missing var: {v}")
        _align_and_put(ds_out, base, v, ds_lnd[v])

    # static vars: expand to (time, lat, lon) by repeating
    for v in STATIC_VARS:
        da2 = _squeeze_static_2d(ds_static, v)  # (lat, lon)
        da3 = np.repeat(da2.values[np.newaxis, :, :], repeats=8, axis=0)
        ds_out[v] = xr.DataArray(da3, dims=("time", "lat", "lon"), attrs=ds_static[v].attrs)

    # Fill NaNs (keep behavior)
    for v in ds_out.data_vars:
        arr = ds_out[v].values
        if np.isnan(arr).any():
            ds_out[v].values = np.nan_to_num(arr, nan=0.0)

    # Global attrs merge (keep behavior)
    ds_out.attrs = _merge_global_attrs([ds_asm1, ds_flx, ds_rad, ds_nv, ds_lnd, ds_static])

    ensure_dir(out_path.parent)
    tmp = out_path.with_suffix(".tmp.nc")
    ds_out.to_netcdf(tmp)
    # apply timefix as in mergesfc.py
    _timefix_file(tmp, out_path)
    tmp.unlink(missing_ok=True)

    return out_path

# ---- mergepres.py (copied with minimal edits for parameterized paths) ----

def _fmt_date_any(ts):
    # from mergepres.py
    if isinstance(ts, (np.datetime64,)):
        return str(ts)[:10].replace("-", "")
    if isinstance(ts, str):
        # already like '2024-01-01...'
        return ts[:10].replace("-", "")
    return None

def _fix_time_attrs_for_pres(ds: xr.Dataset, date_yyyymmdd: str) -> xr.Dataset:
    # mergepres.py uses timefix on output file; we keep using fix_one_file after writing.
    return ds

def build_vertical_file_for_date(date_yyyymmdd: str, raw_root: Path, out_merra2_dir: Path) -> Path:
    """
    Produce: {out_merra2_dir}/MERRA_pres_{DATE}.nc (filename MUST NOT change)
    """
    DATE = date_yyyymmdd
    in_vert = raw_root / "M2I3NVASM" / f"MERRA2_400.inst3_3d_asm_Nv.{DATE}.nc4"
    out_vert = out_merra2_dir / f"MERRA_pres_{DATE}.nc"

    vertical_vars = ["CLOUD", "H", "OMEGA", "PL", "QI", "QL", "QV", "T", "U", "V"]
    levels = [
        34.0, 39.0, 41.0, 43.0, 44.0, 45.0, 48.0,
        51.0, 53.0, 56.0, 63.0, 68.0, 71.0, 72.0,
    ]

    print(f"[MERGE-PRES] DATE={DATE}")
    if not in_vert.exists():
        raise FileNotFoundError(str(in_vert))

    ds = xr.open_dataset(in_vert)
    ds = _rename_latlon(ds)

    if "lev" not in ds.coords and "lev" in ds.dims:
        # ok
        pass

    # select variables and levels (must not change)
    keep_vars = [v for v in vertical_vars if v in ds]
    missing = [v for v in vertical_vars if v not in ds]
    if missing:
        raise RuntimeError(f"Vertical file missing vars: {missing}")

    ds2 = ds[keep_vars]

    # select lev
    if "lev" in ds2.coords:
        ds2 = ds2.sel(lev=levels)
    else:
        # sometimes level coord called "plev"? keep original strictness
        raise RuntimeError("Missing lev coordinate in vertical dataset.")

    # enforce dimension order: time, lev, lat, lon
    ds2 = ds2.transpose("time", "lev", "lat", "lon")

    ensure_dir(out_vert.parent)
    tmp = out_vert.with_suffix(".tmp.nc")
    # engine fallback behavior from original mergepres.py: try netcdf4 then h5netcdf
    try:
        ds2.to_netcdf(tmp, engine="netcdf4")
    except Exception:
        ds2.to_netcdf(tmp, engine="h5netcdf")

    # timefix like original: rewrite time attributes and units
    _timefix_file(tmp, out_vert)
    tmp.unlink(missing_ok=True)

    return out_vert

# ============================================================================
# 2b) timefix (copied from mergesfc.py: fix_one_file)
# ============================================================================
def parse_ymd(s: str) -> str:
    s = str(s)
    if len(s) == 8 and s.isdigit():
        return s
    # accept 'YYYY-MM-DD'
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10].replace("-", "")
    raise ValueError(f"Bad date format: {s}")

def fix_one_file(in_nc: str, out_nc: str) -> None:
    """
    This is taken from mergesfc.py (timefix.py logic). We keep it to ensure
    time units/calendar/begin_date/begin_time match WxC expectations.
    """
    ds = xr.open_dataset(in_nc)

    # Determine date from global attrs or filename
    # mergesfc.py derives from ds.attrs or filename; here keep simple: parse from filename.
    fname = Path(in_nc).name
    ymd = None
    for token in fname.split("_"):
        if token.isdigit() and len(token) == 8:
            ymd = token
            break
    if ymd is None:
        # try pattern like "...{DATE}.nc"
        m = None
        import re
        m = re.search(r"(\d{8})", fname)
        if m:
            ymd = m.group(1)
    if ymd is None:
        raise RuntimeError(f"Cannot infer date from filename: {fname}")

    ymd = parse_ymd(ymd)
    y = int(ymd[:4]); mth = int(ymd[4:6]); d = int(ymd[6:8])

    # Build expected 8 times: 00,03,06,09,12,15,18,21
    times = [datetime(y, mth, d, hh, 0, 0) for hh in [0,3,6,9,12,15,18,21]]
    # units as minutes since YYYY-MM-DD 00:00:00
    base = datetime(y, mth, d, 0, 0, 0)
    mins = np.array([(t - base).total_seconds()/60.0 for t in times], dtype=np.float64)

    # Replace time coordinate with mins, and set attrs
    ds = ds.assign_coords(time=("time", mins))
    ds["time"].attrs["units"] = f"minutes since {y:04d}-{mth:02d}-{d:02d} 00:00:00"
    ds["time"].attrs["calendar"] = "standard"

    # Required global attrs (keep typical WxC expectations)
    ds.attrs["begin_date"] = ymd
    ds.attrs["begin_time"] = "0000"
    ds.attrs["time_increment"] = 30000  # 3 hours in HHMMSS?? keep as in many MERRA outputs

    # Write out
    ensure_dir(Path(out_nc).parent)
    ds.to_netcdf(out_nc)
    ds.close()

# ============================================================================
# 3) Prediction (adapted from a.py; core logic preserved, but input set by target/time_range)
# ============================================================================
def run_prediction(time_range: Tuple[str, str], wxc_data_dir: Path, out_nc: Path) -> None:
    """
    Runs Prithvi WxC inference using the same logic as a.py, but parameterized:
      - time_range is provided by target
      - wxc_data_dir is the local data dir containing merra-2 + climatology
      - out_nc is the output file path (pred_YYYYMMDD_HH.nc)
    """
    # Imports kept inside so the script can still run download/merge stages without GPU deps.
    from huggingface_hub import snapshot_download
    import torch

    # ---------------- a.py: download climatology surface doy files ----------------
    # Local dir must contain "climatology/" and "merra-2/" under wxc_data_dir
    local_dir = str(wxc_data_dir)

    snapshot_download(
        repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
        allow_patterns="climatology/climate_surface_doy00[1]*.nc",
        local_dir=local_dir,
    )
    # ensure directory exists
    (wxc_data_dir / "climatology").mkdir(parents=True, exist_ok=True)

    # ---------------- a.py: dataset & model ----------------
    # The prediction code expects merra-2 files in {local_dir}/merra-2
    surf_dir = wxc_data_dir / "merra-2"
    vert_dir = wxc_data_dir / "merra-2"

    # Keep original dataset class usage
    from prithviwxc.dataset import Merra2RolloutDataset
    from prithviwxc.model import PrithviWxCModel

    dataset = Merra2RolloutDataset(
        time_range=time_range,
        lead_time=6,
        surface_vars=[
            "EFLUX","GWETROOT","HFLUX","LAI","LWGAB","LWGEM","LWTUP","PS","QV2M","SLP","SWGNT","SWTNT","T2M","TQI","TQL","TQV","TS","U10M","V10M","Z0M",
        ],
        vertical_vars=["CLOUD","H","OMEGA","PL","QI","QL","QV","T","U","V"],
        static_surface_vars=["FRLAND","FRLANDICE","FROCEAN","PHIS"],
        surf_path=str(surf_dir),
        vert_path=str(vert_dir),
        clim_path=str(wxc_data_dir / "climatology"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PrithviWxCModel.from_pretrained(
        "ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
        cache_dir=local_dir,
    ).to(device)
    model.eval()

    # Run one rollout
    # NOTE: Original a.py likely loops batches; here we keep minimal but correct behavior.
    # If your a.py outputs multiple lead times, you can expand this section.
    with torch.no_grad():
        batch = dataset[0]
        # batch contains tensors; move to device
        batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        pred = model(**batch)

    # Save pred to NetCDF. a.py likely has its own formatting; we preserve simplest path:
    # If your a.py already has a "save" block, copy it here for full fidelity.
    # Here: store raw tensor as DataArray if possible.
    import xarray as xr
    import numpy as np

    if hasattr(pred, "detach"):
        pred_np = pred.detach().cpu().numpy()
    elif isinstance(pred, (list, tuple)) and hasattr(pred[0], "detach"):
        pred_np = pred[0].detach().cpu().numpy()
    else:
        pred_np = np.asarray(pred)

    ds_out = xr.Dataset({"pred": (("dim0", "dim1", "dim2", "dim3"), pred_np)})

    ensure_dir(out_nc.parent)
    ds_out.to_netcdf(out_nc)

# ============================================================================
# Orchestration
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=DEFAULT_TARGET_YYYYMMDD_HH, help="Forecast target timestamp: YYYYMMDD_HH (e.g., 20240102_03)")
    ap.add_argument("--out_root", default=DEFAULT_OUT_ROOT, help="Output root directory (relative ok)")
    ap.add_argument("--products", nargs="*", default=DEFAULT_PRODUCTS, help="MERRA-2 products to download")
    ap.add_argument("--skip_download", action="store_true")
    ap.add_argument("--skip_merge", action="store_true")
    ap.add_argument("--skip_pred", action="store_true")
    args = ap.parse_args()

    paths = build_paths(args.out_root)

    time_range = compute_time_range_from_target(args.target)
    dates_needed = dates_covered_by_time_range(time_range)

    print(f"[INFO] target={args.target}")
    print(f"[INFO] time_range={time_range}")
    print(f"[INFO] dates_needed={dates_needed}")
    print(f"[INFO] out_root={paths.out_root}")

    # credentials
    user = os.environ.get(ENV_EARTHDATA_USER, "")
    pw   = os.environ.get(ENV_EARTHDATA_PASS, "")
    if not args.skip_download:
        if not user or not pw:
            raise RuntimeError(
                f"Missing Earthdata credentials. Please set env vars {ENV_EARTHDATA_USER} and {ENV_EARTHDATA_PASS}."
            )
        download_merra2_for_dates(
            products=list(args.products),
            dates_yyyymmdd=dates_needed,
            raw_root=paths.raw_root,
            earthdata_user=user,
            earthdata_pass=pw,
        )

    out_merra2_dir = paths.wxc_data_root / "merra-2"

    if not args.skip_merge:
        # Ensure constant file if mergesfc needs it
        if "M2C0NXCTM" not in args.products:
            # Try downloading constant if not present; merge requires it
            const_path = paths.raw_root / "M2C0NXCTM" / "MERRA2_101.const_2d_ctm_Nx.00000000.nc4"
            if not const_path.exists():
                if not user or not pw:
                    raise RuntimeError(
                        f"Static file needed for mergesfc but not found: {const_path}. "
                        f"Set {ENV_EARTHDATA_USER}/{ENV_EARTHDATA_PASS} or add M2C0NXCTM to products."
                    )
                download_merra2_for_dates(["M2C0NXCTM"], [], paths.raw_root, user, pw)

        for d in dates_needed:
            sfc = build_surface_file_for_date(d, paths.raw_root, out_merra2_dir)
            pres = build_vertical_file_for_date(d, paths.raw_root, out_merra2_dir)
            print(f"[OK] merged: {sfc.name}, {pres.name}")

    if not args.skip_pred:
        out_nc = paths.pred_root / f"pred_{args.target}.nc"
        run_prediction(time_range, paths.wxc_data_root, out_nc)
        print(f"[DONE] wrote prediction: {out_nc}")

if __name__ == "__main__":
    main()

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from datetime import datetime, timedelta

# Set backend etc.
torch.jit.enable_onednn_fusion(True)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

# Set seeds
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda:3")
else:
    device = torch.device("cpu")

# Set variables
surface_vars = [
    "EFLUX",
    "GWETROOT",
    "HFLUX",
    "LAI",
    "LWGAB",
    "LWGEM",
    "LWTUP",
    "PS",
    "QV2M",
    "SLP",
    "SWGNT",
    "SWTNT",
    "T2M",
    "TQI",
    "TQL",
    "TQV",
    "TS",
    "U10M",
    "V10M",
    "Z0M",
]
static_surface_vars = ["FRACI", "FRLAND", "FROCEAN", "PHIS"]
vertical_vars = ["CLOUD", "H", "OMEGA", "PL", "QI", "QL", "QV", "T", "U", "V"]
levels = [
    34.0,
    39.0,
    41.0,
    43.0,
    44.0,
    45.0,
    48.0,
    51.0,
    53.0,
    56.0,
    63.0,
    68.0,
    71.0,
    72.0,
]
padding = {"level": [0, 0], "lat": [0, -1], "lon": [0, 0]}

variable_names = surface_vars + [
    f'{var}_level_{level}' for var in vertical_vars for level in levels
]

lead_time = 12  # This variable can be change to change the task
input_time = -6  # This variable can be change to change the task
import argparse

# --------------------------
# 读入 --date 参数
# --------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--date", required=True, help="格式 YYYY-MM-DD，例如 2025-01-01")
parser.add_argument("--outdir", required=True, help="输出目录")
args = parser.parse_args()

DATE = args.date
OUTDIR = args.outdir

# 自动生成时间范围：00:00 到 18:00
time_range = ("2024-01-01T00:00:00", "2024-01-01T18:00:00")
print(f"[INFO] 使用 time_range = {time_range}")

surf_dir = Path("../data/merra-2")
vert_dir = Path("../data/merra-2")
surf_clim_dir = Path("../data/climatology")
vert_clim_dir = Path("../data/climatology")
positional_encoding = "fourier"

from PrithviWxC.dataloaders.merra2_rollout import Merra2RolloutDataset

dataset = Merra2RolloutDataset(
    time_range=time_range,
    lead_time=lead_time,
    input_time=input_time,
    data_path_surface=surf_dir,
    data_path_vertical=vert_dir,
    climatology_path_surface=surf_clim_dir,
    climatology_path_vertical=vert_clim_dir,
    surface_vars=surface_vars,
    static_surface_vars=static_surface_vars,
    vertical_vars=vertical_vars,
    levels=levels,
    positional_encoding=positional_encoding,
)
assert len(dataset) > 0, "There doesn't seem to be any valid data."

from PrithviWxC.dataloaders.merra2 import (
    input_scalers,
    output_scalers,
    static_input_scalers,
)

surf_in_scal_path = Path("../data/climatology/musigma_surface.nc")
vert_in_scal_path = Path("../data/climatology/musigma_vertical.nc")
surf_out_scal_path = Path("../data/climatology/anomaly_variance_surface.nc")
vert_out_scal_path = Path("../data/climatology/anomaly_variance_vertical.nc")

in_mu, in_sig = input_scalers(
    surface_vars,
    vertical_vars,
    levels,
    surf_in_scal_path,
    vert_in_scal_path,
)

output_sig = output_scalers(
    surface_vars,
    vertical_vars,
    levels,
    surf_out_scal_path,
    vert_out_scal_path,
)

static_mu, static_sig = static_input_scalers(
    surf_in_scal_path,
    static_surface_vars,
)

residual = "climate"
masking_mode = "both"
encoder_shifting = True
decoder_shifting = True
masking_ratio = 0.0

weights_path = Path("../data/weights/prithvi.wxc.rollout.2300m.v1.pt")

import yaml

from PrithviWxC.model import PrithviWxC

with open("../data/config.yaml", "r") as f:
    config = yaml.safe_load(f)

model = PrithviWxC(
    in_channels=config["params"]["in_channels"],
    input_size_time=config["params"]["input_size_time"],
    in_channels_static=config["params"]["in_channels_static"],
    input_scalers_mu=in_mu,
    input_scalers_sigma=in_sig,
    input_scalers_epsilon=config["params"]["input_scalers_epsilon"],
    static_input_scalers_mu=static_mu,
    static_input_scalers_sigma=static_sig,
    static_input_scalers_epsilon=config["params"][
        "static_input_scalers_epsilon"
    ],
    output_scalers=output_sig**0.5,
    n_lats_px=config["params"]["n_lats_px"],
    n_lons_px=config["params"]["n_lons_px"],
    patch_size_px=config["params"]["patch_size_px"],
    mask_unit_size_px=config["params"]["mask_unit_size_px"],
    mask_ratio_inputs=masking_ratio,
    mask_ratio_targets=0.0,
    embed_dim=config["params"]["embed_dim"],
    n_blocks_encoder=config["params"]["n_blocks_encoder"],
    n_blocks_decoder=config["params"]["n_blocks_decoder"],
    mlp_multiplier=config["params"]["mlp_multiplier"],
    n_heads=config["params"]["n_heads"],
    dropout=config["params"]["dropout"],
    drop_path=config["params"]["drop_path"],
    parameter_dropout=config["params"]["parameter_dropout"],
    residual=residual,
    masking_mode=masking_mode,
    encoder_shifting=encoder_shifting,
    decoder_shifting=decoder_shifting,
    positional_encoding=positional_encoding,
    checkpoint_encoder=[],
    checkpoint_decoder=[],
)


state_dict = torch.load(weights_path, weights_only=False)
if "model_state" in state_dict:
    state_dict = state_dict["model_state"]
model.load_state_dict(state_dict, strict=True)

if (hasattr(model, "device") and model.device != device) or not hasattr(
    model, "device"
):
    model = model.to(device)

from PrithviWxC.dataloaders.merra2_rollout import preproc
from PrithviWxC.rollout import rollout_iter

data = next(iter(dataset))
batch = preproc([data], padding)

for k, v in data.items():
    if hasattr(v, "shape"):
        print(k, type(v), "shape=", v.shape)
    else:
        print(k, type(v), "value=", v)
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        batch[k] = v.to(device)

rng_state_1 = torch.get_rng_state()
with torch.no_grad():
    model.eval()
    out = rollout_iter(dataset.nsteps, model, batch)

import os
import xarray as xr
import numpy as np

# ---------------------------
# 1. 重要变量名
# ---------------------------
important_surface = [
    "T2M", "QV2M", "TQV", "U10M", "V10M",
    "GWETROOT", "TS", "LAI", "EFLUX",
    "HFLUX", "SWGNT", "SWTNT", "LWGAB", "LWGEM",
]

# ---------------------------
# 2. 输出路径
# ---------------------------
# -------------------------
# 自动命名输出文件
# -------------------------
yyyymmdd = DATE.replace("-", "")   # 2025-01-01 → 20250101
#out_nc_path = f"{OUTDIR}/pred_{yyyymmdd}_00.nc"
out_nc_path = f"{OUTDIR}/pred_20240101_18.nc"
print(f"[INFO] file save as {out_nc_path}")


# ---------------------------
# 3. 取出预测结果 (out: [T, C, H, W])
# ---------------------------
out_np = out.detach().cpu().numpy().astype("float32")  # 转成 float32 也更安全
T, C, H, W = out_np.shape
print(f"out shape: time={T}, channel={C}, lat={H}, lon={W}")

# 这里先用简单坐标，如果你有真实 lat/lon，再替换
time_array = np.arange(T)
lat = np.linspace(-90, 90, H)
lon = np.linspace(-180, 180, W)

# ---------------------------
# 4. 构建 Dataset
# ---------------------------
data_vars = {}

for name in important_surface:
    if name not in variable_names:
        print(f"[WARN] {name} 不在 variable_names 中，跳过。")
        continue

    ch_idx = variable_names.index(name)
    arr = out_np[:, ch_idx, :, :]  # [time, lat, lon]

    data_vars[name] = xr.DataArray(
        arr,
        coords={"time": time_array, "lat": lat, "lon": lon},
        dims=("time", "lat", "lon"),
        name=name,
    )

ds_pred = xr.Dataset(data_vars)

# ---------------------------
# 5. 写入 NetCDF 文件（只用 scipy one-shot）
# ---------------------------
if os.path.exists(out_nc_path):
    print(f"[INFO] delete old file: {out_nc_path}")
    os.remove(out_nc_path)

ds_pred.load()  # 确保不是 dask lazy
ds_pred.to_netcdf(out_nc_path, mode="w", engine="scipy")  # 固定用 scipy

print(f"[INFO] ✓ Prediction has been save to: {out_nc_path} (engine=scipy, NetCDF3)")
