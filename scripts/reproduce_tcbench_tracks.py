"""Reproduce TCBench's published Pangu-Weather cyclone errors end to end from the raw forecast fields.

TCBench's official notebook (TCBench_Alpha ``dev/Getting_Started.ipynb``) prints the direct position
error (DPE_GCD, km) and absolute wind / pressure errors of sample rows of its 2023 Pangu-Weather
evaluation. For each printed row this script

1. reads the TempestExtremes variables of the raw Pangu-Weather forecast from TCBench's Hugging Face
   dataset (pinned revision; about 200 MB of range requests per forecast, nothing stored),
2. tracks cyclones with :mod:`pyhazards.forecasts.tempest` and checks the tracks against TCBench's
   released ``unmatched_tracks`` file (byte for byte),
3. matches tracks to IBTrACS storms (HuracanPy's rule, as TCBench) and
4. scores the row with ``protocol="tcbench"`` against NCEI IBTrACS v04r01, downloaded per basin.

Usage::

    python scripts/reproduce_tcbench_tracks.py --cache data/tcbench_repro

Rows reproduce exactly when NCEI's IBTrACS still has the positions TCBench evaluated against (NCEI
revises best tracks; Cheneso's 2023-01-18 00 UTC longitude changed from 55.8 to 55.9 since).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PUBLISHED = [
    # basin, SID, initial time, valid time, DPE_GCD (km), AE_wind (kt), AE_pressure (hPa)
    ("SI", "2023013S08081", "2023-01-15 12:00", "2023-01-18 00:00", 216.013123, 4.770411, 6.5610),
    ("EP", "2023290N12256", "2023-10-19 12:00", "2023-10-20 00:00", 18.454374, 62.780767, 39.9856),
    ("NI", "2023129N08091", "2023-05-07 00:00", "2023-05-12 00:00", 424.145817, 25.624799, 9.5378),
    ("NA", "2023193N37305", "2023-07-16 12:00", "2023-07-20 00:00", 302.565885, 21.254614, 9.4880),
]


def main(argv=None) -> int:
    from pyhazards.datasets.tc.ibtracs import download_ibtracs, read_ibtracs
    from pyhazards.forecasts import (
        download_tcbench_file,
        forecast_track_errors,
        read_tcbench_fields,
        tcbench_ibtracs,
        tcbench_path,
        tempest_forecast_tracks,
        write_stitchnodes_csv,
    )

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", default="data/tcbench_repro", help="directory for IBTrACS and TCBench CSV files")
    args = parser.parse_args(argv)
    cache = Path(args.cache)
    basins = sorted({row[0] for row in PUBLISHED})
    ibtracs = pd.concat(
        [read_ibtracs(download_ibtracs(cache / "ibtracs", subset=b), ["USA_WIND", "USA_PRES", "TRACK_TYPE"]) for b in basins],
        ignore_index=True,
    )
    ibtracs = tcbench_ibtracs(ibtracs, 2023)
    ok = True
    for basin, sid, init, valid, dpe, ae_wind, ae_pres in PUBLISHED:
        fields = read_tcbench_fields("pangu", init)
        table, tracks = tempest_forecast_tracks(fields, ibtracs, init, return_tracks=True)
        released = download_tcbench_file(tcbench_path("pangu", "unmatched", init), cache)
        with tempfile.TemporaryDirectory() as tmp:
            ours = Path(tmp) / "tracks.csv"
            write_stitchnodes_csv(tracks, ours)
            identical = ours.read_text() == released.read_text()
        errors = forecast_track_errors(table, ibtracs, protocol="tcbench")
        row = errors[(errors["SID"] == sid) & (errors["valid_time"] == pd.Timestamp(valid))]
        if row.empty:
            print(f"{sid} {init} -> {valid}: storm not matched / no row")
            ok = False
            continue
        row = row.iloc[0]
        print(
            f"{sid} init {init} valid {valid}: tracks identical to TCBench: {identical}; "
            f"forecast ({row.lat:.2f}, {row.lon:.2f}) vs IBTrACS ({row.ref_lat}, {row.ref_lon}); "
            f"DPE {row.track_error_km:.6f} km (published {dpe}); "
            f"AE wind {abs(row.wind_error_kt):.6f} kt (published {ae_wind}); "
            f"AE pressure {abs(row.pres_error_hpa):.4f} hPa (published {ae_pres})"
        )
        ok &= identical
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
