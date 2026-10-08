"""Build a Track-O wildfire occurrence cache (daily weather + FIRMS labels + optional LANDFIRE fuel).

Adapted from scripts/build_wildfire_2024_cache.py and scripts/align_wildfire_2024_fuel.py of
PyHazards PR #33 (runyangxu). The defaults reproduce that PR's 2024 setting: the 14 MERRA-2 surface
variables and the January-September / October / November-December split.

Examples::

    # MERRA-2 surface files written by `python -m pyhazards.datasets.merra2.inspection YYYYMMDD`
    python scripts/build_wildfire_track_o_cache.py --cache-dir data/track_o_2024 \\
        --weather-dir Prithvi-WxC/data/merra-2 --weather-glob 'MERRA2_sfc_2024*.nc' \\
        --firms-dir data/firms_2024 --fuel-raster path/to/LF2024_FBFM13_CONUS.tif

    # add (or redo) the fuel layer of an existing cache
    python scripts/build_wildfire_track_o_cache.py --cache-dir data/track_o_2024 --fuel-only \\
        --fuel-raster path/to/LF2024_FBFM40_CONUS.tif
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyhazards.datasets.wildfire.track_o import DEFAULT_SPLITS_2024, DEFAULT_WEATHER_VARS  # noqa: E402
from pyhazards.datasets.wildfire.track_o_cache import (  # noqa: E402
    DEFAULT_DATE_PATTERN,
    add_fuel_to_cache,
    build_track_o_cache,
)


def _range(text: str) -> tuple[str, str]:
    start, sep, end = text.partition(":")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected START:END (YYYY-MM-DD:YYYY-MM-DD), got {text!r}")
    return start, end


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache-dir", required=True, help="Output directory of the cache.")
    parser.add_argument("--weather-dir", help="Directory of daily weather NetCDF files.")
    parser.add_argument("--weather-glob", default="*.nc*", help="Glob for weather files inside --weather-dir.")
    parser.add_argument(
        "--weather-date-pattern",
        default=DEFAULT_DATE_PATTERN,
        help="Regex whose first group is the file's date (YYYYMMDD or YYYY-MM-DD).",
    )
    parser.add_argument("--weather-vars", default=",".join(DEFAULT_WEATHER_VARS), help="Comma-separated variables.")
    parser.add_argument("--firms-dir", help="Directory of FIRMS CSV files (archive/NRT downloads or one file per day).")
    parser.add_argument("--firms-glob", default="*.csv")
    parser.add_argument(
        "--firms-types",
        default="0",
        help="Comma-separated FIRMS 'type' values to keep (0 = presumed vegetation fire), or 'all'.",
    )
    parser.add_argument("--train", type=_range, default=DEFAULT_SPLITS_2024["train"], help="START:END")
    parser.add_argument("--val", type=_range, default=DEFAULT_SPLITS_2024["val"], help="START:END")
    parser.add_argument("--test", type=_range, default=DEFAULT_SPLITS_2024["test"], help="START:END")
    parser.add_argument("--fuel-raster", help="LANDFIRE fuel-model GeoTIFF (needs pyhazards[geo]).")
    parser.add_argument(
        "--fuel-nodata",
        type=float,
        default=-9999,
        help="Source value left out of the fuel mode (LANDFIRE's -9999 fill); pass nan to use the file's nodata tag.",
    )
    parser.add_argument("--fuel-only", action="store_true", help="Only (re)align the fuel layer of an existing cache.")
    parser.add_argument("--limit-days", type=int, default=0, help="Write only the first N days (quick checks).")
    args = parser.parse_args(argv)
    if args.fuel_nodata != args.fuel_nodata:  # nan: use the raster's own nodata tag
        args.fuel_nodata = None
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.fuel_only:
        if not args.fuel_raster:
            raise SystemExit("--fuel-only needs --fuel-raster")
        info = add_fuel_to_cache(args.cache_dir, args.fuel_raster, src_nodata=args.fuel_nodata)
        print(json.dumps(info, indent=2))
        return 0
    if not args.weather_dir or not args.firms_dir:
        raise SystemExit("--weather-dir and --firms-dir are required (unless --fuel-only)")
    weather_files = sorted(Path(args.weather_dir).glob(args.weather_glob))
    firms_files = sorted(Path(args.firms_dir).glob(args.firms_glob))
    if not weather_files:
        raise SystemExit(f"no weather files match {args.weather_dir}/{args.weather_glob}")
    if not firms_files:
        raise SystemExit(f"no FIRMS files match {args.firms_dir}/{args.firms_glob}")
    firms_types = None if args.firms_types.strip().lower() == "all" else [int(v) for v in args.firms_types.split(",")]
    summary = build_track_o_cache(
        args.cache_dir,
        weather_files,
        firms_files,
        weather_vars=[v.strip() for v in args.weather_vars.split(",") if v.strip()],
        splits={"train": args.train, "val": args.val, "test": args.test},
        weather_date_pattern=args.weather_date_pattern,
        firms_types=firms_types,
        fuel_raster=args.fuel_raster,
        fuel_nodata=args.fuel_nodata,
        limit_days=args.limit_days,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
