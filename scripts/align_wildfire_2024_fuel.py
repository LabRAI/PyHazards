from __future__ import annotations

import argparse
from pathlib import Path

from pyhazards.benchmarks.wildfire_benchmark.cache_builder import align_static_fuel_to_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align LANDFIRE fuel to the wildfire 2024 benchmark cache grid.")
    parser.add_argument("--cache_dir", type=str, default="/home/runyang/my-copy/data_cache/wildfire_2024_v1")
    parser.add_argument(
        "--landfire_tif",
        type=str,
        default="/home/runyang/ryang/landfire_fbfm40/LF2024_FBFM13_250_CONUS/Tif/LC24_F13_250.tif",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = align_static_fuel_to_cache(
        cache_root=args.cache_dir,
        landfire_tif=args.landfire_tif,
        overwrite=bool(args.overwrite),
    )
    print(f"[done] aligned fuel written under {Path(args.cache_dir) / 'static'}")
    print(f"[summary] valid_cells={payload.get('valid_cells')} valid_fraction={payload.get('valid_fraction')}")


if __name__ == "__main__":
    main()
