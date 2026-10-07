from __future__ import annotations

import argparse

import numpy as np

from .reader import read_wrf_sfire_fire_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pyhazards.datasets.wrf_sfire.inspection",
        description="Summarise the fire grid of WRF-SFIRE wrfout files (burned area and arrival times).",
    )
    parser.add_argument("--path", nargs="+", default=None, help="wrfout file(s) or glob pattern(s) of one domain.")
    parser.add_argument("--max-frames", type=int, default=10, help="Number of output frames to list.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.path:
        print("[INFO] WRF-SFIRE output inspection is callable; pass --path run/wrfout_d01_* to read files.")
        print("[INFO] Model: https://github.com/openwfm/WRF-SFIRE")
        return 0
    try:
        grid = read_wrf_sfire_fire_grid(args.path, variables=("TIGN_G", "LFN"), origin="lower")
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 2
    rows, cols = grid.shape
    print(f"[OK] files: {len(grid.source_files)} | frames: {len(grid.times)}")
    print(f"[OK] fire grid: {rows} x {cols} cells of {grid.fire_dx:g} x {grid.fire_dy:g} m (sr_x={grid.sr_x}, sr_y={grid.sr_y})")
    cell_area = grid.fire_dx * grid.fire_dy
    masks = grid.burned_masks()
    for index in range(min(args.max_frames, len(grid.times))):
        label = grid.time_strings[index] if grid.time_strings else f"t={grid.times[index]:g} s"
        print(f"  {label}: burned area {masks[index].sum() * cell_area:.0f} m^2")
    arrival = grid.arrival_time()
    burned = np.isfinite(arrival)
    if burned.any():
        print(f"[OK] arrival time (TIGN_G) over burned cells: {arrival[burned].min():g} .. {arrival[burned].max():g} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
