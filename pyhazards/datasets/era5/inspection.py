from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import xarray as xr


def _default_era5_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "era5_subset"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pyhazards.datasets.era5.inspection",
        description="Inspect local ERA5 NetCDF files.",
    )
    parser.add_argument(
        "--path",
        default=str(_default_era5_path()),
        help="Path to directory containing ERA5 NetCDF files.",
    )
    parser.add_argument(
        "--max-vars",
        type=int,
        default=20,
        help="Maximum number of variable names to print.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_path = Path(args.path).expanduser().resolve()

    files = sorted(data_path.glob("*.nc"))
    if not files:
        print(f"[ERROR] No ERA5 NetCDF files found in: {data_path}")
        return 2

    print(f"[INFO] ERA5 files found: {len(files)}")
    try:
        ds = xr.open_mfdataset(files, combine="by_coords", chunks={})
        try:
            print("[OK] Dataset opened successfully (xarray).")
            print(f"[OK] Dimensions: {dict(ds.sizes)}")
            vars_list = list(ds.data_vars)
            print(f"[OK] Data variables: {len(vars_list)}")
            for name in vars_list[: args.max_vars]:
                print(f"  - {name}")
        finally:
            ds.close()
        return 0
    except Exception as exc:
        print(f"[WARN] xarray open failed ({exc}). Falling back to h5py inspection.")

    sample = files[0]
    with h5py.File(sample, "r") as h5:
        datasets: list[str] = []

        def collect(name: str, obj) -> None:
            if isinstance(obj, h5py.Dataset):
                datasets.append(name)

        h5.visititems(collect)

    print(f"[OK] HDF5/NetCDF file opened: {sample.name}")
    print(f"[OK] Datasets discovered: {len(datasets)}")
    for name in datasets[: args.max_vars]:
        print(f"  - {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
