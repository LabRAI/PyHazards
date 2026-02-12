from __future__ import annotations

import argparse
from pathlib import Path


def run_generic_dataset_inspection(
    dataset_name: str,
    dataset_doc_url: str,
    argv: list[str] | None = None,
) -> int:
    """
    Lightweight inspection entrypoint for datasets without a dedicated parser yet.
    This keeps module paths stable and callable from CLI.
    """
    parser = argparse.ArgumentParser(
        prog=f"python -m pyhazards.datasets.{dataset_name}.inspection",
        description=f"Inspect local {dataset_name} dataset files.",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="Path to a local file or directory for this dataset.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=10,
        help="Maximum number of directory entries to print (if --path is a directory).",
    )
    args = parser.parse_args(argv)

    print(f"[INFO] Dataset inspection entrypoint for '{dataset_name}' is callable.")
    print(f"[INFO] Reference: {dataset_doc_url}")

    if args.path is None:
        print("[INFO] No --path provided. Pass --path to validate local files.")
        return 0

    path = Path(args.path).expanduser().resolve()
    if not path.exists():
        print(f"[ERROR] Path does not exist: {path}")
        return 2

    if path.is_file():
        print(f"[OK] File exists: {path}")
        print(f"[OK] Size (bytes): {path.stat().st_size}")
        return 0

    files = sorted([p for p in path.iterdir() if p.is_file()])
    dirs = sorted([p for p in path.iterdir() if p.is_dir()])
    print(f"[OK] Directory exists: {path}")
    print(f"[OK] Files: {len(files)} | Subdirectories: {len(dirs)}")

    if files:
        print("[INFO] Sample files:")
        for p in files[: args.max_items]:
            print(f"  - {p.name}")

    return 0

