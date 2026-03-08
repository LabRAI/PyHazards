from __future__ import annotations

import argparse
from typing import Sequence

from .interactive_map import open_interactive_map


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pyhazards",
        description="PyHazards command line utilities.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "map",
        help="Open the external RAI Fire interactive map.",
        description="Open the external RAI Fire interactive map.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "map":
        url = open_interactive_map(open_browser=True)
        print(f"RAI Fire interactive map: {url}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
