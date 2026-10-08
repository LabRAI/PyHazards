from __future__ import annotations

import argparse
from pathlib import Path

from pyhazards.prompted.catalog import sync_prompted_docs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render docs/source/pyhazards_prompted.rst from the prompted-baseline cards."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the generated page is out of date instead of writing it.",
    )
    args = parser.parse_args()

    changes = sync_prompted_docs(check=args.check)
    if changes:
        action = "would update" if args.check else "updated"
        print(f"Prompted baseline docs {action}:")
        for path in changes:
            print(f" - {Path(path)}")
        return 1 if args.check else 0

    print("Prompted baseline docs are in sync.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
