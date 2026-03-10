from __future__ import annotations

import argparse
from pathlib import Path

from pyhazards.appendix_a_catalog import sync_generated_appendix_a_docs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render coverage-audit docs from the checked-in audit catalog."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if generated files are out of date instead of writing them.",
    )
    args = parser.parse_args()

    changes = sync_generated_appendix_a_docs(check=args.check)
    if changes:
        action = "would update" if args.check else "updated"
        print("Coverage audit docs {action}:".format(action=action))
        for path in changes:
            print(" - {path}".format(path=Path(path)))
        return 1 if args.check else 0

    print("Coverage audit docs are in sync.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
