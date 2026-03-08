from __future__ import annotations

import argparse
from pathlib import Path

from pyhazards.model_catalog import load_model_cards, sync_generated_docs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render generated model catalog pages from YAML model cards."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if generated files are out of date instead of writing them.",
    )
    args = parser.parse_args()

    cards = load_model_cards()
    changes = sync_generated_docs(cards, check=args.check)

    if changes:
        action = "would update" if args.check else "updated"
        print("Model docs {action}:".format(action=action))
        for path in changes:
            print(" - {path}".format(path=Path(path)))
        return 1 if args.check else 0

    print("Model docs are in sync.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
