from __future__ import annotations

import argparse
from pathlib import Path

from pyhazards.dataset_catalog import load_dataset_cards, sync_generated_dataset_docs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render generated dataset catalog pages from YAML dataset cards."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if generated files are out of date instead of writing them.",
    )
    args = parser.parse_args()

    cards = load_dataset_cards()
    changes = sync_generated_dataset_docs(cards, check=args.check)

    if changes:
        action = "would update" if args.check else "updated"
        print(f"Dataset docs {action}:")
        for path in changes:
            print(f" - {Path(path)}")
        return 1 if args.check else 0

    print("Dataset docs are in sync.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
