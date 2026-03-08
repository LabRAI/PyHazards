from __future__ import annotations

import argparse
from typing import List

from pyhazards.model_catalog import card_by_registry_name, load_model_cards, run_smoke_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run synthetic smoke tests for cataloged PyHazards models."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Specific registry names to test. Defaults to all cataloged models.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cards = load_model_cards()
    mapping = card_by_registry_name(cards)

    selected: List[str]
    if args.models:
        selected = []
        for name in args.models:
            if name not in mapping:
                raise SystemExit("Unknown cataloged model: {name}".format(name=name))
            card = mapping[name]
            if card.model_name not in selected:
                selected.append(card.model_name)
    else:
        selected = [card.model_name for card in cards]

    ok = True
    for name in selected:
        card = mapping[name]
        result = run_smoke_test(card)
        status = "PASS" if result["ok"] else "FAIL"
        print(
            "[{status}] {name}: expected {expected}, got {actual}".format(
                status=status,
                name=card.model_name,
                expected=result["expected_shape"],
                actual=result["actual_shape"],
            )
        )
        ok = ok and result["ok"]

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
