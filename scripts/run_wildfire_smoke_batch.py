from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyhazards.benchmarks.wildfire_benchmark import run_smoke_batch
from pyhazards.benchmarks.wildfire_benchmark.adapters import create_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run wildfire benchmark smoke batches inside my-copy.")
    parser.add_argument("--track", default="smoke", choices=["smoke", "real", "archive"])
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--catalog_kind", default="main", choices=["main", "extensions"])
    parser.add_argument("--catalog_path", default=None)
    parser.add_argument("--contract_path", default=None)
    parser.add_argument("--source_tier", default="all")
    parser.add_argument("--models", default="")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--limit_models", type=int, default=0)
    parser.add_argument("--max_epoch_steps", type=int, default=12)
    parser.add_argument("--max_round_steps", type=int, default=30)
    parser.add_argument("--max_iter_steps", type=int, default=20)
    parser.add_argument("--max_tree_steps", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_models = [item.strip() for item in args.models.split(",") if item.strip()] or None
    run_root = run_smoke_batch(
        adapter_factory=create_adapter,
        run_name=args.run_name,
        track=args.track,
        catalog_kind=args.catalog_kind,
        catalog_path=args.catalog_path,
        contract_path=args.contract_path,
        source_tier=args.source_tier,
        models=selected_models,
        seeds=args.seeds,
        limit_models=args.limit_models,
        step_limits={
            "epoch": int(args.max_epoch_steps),
            "round": int(args.max_round_steps),
            "iteration": int(args.max_iter_steps),
            "tree": int(args.max_tree_steps),
        },
    )
    print(f"[done] wildfire benchmark smoke batch saved to: {run_root}")


if __name__ == "__main__":
    main()
