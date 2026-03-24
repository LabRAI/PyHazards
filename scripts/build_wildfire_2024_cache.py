from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyhazards.benchmarks.wildfire_benchmark.cache_builder import build_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the wildfire 2024 real-data cache for benchmark runs.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "pyhazards" / "configs" / "wildfire_benchmark" / "cache_2024_v1.yaml"),
    )
    parser.add_argument("--limit_days", type=int, default=0, help="Only materialize the first N shared dates for smoke-like validation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_cache(args.config, limit_days=int(args.limit_days))
    print("[done] wildfire cache built")
    print(f"cache_root={summary.cache_root}")
    print(f"label_days={summary.n_label_days} met_days={summary.n_met_days} shared_days={summary.n_shared_days}")
    print(f"train={summary.train_days} val={summary.val_days} test={summary.test_days}")
    print(f"weather_vars={','.join(summary.weather_vars)}")


if __name__ == "__main__":
    main()
