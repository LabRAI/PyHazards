from __future__ import annotations

import argparse

from pyhazards.benchmarks import available_benchmarks
from pyhazards.configs import load_experiment_config
from pyhazards.engine.runner import BenchmarkRunner
from pyhazards.tasks import available_hazard_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a PyHazards benchmark from an experiment config.",
    )
    parser.add_argument("--config", help="Path to an experiment YAML config.")
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="Print registered benchmark names and exit.",
    )
    parser.add_argument(
        "--list-hazard-tasks",
        action="store_true",
        help="Print canonical hazard-task names and exit.",
    )
    parser.add_argument(
        "--output-dir",
        help="Override the output directory declared in the config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_benchmarks:
        for name in available_benchmarks():
            print(name)
        return 0

    if args.list_hazard_tasks:
        for name in available_hazard_tasks():
            print(name)
        return 0

    if not args.config:
        raise SystemExit("--config is required unless a list flag is used.")

    experiment = load_experiment_config(args.config)
    summary = BenchmarkRunner().run(experiment, output_dir=args.output_dir)
    print("benchmark:", summary.benchmark_name)
    print("hazard_task:", summary.hazard_task)
    for key, value in sorted(summary.metrics.items()):
        print("metric.{key}={value}".format(key=key, value=value))
    for fmt, path in sorted(summary.report_paths.items()):
        print("report.{fmt}={path}".format(fmt=fmt, path=path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
