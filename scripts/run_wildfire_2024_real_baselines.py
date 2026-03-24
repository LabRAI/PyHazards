from __future__ import annotations

import argparse
from pathlib import Path

from pyhazards.benchmarks.wildfire_benchmark.real_runner import run_real_baselines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run first real-data wildfire baselines on the 2024 cache.")
    parser.add_argument("--cache_dir", type=str, default="/home/runyang/my-copy/data_cache/wildfire_2024_v1")
    parser.add_argument("--run_name", type=str, default="track_o_2024_real_v1_first4_dryrun")
    parser.add_argument("--models", type=str, default="logistic_regression,xgboost,unet,convlstm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_limit_days", type=int, default=0)
    parser.add_argument("--val_limit_days", type=int, default=0)
    parser.add_argument("--test_limit_days", type=int, default=0)
    parser.add_argument("--tabular_downsample", type=int, default=8)
    parser.add_argument("--raster_downsample", type=int, default=4)
    parser.add_argument("--temporal_downsample", type=int, default=8)
    parser.add_argument("--temporal_history", type=int, default=6)
    parser.add_argument("--xgboost_rounds", type=int, default=120)
    parser.add_argument("--lightgbm_rounds", type=int, default=120)
    parser.add_argument("--unet_epochs", type=int, default=12)
    parser.add_argument("--convlstm_epochs", type=int, default=12)
    parser.add_argument("--deep_patience", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_models = [item.strip() for item in str(args.models).split(",") if item.strip()]
    run_root = run_real_baselines(
        cache_dir=args.cache_dir,
        run_name=args.run_name,
        models=selected_models,
        seed=args.seed,
        train_limit_days=args.train_limit_days or None,
        val_limit_days=args.val_limit_days or None,
        test_limit_days=args.test_limit_days or None,
        tabular_downsample=args.tabular_downsample,
        raster_downsample=args.raster_downsample,
        temporal_downsample=args.temporal_downsample,
        temporal_history=args.temporal_history,
        xgboost_rounds=args.xgboost_rounds,
        lightgbm_rounds=args.lightgbm_rounds,
        unet_epochs=args.unet_epochs,
        convlstm_epochs=args.convlstm_epochs,
        deep_patience=args.deep_patience,
        device=args.device,
    )
    print(f"[done] real wildfire benchmark run written to {Path(run_root)}")


if __name__ == "__main__":
    main()
