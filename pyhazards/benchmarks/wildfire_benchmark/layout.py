from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

WILDFIRE_RUNS_ROOT = Path(__file__).resolve().parents[3] / "runs" / "wildfire_benchmark"


@dataclass(frozen=True)
class RunPaths:
    track: str
    run_name: str
    model_name: str
    seed: int
    run_root: Path
    model_root: Path
    seed_root: Path
    experiment_setting_path: Path
    history_csv_path: Path
    loss_curve_path: Path
    metrics_path: Path
    model_summary_path: Path
    model_template_path: Path
    benchmark_summary_path: Path
    benchmark_contract_snapshot_path: Path


def prepare_run_paths(
    track: str,
    run_name: str,
    model_name: str,
    seed: int,
    create: bool = True,
) -> RunPaths:
    if track not in {"smoke", "real", "archive"}:
        raise ValueError(f"Unsupported wildfire benchmark track: {track!r}")

    run_root = WILDFIRE_RUNS_ROOT / track / run_name
    model_root = run_root / model_name
    seed_root = model_root / f"seed_{int(seed)}"

    paths = RunPaths(
        track=track,
        run_name=run_name,
        model_name=model_name,
        seed=int(seed),
        run_root=run_root,
        model_root=model_root,
        seed_root=seed_root,
        experiment_setting_path=seed_root / "experiment_setting.json",
        history_csv_path=seed_root / "history.csv",
        loss_curve_path=seed_root / "loss_curve.png",
        metrics_path=seed_root / "metrics.json",
        model_summary_path=model_root / "model_summary.json",
        model_template_path=model_root / "model_template.json",
        benchmark_summary_path=run_root / "benchmark_summary.json",
        benchmark_contract_snapshot_path=run_root / "benchmark_contract_snapshot.json",
    )

    if create:
        seed_root.mkdir(parents=True, exist_ok=True)
    return paths
