from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .experiment_settings import build_default_experiment_setting

STEP_LABEL = {
    "epoch": "epoch",
    "round": "round",
    "iteration": "iteration",
    "tree": "tree_count",
}


@dataclass
class AdapterRunOutput:
    history: List[Dict[str, float]]
    metrics: Dict[str, float]
    best_step: int
    converged_step: int
    train_unit: str
    notes: Dict[str, Any]


def mean_std(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=0))}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")


def write_history_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_loss_curve(history: List[Dict[str, float]], train_unit: str, output_png: Path, title: str) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    step_key = STEP_LABEL[train_unit]
    x = [int(row[step_key]) for row in history]
    y_tr = [float(row["train_loss"]) for row in history]
    y_va = [float(row["val_loss"]) for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y_tr, marker="o", linewidth=1.6, label="train_loss")
    plt.plot(x, y_va, marker="s", linewidth=1.4, label="val_loss")
    plt.xlabel(step_key)
    plt.ylabel("loss")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()


def build_model_template(contract: Dict[str, Any], model_spec: Dict[str, Any]) -> Dict[str, Any]:
    shared = contract["shared_training"]
    seed_list = shared.get("seed_list") or shared.get("dry_run_seed_list") or shared.get("final_seed_list") or [42]
    return {
        "template_version": "track_o_model_template_v1",
        "benchmark_name": contract["benchmark_name"],
        "contract_version": contract["contract_version"],
        "task": contract["task"],
        "model": {
            "name": model_spec["name"],
            "display_name": model_spec["display_name"],
            "group": model_spec["group"],
            "source_tier": model_spec["source_tier"],
            "train_unit": model_spec["train_unit"],
            "defaults": model_spec.get("defaults", {}),
        },
        "reproducibility": {
            "seed_list": seed_list,
            "must_report_mean_std": contract["shared_training"]["report_requirements"]["report_mean_std_across_seeds"],
        },
        "expected_metrics": [
            "auprc",
            "auroc",
            "brier",
            "nll",
            "ece",
            "mean_day_to_day_change",
            "normalized_consistency_score",
        ],
        "required_fields_for_real_runs": [
            "repo_url",
            "repo_commit_or_tag",
            "data_version",
            "split_version",
            "feature_set_version",
            "hyperparam_search_budget",
            "hardware",
            "software_versions",
        ],
    }


def build_experiment_setting_from_run_output(
    *,
    contract: Dict[str, Any],
    model_spec: Dict[str, Any],
    seed: int,
    run_output: AdapterRunOutput,
) -> Dict[str, Any]:
    setting = build_default_experiment_setting(
        model_name=str(model_spec["name"]),
        display_name=str(model_spec["display_name"]),
        group=str(model_spec["group"]),
        source_tier=str(model_spec["source_tier"]),
        train_unit=str(model_spec["train_unit"]),
        defaults=model_spec.get("defaults", {}),
        seed=int(seed),
        num_steps=len(run_output.history),
        best_step=int(run_output.best_step),
        converged_step=int(run_output.converged_step),
        step_name=STEP_LABEL[str(model_spec["train_unit"])],
        mode=str(contract["mode"]),
        task=str(contract["task"]),
        notes=dict(run_output.notes),
        metrics=run_output.metrics,
    )
    setting.run.learning_weight = {
        "kind": contract["shared_training"]["class_imbalance"]["policy"],
        "value": "to_be_computed_from_real_train_split",
        "clip_max": contract["shared_training"]["class_imbalance"]["clip_max"],
    }
    return setting.to_dict()
