from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass
class BenchmarkSection:
    name: str = "WildfireBench"
    contract_version: str = "track_o_2024_v1"
    mode: str = "scaffold_no_data"
    task: str = "Track-O"


@dataclass
class EvaluationProtocolSection:
    discrimination: dict[str, Any] = field(
        default_factory=lambda: {"primary": "auprc", "secondary": "auroc"}
    )
    reliability: dict[str, Any] = field(
        default_factory=lambda: {"metrics": ["brier", "nll", "ece"]}
    )
    temporal_consistency: dict[str, Any] = field(
        default_factory=lambda: {
            "metrics": ["mean_day_to_day_change", "normalized_consistency_score"]
        }
    )


@dataclass
class ModelSection:
    name: str
    display_name: str
    group: str
    source_tier: str
    train_unit: str
    defaults: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunSection:
    seed: int = 42
    num_steps: int = 0
    best_step: int = 0
    converged_step: int = 0
    step_name: str = "epoch"
    learning_weight: dict[str, Any] = field(
        default_factory=lambda: {
            "kind": "pos_weight_neg_over_pos",
            "value": "to_be_computed_from_real_train_split",
            "clip_max": 50.0,
        }
    )


@dataclass
class WildfireExperimentSetting:
    benchmark: BenchmarkSection
    evaluation_protocol: EvaluationProtocolSection
    model: ModelSection
    run: RunSection
    metrics: dict[str, Any] = field(
        default_factory=lambda: {
            "auprc": None,
            "auroc": None,
            "brier": None,
            "nll": None,
            "ece": None,
            "mean_day_to_day_change": None,
            "normalized_consistency_score": None,
        }
    )
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False), encoding="utf-8")
        return target


def build_default_experiment_setting(
    *,
    model_name: str,
    display_name: str,
    group: str,
    source_tier: str,
    train_unit: str,
    defaults: Mapping[str, Any] | None = None,
    seed: int = 42,
    num_steps: int = 0,
    best_step: int = 0,
    converged_step: int = 0,
    step_name: str = "epoch",
    mode: str = "scaffold_no_data",
    task: str = "Track-O",
    notes: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> WildfireExperimentSetting:
    setting = WildfireExperimentSetting(
        benchmark=BenchmarkSection(mode=mode, task=task),
        evaluation_protocol=EvaluationProtocolSection(),
        model=ModelSection(
            name=model_name,
            display_name=display_name,
            group=group,
            source_tier=source_tier,
            train_unit=train_unit,
            defaults=dict(defaults or {}),
        ),
        run=RunSection(
            seed=int(seed),
            num_steps=int(num_steps),
            best_step=int(best_step),
            converged_step=int(converged_step),
            step_name=step_name,
        ),
        notes=dict(notes or {}),
    )
    if metrics:
        setting.metrics.update(dict(metrics))
    return setting


def write_experiment_setting(
    path: str | Path,
    setting: WildfireExperimentSetting,
) -> Path:
    return setting.write_json(path)
