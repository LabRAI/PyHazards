from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

from ..tasks import get_hazard_task

_REPORT_FORMATS = {"json", "md", "csv"}


@dataclass
class DatasetRef:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelRef:
    name: str
    task: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportConfig:
    output_dir: str = "reports"
    formats: List[str] = field(default_factory=lambda: ["json"])

    def __post_init__(self) -> None:
        normalized = [fmt.lower() for fmt in self.formats]
        unknown = [fmt for fmt in normalized if fmt not in _REPORT_FORMATS]
        if unknown:
            raise ValueError(
                "Unknown report format(s): {unknown}. Known: {known}".format(
                    unknown=", ".join(sorted(set(unknown))),
                    known=", ".join(sorted(_REPORT_FORMATS)),
                )
            )
        self.formats = normalized


@dataclass
class BenchmarkConfig:
    name: str
    hazard_task: str
    metrics: List[str] = field(default_factory=list)
    eval_split: str = "test"
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.hazard_task = get_hazard_task(self.hazard_task).name


@dataclass
class ExperimentConfig:
    benchmark: BenchmarkConfig
    dataset: DatasetRef
    model: ModelRef
    report: ReportConfig = field(default_factory=ReportConfig)
    seed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return ExperimentConfig(
        benchmark=BenchmarkConfig(**raw["benchmark"]),
        dataset=DatasetRef(**raw["dataset"]),
        model=ModelRef(**raw["model"]),
        report=ReportConfig(**raw.get("report", {})),
        seed=raw.get("seed", 0),
        metadata=raw.get("metadata", {}),
    )


def dump_experiment_config(config: ExperimentConfig, path: str | Path) -> None:
    payload = config.to_dict()
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


__all__ = [
    "BenchmarkConfig",
    "DatasetRef",
    "ExperimentConfig",
    "ModelRef",
    "ReportConfig",
    "dump_experiment_config",
    "load_experiment_config",
]
