from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class BenchmarkResult:
    benchmark_name: str
    hazard_task: str
    metrics: Dict[str, float]
    predictions: List[Any] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkRunSummary:
    benchmark_name: str
    hazard_task: str
    metrics: Dict[str, float]
    report_paths: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
