from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


@dataclass
class BenchmarkReport:
    benchmark_name: str
    hazard_task: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "hazard_task": self.hazard_task,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "artifacts": self.artifacts,
        }


def export_report_bundle(
    report: BenchmarkReport,
    output_dir: str | Path,
    formats: Sequence[str],
) -> Dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    for fmt in formats:
        fmt = fmt.lower()
        path = target / "{name}.{fmt}".format(name=report.benchmark_name, fmt=fmt)
        if fmt == "json":
            path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        elif fmt == "md":
            path.write_text(_markdown_report(report), encoding="utf-8")
        elif fmt == "csv":
            _write_csv(path, report.metrics, report.metadata)
        else:
            raise ValueError("Unsupported report format: {fmt}".format(fmt=fmt))
        paths[fmt] = str(path)
    return paths


def _markdown_report(report: BenchmarkReport) -> str:
    lines = [
        "# {name}".format(name=report.benchmark_name),
        "",
        "- Hazard task: `{task}`".format(task=report.hazard_task),
        "",
        "## Metrics",
        "",
    ]
    if report.metrics:
        for key, value in sorted(report.metrics.items()):
            lines.append("- `{key}`: {value}".format(key=key, value=value))
    else:
        lines.append("- No metrics recorded.")
    if report.metadata:
        lines.extend(["", "## Metadata", ""])
        for key, value in sorted(report.metadata.items()):
            lines.append("- `{key}`: {value}".format(key=key, value=value))
    if report.artifacts:
        lines.extend(["", "## Artifacts", ""])
        for key, value in sorted(report.artifacts.items()):
            lines.append("- `{key}`: {value}".format(key=key, value=value))
    lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, metrics: Mapping[str, float], metadata: Mapping[str, Any]) -> None:
    row: Dict[str, Any] = {}
    row.update(metrics)
    row.update({"metadata.{key}".format(key=key): value for key, value in metadata.items()})
    fieldnames = list(row.keys()) or ["status"]
    if not row:
        row = {"status": "empty"}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
