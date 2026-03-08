import csv
import json

from pyhazards.reports import BenchmarkReport, export_report_bundle


def test_export_report_bundle_writes_requested_formats(tmp_path):
    report = BenchmarkReport(
        benchmark_name="dummy_benchmark",
        hazard_task="earthquake.picking",
        metrics={"mae": 0.5, "f1": 0.9},
        metadata={"split": "test"},
    )
    paths = export_report_bundle(report, tmp_path, formats=["json", "md", "csv"])

    json_payload = json.loads((tmp_path / "dummy_benchmark.json").read_text(encoding="utf-8"))
    assert json_payload["hazard_task"] == "earthquake.picking"
    assert json_payload["metrics"]["mae"] == 0.5

    markdown = (tmp_path / "dummy_benchmark.md").read_text(encoding="utf-8")
    assert "# dummy_benchmark" in markdown
    assert "`earthquake.picking`" in markdown

    with (tmp_path / "dummy_benchmark.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["mae"] == "0.5"
    assert rows[0]["metadata.split"] == "test"
    assert paths["json"].endswith("dummy_benchmark.json")
