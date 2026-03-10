from pathlib import Path

from pyhazards.benchmark_catalog import (
    benchmark_catalog_alignment_issues,
    render_benchmark_page,
)


def test_benchmark_catalog_aligns_with_registry_and_configs() -> None:
    assert not benchmark_catalog_alignment_issues()


def test_benchmark_page_lists_summary_table_rows() -> None:
    page = render_benchmark_page()
    assert "Implemented Benchmarks" in page
    assert page.index("Wildfire") < page.index("Earthquake")
    assert "``WildfireBenchmark`` (``wildfire``)" in page
    assert "``EarthquakeBenchmark`` (``earthquake``)" in page
    assert "``FloodBenchmark`` (``flood``)" in page
    assert "``TropicalCycloneBenchmark`` (``tc``)" in page
    assert "``wildfire_danger_smoke.yaml``" in page
    assert "``wildfire_forecasting_smoke.yaml``" in page
    assert "``asufm_smoke.yaml``" in page
    assert "``wildfirespreadts_smoke.yaml``" in page
    assert "``forefire_smoke.yaml``" in page
    assert "``wrf_sfire_smoke.yaml``" in page
    assert "``firecastnet_smoke.yaml``" in page
    assert "``wavecastnet_benchmark_smoke.yaml``" in page
    assert "``fourcastnet_tc_smoke.yaml``" in page
    assert "``precision``" in page
    assert "``recall``" in page
    assert "Current Backing" in page
    assert "full roadmap parity" in page
    assert "SeisBench, pick-benchmark, AEFA, and pyCSEP-style integrations" in page


def test_api_reference_order_is_curated() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    index_text = (repo_root / "docs" / "source" / "index.rst").read_text(encoding="utf-8")
    api_order = [
        "pyhazards_datasets",
        "pyhazards_models",
        "interactive_map",
        "pyhazards_benchmarks",
        "pyhazards_configs",
        "pyhazards_reports",
        "pyhazards_engine",
        "pyhazards_metrics",
        "pyhazards_utils",
    ]
    positions = [index_text.index(name) for name in api_order]
    assert positions == sorted(positions)

    package_api_text = (
        repo_root / "docs" / "source" / "api" / "pyhazards.rst"
    ).read_text(encoding="utf-8")
    package_order = [
        "pyhazards.datasets",
        "pyhazards.models",
        "pyhazards.benchmarks",
        "pyhazards.configs",
        "pyhazards.reports",
        "pyhazards.engine",
        "pyhazards.metrics",
        "pyhazards.utils",
    ]
    package_positions = [package_api_text.index(name) for name in package_order]
    assert package_positions == sorted(package_positions)
