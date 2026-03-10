from pathlib import Path

from pyhazards.benchmark_catalog import (
    BENCHMARK_PAGE_PATH,
    benchmark_catalog_alignment_issues,
    render_benchmark_page,
    rendered_benchmark_docs,
)


def test_benchmark_catalog_aligns_with_registry_and_configs() -> None:
    assert not benchmark_catalog_alignment_issues()


def test_benchmark_page_lists_family_and_ecosystem_tables() -> None:
    page = render_benchmark_page()

    assert "At a Glance" in page
    assert "Benchmark Families" in page
    assert "Coverage Matrix" in page
    assert "Benchmark Ecosystems" in page
    assert "Programmatic Use" in page
    assert page.index("Wildfire") < page.index("Earthquake")
    assert ".. tab-item:: Tropical Cyclone" in page
    assert "Tropical Cyclone / Hurricane" not in page

    family_cards = [
        ".. grid-item-card:: Wildfire Benchmark",
        ".. grid-item-card:: Earthquake Benchmark",
        ".. grid-item-card:: Flood Benchmark",
        ".. grid-item-card:: Tropical Cyclone Benchmark",
    ]
    for card in family_cards:
        assert page.count(card) == 1

    ecosystem_cards = [
        ".. grid-item-card:: WildfireSpreadTS",
        ".. grid-item-card:: SeisBench",
        ".. grid-item-card:: pick-benchmark",
        ".. grid-item-card:: pyCSEP",
        ".. grid-item-card:: AEFA",
        ".. grid-item-card:: Caravan",
        ".. grid-item-card:: WaterBench",
        ".. grid-item-card:: FloodCastBench",
        ".. grid-item-card:: HydroBench",
        ".. grid-item-card:: TCBench Alpha",
        ".. grid-item-card:: IBTrACS",
        ".. grid-item-card:: TropiCycloneNet-Dataset",
    ]
    for card in ecosystem_cards:
        assert page.count(card) == 1

    assert "WildfireSpreadTS: A Dataset of Multi-Modal Time Series for Wildfire Spread Prediction" in page
    assert "8 smoke configs | 8 models | 1 ecosystem" in page
    assert "5 smoke configs | 5 models | 4 ecosystems" in page
    assert "6 smoke configs | 6 models | 4 ecosystems" in page
    assert "8 smoke configs | 8 models | 3 ecosystems" in page


def test_rendered_docs_include_detail_pages_with_absolute_cross_links() -> None:
    docs = rendered_benchmark_docs()
    benchmark_docs_dir = BENCHMARK_PAGE_PATH.parent / "benchmarks"

    assert BENCHMARK_PAGE_PATH in docs
    earthquake_detail = docs[benchmark_docs_dir / "earthquake_benchmark.rst"]
    ecosystem_detail = docs[benchmark_docs_dir / "seisbench.rst"]

    assert ":doc:`SeisBench </benchmarks/seisbench>`" in earthquake_detail
    assert ":doc:`WaveCastNet </modules/models_wavecastnet>`" in earthquake_detail
    assert ":doc:`Earthquake Benchmark </benchmarks/earthquake_benchmark>`" in ecosystem_detail
    assert ":doc:`PhaseNet </modules/models_phasenet>`" in ecosystem_detail
    assert ".. dropdown:: Supported Tasks" in ecosystem_detail
    assert ".. dropdown:: Linked Models" in ecosystem_detail


def test_api_reference_order_is_curated() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    index_text = (repo_root / "docs" / "source" / "index.rst").read_text(encoding="utf-8")
    api_toc_start = index_text.index(":caption: API Reference")
    additional_info_start = index_text.index(":caption: Additional Information")
    api_toc_text = index_text[api_toc_start:additional_info_start]
    api_order = [
        "pyhazards_datasets",
        "pyhazards_models",
        "pyhazards_benchmarks",
        "pyhazards_configs",
        "pyhazards_reports",
        "pyhazards_engine",
        "pyhazards_metrics",
        "pyhazards_utils",
        "interactive_map",
    ]
    positions = [api_toc_text.index(name) for name in api_order]
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
