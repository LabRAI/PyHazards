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

    assert "Benchmark Families" in page
    assert "Benchmark Ecosystems" in page
    assert page.index("Wildfire") < page.index("Earthquake")

    family_rows = [
        "   * - Wildfire\n     - :doc:`Wildfire Benchmark <benchmarks/wildfire_benchmark>`",
        "   * - Earthquake\n     - :doc:`Earthquake Benchmark <benchmarks/earthquake_benchmark>`",
        "   * - Flood\n     - :doc:`Flood Benchmark <benchmarks/flood_benchmark>`",
        (
            "   * - Tropical Cyclone / Hurricane\n"
            "     - :doc:`Tropical Cyclone Benchmark <benchmarks/tropical_cyclone_benchmark>`"
        ),
    ]
    for row in family_rows:
        assert page.count(row) == 1

    ecosystem_rows = [
        "   * - Wildfire\n     - :doc:`WildfireSpreadTS <benchmarks/wildfirespreadts_ecosystem>`",
        "   * - Earthquake\n     - :doc:`SeisBench <benchmarks/seisbench>`",
        "   * - Earthquake\n     - :doc:`pick-benchmark <benchmarks/pick_benchmark>`",
        "   * - Earthquake\n     - :doc:`pyCSEP <benchmarks/pycsep>`",
        "   * - Earthquake\n     - :doc:`AEFA <benchmarks/aefa>`",
        "   * - Flood\n     - :doc:`Caravan <benchmarks/caravan>`",
        "   * - Flood\n     - :doc:`WaterBench <benchmarks/waterbench>`",
        "   * - Flood\n     - :doc:`FloodCastBench <benchmarks/floodcastbench>`",
        "   * - Flood\n     - :doc:`HydroBench <benchmarks/hydrobench>`",
        (
            "   * - Tropical Cyclone / Hurricane\n"
            "     - :doc:`TCBench Alpha <benchmarks/tcbench_alpha>`"
        ),
        "   * - Tropical Cyclone / Hurricane\n     - :doc:`IBTrACS <benchmarks/ibtracs>`",
        (
            "   * - Tropical Cyclone / Hurricane\n"
            "     - :doc:`TropiCycloneNet-Dataset <benchmarks/tropicyclonenet_dataset>`"
        ),
    ]
    for row in ecosystem_rows:
        assert page.count(row) == 1

    assert "WildfireSpreadTS: A Dataset of Multi-Modal Time Series for Wildfire Spread Prediction" in page
    assert "``wildfirespreadts_smoke.yaml``" in page
    assert "``wavecastnet_benchmark_smoke.yaml``" in page
    assert "``floodcast_smoke.yaml``" in page
    assert "``tropicyclonenet_smoke.yaml``" in page


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
