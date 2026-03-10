from pyhazards.dataset_catalog import (
    API_PAGE_PATH,
    DATASET_PAGE_PATH,
    dataset_catalog_alignment_issues,
    load_dataset_cards,
    render_dataset_api_page,
    render_dataset_page,
    rendered_dataset_docs,
)


def test_dataset_catalog_aligns_with_registry_and_links() -> None:
    cards = load_dataset_cards()
    assert not dataset_catalog_alignment_issues(cards)


def test_dataset_page_lists_curated_hazard_tabs() -> None:
    cards = load_dataset_cards()
    page = render_dataset_page(cards)

    assert "At a Glance" in page
    assert "Catalog by Hazard" in page
    assert "Recommended Entry Points" in page
    assert "Programmatic Use" in page
    assert ".. tab-set::" in page

    tabs = [
        ".. tab-item:: Shared Forcing",
        ".. tab-item:: Wildfire",
        ".. tab-item:: Flood",
        ".. tab-item:: Earthquake",
        ".. tab-item:: Tropical Cyclone",
    ]
    for tab in tabs:
        assert tab in page

    positions = [page.index(tab) for tab in tabs]
    assert positions == sorted(positions)

    assert ":doc:`ERA5 <datasets/era5>`" in page
    assert ":doc:`WFIGS <datasets/wfigs>`" in page
    assert ":doc:`Caravan <datasets/caravan_streamflow>`" in page
    assert ":doc:`SeisBench <datasets/seisbench_waveforms>`" in page
    assert ":doc:`IBTrACS <datasets/ibtracs_tracks>`" in page
    assert "Registry-loadable Datasets" in page
    assert "Inspection Entry Points" in page


def test_dataset_api_and_detail_pages_use_generated_structure() -> None:
    cards = load_dataset_cards()
    docs = rendered_dataset_docs(cards)

    assert DATASET_PAGE_PATH in docs
    assert API_PAGE_PATH in docs

    api_page = render_dataset_api_page(cards)
    assert "Catalog Summary" in api_page
    assert "Developer Dataset Workflow" in api_page
    assert "Inspect an External Dataset Source" in api_page
    assert "Register a Custom Dataset" in api_page
    assert "pyhazards/dataset_cards" in api_page

    dataset_docs_dir = DATASET_PAGE_PATH.parent / "datasets"
    era5_detail = docs[dataset_docs_dir / "era5.rst"]
    seisbench_detail = docs[dataset_docs_dir / "seisbench_waveforms.rst"]
    ibtracs_detail = docs[dataset_docs_dir / "ibtracs_tracks.rst"]

    for detail in (era5_detail, seisbench_detail, ibtracs_detail):
        assert "Overview" in detail
        assert "At a Glance" in detail
        assert "Data Characteristics" in detail
        assert "Access" in detail
        assert "PyHazards Usage" in detail
        assert "Inspection Workflow" in detail
        assert "Reference" in detail

    assert "This dataset is currently documented as an external or inspection-first" in era5_detail
    assert ":doc:`Earthquake Benchmark </benchmarks/earthquake_benchmark>`" in seisbench_detail
    assert "there is no standalone inspection cli documented for it." in seisbench_detail.lower()
    assert ":doc:`Tropical Cyclone Benchmark </benchmarks/tropical_cyclone_benchmark>`" in ibtracs_detail
