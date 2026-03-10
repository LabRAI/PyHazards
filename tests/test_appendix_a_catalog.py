from pathlib import Path

from pyhazards.appendix_a_catalog import (
    appendix_a_alignment_issues,
    render_appendix_a_page,
)
from pyhazards.model_catalog import load_model_cards


def test_appendix_a_catalog_aligns_with_model_card_statuses() -> None:
    cards = load_model_cards()
    assert not appendix_a_alignment_issues(cards)


def test_appendix_a_page_lists_missing_and_non_core_entries() -> None:
    cards = load_model_cards()
    page = render_appendix_a_page(cards)
    assert "Coverage Audit" in page
    assert "`wildfire_forecasting <https://github.com/Orion-AI-Lab/wildfire_forecasting>`_" in page
    assert "``Missing``" in page
    assert "``Experimental``" in page
    assert "GraphCast / GenCast" in page
    assert "Shen et al. (2023) FPA-FOD family" in page
    assert ":doc:`WaveCastNet <modules/models_wavecastnet>`" in page


def test_appendix_a_page_is_linked_from_docs_index() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    index_text = (repo_root / "docs" / "source" / "index.rst").read_text(encoding="utf-8")
    assert "appendix_a_coverage" in index_text
