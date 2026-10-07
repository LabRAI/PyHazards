from pathlib import Path

from pyhazards.appendix_a_catalog import (
    APPENDIX_A_ENTRIES,
    AppendixAEntry,
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
    assert "``Implemented``" in page
    assert "``Experimental``" in page
    assert "GraphCast / GenCast" in page
    assert "models_forefire" not in page
    assert "models_wrf_sfire" not in page
    assert ":doc:`FireCastNet <modules/models_firecastnet>`" in page
    assert ":doc:`WaveCastNet <modules/models_wavecastnet>`" in page


def test_appendix_a_page_is_linked_from_docs_index() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    index_text = (repo_root / "docs" / "source" / "index.rst").read_text(encoding="utf-8")
    assert "appendix_a_coverage" in index_text


def test_external_simulators_are_listed_without_model_cards() -> None:
    cards = load_model_cards()
    page = render_appendix_a_page(cards)
    entries = {entry.source_name: entry for entry in APPENDIX_A_ENTRIES}
    for name in ("ForeFire", "WRF-SFIRE"):
        assert entries[name].status == "external"
        assert not entries[name].mapped_models
    assert "``External simulator``" in page
    assert "     - External" in page
    assert ":doc:`Simulators <pyhazards_simulators>`" in page
    assert ":doc:`WRF-SFIRE Outputs <datasets/wrf_sfire>`" in page
    assert not {"forefire", "wrf_sfire"} & {card.model_name for card in cards}


def test_external_status_rules_are_enforced() -> None:
    cards = load_model_cards()
    bad = [
        AppendixAEntry("Wildfire", "Sim A", "Simulator Adapter", "https://example.org", "external", ("firecastnet",),
                       mapped_docs=(("Simulators", "pyhazards_simulators"),)),
        AppendixAEntry("Wildfire", "Sim B", "Simulator Adapter", "https://example.org", "external"),
        AppendixAEntry("Wildfire", "Sim C", "Simulator Adapter", "https://example.org", "external",
                       mapped_docs=(("Nowhere", "no_such_page"), ("No data", "datasets/no_such_dataset"))),
        AppendixAEntry("Wildfire", "Sim D", "Simulator Adapter", "https://example.org", "unknown"),
    ]
    issues = appendix_a_alignment_issues(cards, entries=bad)
    assert any("Sim A" in issue and "must not map to model cards" in issue for issue in issues)
    assert any("Sim B" in issue and "links no documentation" in issue for issue in issues)
    assert any("no_such_page" in issue for issue in issues)
    assert any("no_such_dataset" in issue for issue in issues)
    assert any("Sim D" in issue and "unknown status" in issue for issue in issues)
