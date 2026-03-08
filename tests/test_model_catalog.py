from pyhazards.model_catalog import (
    load_model_cards,
    model_catalog_alignment_issues,
    render_api_page,
    render_model_page,
)


def test_model_catalog_aligns_with_registry() -> None:
    cards = load_model_cards()
    assert not model_catalog_alignment_issues(cards)


def test_model_page_lists_generated_hazard_sections() -> None:
    cards = load_model_cards()
    page = render_model_page(cards)
    assert "Earthquake" in page
    assert "Flood" in page
    assert "Wildfire" in page
    assert ":doc:`WaveCastNet <modules/models_wavecastnet>`" in page
    assert ":doc:`Wildfire Mamba <modules/models_wildfire_mamba>`" not in page


def test_hidden_models_are_omitted_from_public_catalog_pages() -> None:
    cards = load_model_cards()
    api_page = render_api_page(cards)
    assert ":doc:`Wildfire Mamba </modules/models_wildfire_mamba>`" not in api_page
