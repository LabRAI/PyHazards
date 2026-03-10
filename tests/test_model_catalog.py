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
    assert "At a Glance" in page
    assert "Catalog by Hazard" in page
    assert "Recommended Entry Points" in page
    assert "Programmatic Use" in page
    assert ".. tab-set::" in page
    assert page.index(".. tab-item:: Wildfire") < page.index(".. tab-item:: Earthquake")
    assert ".. tab-item:: Flood" in page
    assert ".. tab-item:: Tropical Cyclone" in page
    assert ".. tab-item:: Hurricane" not in page
    assert ":doc:`DNN-LSTM-AutoEncoder <modules/models_wildfire_fpa>`" in page
    assert ":doc:`Wildfire Forecasting <modules/models_wildfire_forecasting>`" in page
    assert ":doc:`WildfireSpreadTS <modules/models_wildfirespreadts>`" in page
    assert ":doc:`ASUFM <modules/models_asufm>`" in page
    assert ":doc:`ForeFire Adapter <modules/models_forefire>`" in page
    assert ":doc:`WRF-SFIRE Adapter <modules/models_wrf_sfire>`" in page
    assert ":doc:`FireCastNet <modules/models_firecastnet>`" in page
    assert ":doc:`WaveCastNet <modules/models_wavecastnet>`" in page
    assert ":doc:`GraphCast TC Adapter <modules/models_graphcast_tc>`" in page
    assert "Wildfire Danger Prediction and Understanding with Deep Learning" in page
    assert "`Repository <https://github.com/Orion-AI-Lab/wildfire_forecasting>`_" in page
    assert page.count("Implemented Models") == 5
    assert page.count("Experimental Adapters") == 2
    assert "Core Baselines" not in page
    assert "Variants and Additional Implementations" not in page
    assert page.count(":doc:`DNN-LSTM-AutoEncoder <modules/models_wildfire_fpa>`") == 1
    assert page.count(":doc:`WaveCastNet <modules/models_wavecastnet>`") == 1
    assert ":doc:`Wildfire Mamba <modules/models_wildfire_mamba>`" not in page
    assert ":doc:`DNN <modules/models_wildfire_fpa_dnn>`" not in page
    assert ":doc:`LSTM-AutoEncoder <modules/models_wildfire_fpa_forecast>`" not in page
    assert ":doc:`LSTM <modules/models_wildfire_fpa_lstm>`" not in page


def test_hidden_models_are_omitted_from_public_catalog_pages() -> None:
    cards = load_model_cards()
    api_page = render_api_page(cards)
    assert ":doc:`Wildfire Mamba </modules/models_wildfire_mamba>`" not in api_page
    assert api_page.count("Implemented Models") == 4
    assert api_page.count("Experimental Adapters") == 1
    assert "Core Baselines" not in api_page
    assert "Variants and Additional Implementations" not in api_page
    assert ":doc:`GraphCast TC Adapter </modules/models_graphcast_tc>`" in api_page
    assert "Developer Registry Workflow" in api_page
    assert "Catalog Summary" in api_page
    assert "Hurricane" not in api_page
