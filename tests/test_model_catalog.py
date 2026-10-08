import importlib.util

import pytest
import torch
from pydantic import ValidationError

from pyhazards.model_catalog import (
    SmokeFitSpec,
    _make_fit_targets,
    card_by_registry_name,
    load_model_cards,
    model_catalog_alignment_issues,
    render_api_page,
    render_model_page,
    run_smoke_test,
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
    assert ":doc:`WildfireSpreadTS Baselines <modules/models_wildfirespreadts>`" in page
    assert ":doc:`U-TAE <modules/models_utae>`" in page
    assert ":doc:`SegFormer <modules/models_segformer>`" in page
    assert ":doc:`Prithvi-EO-2.0 BurnScars <modules/models_prithvi_burnscars>`" in page
    assert ":doc:`Prithvi-EO-2.0-TL <modules/models_prithvi_eo_2_tl>`" in page
    assert ":doc:`ASUFM <modules/models_asufm>`" in page
    assert ":doc:`Swin-Unet <modules/models_swin_unet>`" in page
    assert ":doc:`Earthfarseer <modules/models_earthfarseer>`" in page
    assert ":doc:`Rainformer <modules/models_rainformer>`" in page
    # ForeFire and WRF-SFIRE are external simulators (pyhazards.simulators, datasets/wrf_sfire), not models.
    assert "models_forefire" not in page
    assert "models_wrf_sfire" not in page
    assert ":doc:`FireCastNet <modules/models_firecastnet>`" in page
    assert ":doc:`WaveCastNet <modules/models_wavecastnet>`" in page
    assert ":doc:`GraphCast TC Adapter <modules/models_graphcast_tc>`" in page
    assert "Wildfire Danger Prediction and Understanding With Deep Learning" in page
    assert "`Repository <https://github.com/Orion-AI-Lab/wildfire_forecasting>`_" in page
    assert page.count("Implemented Models") == 5
    assert page.count("Experimental Adapters") == 3  # Flood (hydrographnet, floodcast, urbanfloodcast) and TC
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
    assert api_page.count("Experimental Adapters") == 2
    assert "Core Baselines" not in api_page
    assert "Variants and Additional Implementations" not in api_page
    assert ":doc:`GraphCast TC Adapter </modules/models_graphcast_tc>`" in api_page
    assert "Developer Registry Workflow" in api_page
    assert "Catalog Summary" in api_page
    assert "Hurricane" not in api_page


def test_smoke_test_fits_estimator_models_first() -> None:
    card = card_by_registry_name(load_model_cards())["random_forest"]
    assert card.smoke_test.fit is not None
    result = run_smoke_test(card)
    assert result["ok"] and result["skipped"] is None
    assert result["actual_shape"] == [4, 2]


def test_smoke_test_skips_missing_optional_packages() -> None:
    cards = card_by_registry_name(load_model_cards())
    card = cards["xgboost"].model_copy(deep=True)
    card.smoke_test.requires = ["pyhazards_no_such_package"]
    result = run_smoke_test(card)
    assert result["ok"] and "pyhazards_no_such_package" in result["skipped"]
    assert result["actual_shape"] is None and result["expected_shape"] == [4, 2]

    xgboost_result = run_smoke_test(cards["xgboost"])
    assert xgboost_result["ok"]
    assert (xgboost_result["skipped"] is None) == (importlib.util.find_spec("xgboost") is not None)


def test_smoke_fit_spec_validation_and_targets() -> None:
    spec = SmokeFitSpec.model_validate(
        {"inputs": {"shape": [6, 3]}, "targets": {"shape": [6], "dtype": "int64"}, "num_classes": 3}
    )
    torch.manual_seed(0)
    targets = _make_fit_targets(spec)
    assert targets.dtype == torch.int64 and targets.shape == (6,)
    assert set(targets.tolist()) == {0, 1, 2}
    regression = SmokeFitSpec.model_validate({"inputs": {"shape": [4, 3]}, "targets": {"shape": [4, 2]}})
    assert _make_fit_targets(regression).dtype == torch.float32
    with pytest.raises(ValidationError):
        SmokeFitSpec.model_validate({"inputs": {"shape": [6, 3]}, "targets": {"shape": [5]}})
    with pytest.raises(ValidationError):
        SmokeFitSpec.model_validate(
            {"inputs": {"shape": [2, 3]}, "targets": {"shape": [2], "dtype": "int64"}, "num_classes": 3}
        )
