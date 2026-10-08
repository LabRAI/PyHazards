from pyhazards.datasets import available_datasets, load_dataset


def test_named_adapter_datasets_are_registered_and_loadable():
    expected = {
        "earthquake_waveforms_synthetic",
        "earthquake_wavefield_synthetic",
        "flood_streamflow_synthetic",
        "flood_mesh_synthetic",
        "flood_inundation_synthetic",
        "tc_tracks_synthetic",
        "ships_xu2021_synthetic",
        "safnet_cma_era_interim_synthetic",
        "tropicyclonenet_dataset_synthetic",
        "wildfire_spread_temporal_synthetic",
        "wildfire_danger_synthetic",
    }
    assert expected.issubset(set(available_datasets()))

    for name in sorted(expected):
        bundle = load_dataset(name, micro=True).load()
        assert bundle.splits["test"].inputs is not None
        assert bundle.metadata.get("source_dataset", name) == name or bundle.metadata.get("dataset") == name


def test_former_wavefield_name_is_a_deprecated_alias():
    bundle = load_dataset("earthquake_forecast_synthetic", micro=True).load()
    assert bundle.metadata["dataset"] == "earthquake_wavefield_synthetic" and bundle.metadata["synthetic"]


def test_real_earthquake_reader_is_registered_but_needs_data():
    import pytest

    assert "seisbench_waveforms" in available_datasets()
    for removed in ("pick_benchmark_waveforms", "aefa_forecast", "earthquake_waveforms"):
        assert removed not in available_datasets()
    with pytest.raises(ValueError, match="path="):
        load_dataset("seisbench_waveforms", micro=True).load()
