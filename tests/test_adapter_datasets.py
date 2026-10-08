from pyhazards.datasets import available_datasets, load_dataset


def test_named_adapter_datasets_are_registered_and_loadable():
    expected = {
        "seisbench_waveforms",
        "pick_benchmark_waveforms",
        "aefa_forecast",
        "caravan_streamflow",
        "waterbench_streamflow",
        "hydrobench_streamflow",
        "floodcastbench_inundation",
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
