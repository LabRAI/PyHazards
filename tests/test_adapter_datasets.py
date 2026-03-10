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
        "ibtracs_tracks",
        "tcbench_alpha",
        "tropicyclonenet_dataset",
        "wildfire_spread_temporal_synthetic",
    }
    assert expected.issubset(set(available_datasets()))

    for name in sorted(expected):
        bundle = load_dataset(name, micro=True).load()
        assert bundle.splits["test"].inputs is not None
        assert bundle.metadata.get("source_dataset", name) == name or bundle.metadata.get("dataset") == name
