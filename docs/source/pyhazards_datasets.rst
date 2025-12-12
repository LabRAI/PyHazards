Datasets
===================

Summary
-------

PyHazards provides a unified dataset interface for hazard prediction across tabular, temporal, and raster data. Each dataset returns a ``DataBundle`` containing splits, feature specs, label specs, and metadata.
Datasets
--------------------

.. list-table::
   :widths: 18 82
   :header-rows: 0
   :class: dataset-list

   * - ``MERRA-2``
     - MERRA-2 is a global weather and climate dataset created by NASA that provides long-term records of atmospheric conditions from 1980 to the present. The data are organized on a regular global grid with hourly to monthly updates and a spatial resolution of about 0.5° × 0.625°, and include variables such as temperature, wind, humidity, surface fluxes, and aerosols, making it useful for large-scale climate and hazard analysis worldwide.
   * - ``ERA5``
     - ERA5 is a global reanalysis dataset produced by ECMWF and is commonly used as a reference for weather and climate studies. It covers the period from 1950 to the present and provides hourly data on a global grid at roughly 0.25° resolution, including precipitation, temperature, wind, and land-surface variables, which allows detailed environmental and hazard modeling at a global scale.
   * - ``NOAA Flood Events (Storm Events Database)``
     - The NOAA Flood Events dataset collects reported flood events across the United States, including when and where floods occurred and the impacts they caused. The data are stored as event-based records rather than grid data, usually at county or local levels, and span many decades, making them suitable for analyzing flood patterns and risks within the U.S.
   * - ``FIRMS (Fire Information for Resource Management System)``
     - FIRMS (Fire Information for Resource Management System) provides near-real-time satellite observations of active fires around the world. The dataset consists of time-stamped fire locations detected by MODIS and VIIRS sensors at spatial resolutions between about 375 m and 1 km, and is widely used for global wildfire monitoring and analysis.
   * - ``MTBS (Monitoring Trends in Burn Severity)``
     - MTBS (Monitoring Trends in Burn Severity) is a dataset that maps wildfire burn areas and severity across the United States. It focuses on large fires since 1984 and uses Landsat satellite imagery at 30 m resolution, making it useful for studying long-term wildfire impacts and regional fire risk.
   * - ``LANDFIRE Fuel (USFS / LANDFIRE)``
     - LANDFIRE provides fuel and vegetation information across the United States to support wildfire modeling and risk assessment. The data are distributed as raster layers at about 30 m resolution and describe fuel models, canopy structure, and vegetation types, helping researchers and practitioners understand and simulate wildfire behavior at landscape scales.

Core classes
------------

- ``Dataset``: base class to implement ``_load()`` and return a ``DataBundle``.
- ``DataBundle``: holds named ``DataSplit`` objects, plus ``feature_spec`` and ``label_spec``.
- ``FeatureSpec`` / ``LabelSpec``: describe inputs/targets to simplify model construction.
- ``register_dataset`` / ``load_dataset``: lightweight registry for discovering datasets by name.

Example skeleton
----------------

.. code-block:: python

    import torch
    from pyhazards.datasets import (
        DataBundle, DataSplit, Dataset, FeatureSpec, LabelSpec, register_dataset
    )

    class MyHazardDataset(Dataset):
        name = "my_hazard"

        def _load(self):
            x = torch.randn(1000, 16)
            y = torch.randint(0, 2, (1000,))
            splits = {
                "train": DataSplit(x[:800], y[:800]),
                "val": DataSplit(x[800:900], y[800:900]),
                "test": DataSplit(x[900:], y[900:]),
            }
            return DataBundle(
                splits=splits,
                feature_spec=FeatureSpec(input_dim=16, description="example features"),
                label_spec=LabelSpec(num_targets=2, task_type="classification"),
            )

    register_dataset(MyHazardDataset.name, MyHazardDataset)
