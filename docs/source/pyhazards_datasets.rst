Datasets
===================

Summary
-------

PyHazards provides a unified dataset interface for hazard prediction across tabular, temporal, and raster data. Each dataset returns a DataBundle containing splits, feature specs, label specs, and metadata.

Datasets
--------------------

.. list-table::
   :widths: 15 85
   :header-rows: 0
   :class: dataset-list

   * - :doc:`merra2 <datasets/merra2>`
     - Global atmospheric reanalysis from NASA GMAO MERRA-2 (`overview <https://gmao.gsfc.nasa.gov/gmao-products/merra-2/>`_), widely used as hourly gridded meteorological drivers for hazard modeling; see `Gelaro et al. (2017) <https://journals.ametsoc.org/view/journals/clim/30/14/jcli-d-16-0758.1.xml>`_.

   * - :doc:`era5 <datasets/era5>`
     - ECMWF ERA5 reanalysis served via the `Copernicus CDS <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview>`_, providing hourly single-/pressure-level variables for benchmarks and hazard covariates; see `Hersbach et al. (2020) <https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.3803>`_.

   * - :doc:`noaa_flood <datasets/noaa_flood>`
     - Flood-related event reports from the `NOAA Storm Events Database <https://www.ncei.noaa.gov/products/storm-events-database>`_ (time, location, impacts), commonly used for event-level labeling and impact analysis.

   * - :doc:`firms <datasets/firms>`
     - Near-real-time active fire detections from `NASA FIRMS <https://firms.modaps.eosdis.nasa.gov/>`_ (MODIS/VIIRS), used for operational monitoring and as wildfire occurrence labels; see `Schroeder et al. (2014) <https://doi.org/10.1016/j.rse.2013.08.008>`_.

   * - :doc:`mtbs <datasets/mtbs>`
     - US wildfire perimeters and burn severity layers from `MTBS <https://burnseverity.cr.usgs.gov/>`_ (Landsat-derived), used for post-fire assessment and long-term regime studies; see `Eidenshink et al. (2007) <https://doi.org/10.4996/fireecology.0301003>`_.

   * - :doc:`landfire <datasets/landfire>`
     - Nationwide fuels and vegetation layers from the `USFS LANDFIRE <https://landfire.gov/>`_ program, often used as static landscape covariates for wildfire behavior and risk modeling; see `the program overview <https://research.fs.usda.gov/firelab/products/dataandtools/landfire-landscape-fire-and-resource-management-planning>`_.

   * - :doc:`wfigs <datasets/wfigs>`
     - Authoritative incident-level wildfire records from the `U.S. interagency WFIGS <https://data-nifc.opendata.arcgis.com/>`_ ecosystem (ignition, location, status, extent), commonly used as ground-truth labels for wildfire occurrence.

   * - :doc:`goesr <datasets/goesr>`
     - High-frequency geostationary multispectral imagery from the `NOAA GOES-R series <https://www.goes-r.gov/>`_, supporting continuous monitoring (e.g., smoke/thermal context) and early detection workflows when paired with fire and meteorology datasets.

Dataset inspection
------------------

PyHazards provides a built-in inspection utility that allows users to
quickly explore dataset structure and contents through a unified API.

The example below demonstrates how to inspect a daily MERRA-2 file using
the PyHazards dataset interface.

.. code-block:: bash

   python -m pyhazards.datasets.inspection 20260101


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
