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
     - Global atmospheric reanalysis (NASA GMAO) with hourly gridded fields used as meteorological drivers for hazard modeling (e.g., wildfire, hurricane). Reference: `Gelaro et al. (2017) <https://journals.ametsoc.org/view/journals/clim/30/14/jcli-d-16-0758.1.xml>`_.

   * - :doc:`era5 <datasets/era5>`
     - ECMWF reanalysis providing hourly single-level and pressure-level variables via Copernicus CDS; widely used as standardized covariates for weather/climate and hazard prediction benchmarks. Reference: `Hersbach et al. (2020) <https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.3803>`_.

   * - :doc:`noaa_flood <datasets/noaa_flood>`
     - NOAA Storm Events Database records flood-related event reports (time, location, impacts), commonly used for event-level labeling and flood occurrence/impact analysis. Reference: `NOAA NCEI (C00648) <https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00648>`_.

   * - :doc:`firms <datasets/firms>`
     - Near-real-time satellite active fire detections (MODIS/VIIRS) used for operational monitoring and as wildfire occurrence labels in prediction pipelines. Reference: `Giglio et al. (2013) <https://doi.org/10.1016/j.rse.2013.08.008>`_.

   * - :doc:`mtbs <datasets/mtbs>`
     - US wildfire perimeters and burn severity products (Landsat-derived), widely used for post-fire assessment and long-term wildfire regime analysis. Reference: `Eidenshink et al. (2007) <https://doi.org/10.4996/fireecology.0301003>`_.

   * - :doc:`landfire <datasets/landfire>`
     - US vegetation and fuels layers (e.g., fuel models, structure, cover) used as static landscape covariates for wildfire behavior and risk modeling. Reference: `LANDFIRE Program <https://research.fs.usda.gov/firelab/products/dataandtools/landfire-landscape-fire-and-resource-management-planning>`_.

   * - :doc:`<datasets/wfigs>`
     - Authoritative incident-level wildfire records maintained by U.S. interagency fire management systems, describing ignition time, location, status, and extent of real wildfire events. Commonly used as ground-truth labels for wildfire occurrence and for validating satellite-based fire detections. Reference: `WFIGS portal <https://data-nifc.opendata.arcgis.com/>`_.

   * - :doc:`<datasets/goesr>`
     - High-frequency geostationary satellite observations from the NOAA GOES-R series, providing multispectral imagery for continuous monitoring of atmospheric and surface conditions. Frequently used for early wildfire detection, fire evolution analysis, and integration with fire and meteorological datasets. Reference: `GOES-R Program <https://www.goes-r.gov/>`_.

.. toctree::
   :maxdepth: 1
   :hidden:

   datasets/merra2
   datasets/era5
   datasets/noaa_flood
   datasets/firms
   datasets/mtbs
   datasets/landfire
   datasets/wfigs
   datasets/goesr

Dataset inspection
------------------

A short, step-by-step example to inspect and visualize daily MERRA-2 NetCDF files.

.. topic:: 1) Setup (imports the data)

   This block imports the dependencies used throughout the inspection workflow.

   .. code-block:: python

      import os
      from pathlib import Path
      from datetime import date

      import numpy as np
      import pandas as pd
      import xarray as xr
      import matplotlib.pyplot as plt
      from IPython.display import display

.. topic:: 2) Config (paths + date + filename patterns)

   Set the root directory and choose a test day. The code assumes one file per day.

   .. code-block:: python

      ROOT = Path("/home/runyang/WxC/Prithvi-WxC/data/merra2")

      DATE_START = date(2024, 1, 1)
      DATE_END   = date(2025, 10, 31)
      TEST_DAY = DATE_START

      PATTERN_SFC = "MERRA2_sfc_{yyyymmdd}.nc"
      PATTERN_PRES = "MERRA_pres_{yyyymmdd}.nc"

      def yyyymmdd(d: date) -> str:
         return d.strftime('%Y%m%d')

      def build_path(kind: str, d: date) -> Path:
         if kind.lower() in ['sfc', 'surface']:
            return ROOT / PATTERN_SFC.format(yyyymmdd=yyyymmdd(d))
         if kind.lower() in ['pres', 'pressure']:
            return ROOT / PATTERN_PRES.format(yyyymmdd=yyyymmdd(d))
         raise ValueError("kind must be 'sfc' or 'pres'")

      build_path('sfc', TEST_DAY), build_path('pres', TEST_DAY)

.. topic:: 3) Load helpers

   Use ``xarray.open_dataset`` so you can work with named dimensions and variables.

   .. code-block:: python

      def open_merra(kind: str, d: date, *, engine: str | None = None, chunks=None) -> xr.Dataset:
          """Open one daily MERRA2 file as an xarray Dataset."""
          path = build_path(kind, d)
          if not path.exists():
              raise FileNotFoundError(f"Missing file: {path}")
      
          # engine=None lets xarray pick; you can set engine='netcdf4' or 'h5netcdf' if needed.
          ds = xr.open_dataset(path, engine=engine, chunks=chunks)
          return ds
      
      def list_vars(ds: xr.Dataset, max_show: int = 60) -> pd.DataFrame:
          rows = []
          for name, da in ds.data_vars.items():
              rows.append({
                  'var': name,
                  'dims': str(da.dims),
                  'shape': str(tuple(da.shape)),
                  'dtype': str(da.dtype),
              })
          df = pd.DataFrame(rows).sort_values('var').reset_index(drop=True)
          return df.head(max_show) if len(df) > max_show else df
      
      def inspect_ds(ds: xr.Dataset, name: str = 'dataset', max_vars: int = 60):
          print(f"=== {name} ===")
          print('dims:', dict(ds.dims))
          print('coords:', list(ds.coords))
          print('n_vars:', len(ds.data_vars))
          display(list_vars(ds, max_show=max_vars))
      
      def summarize_da(da: xr.DataArray, *, load: bool = False) -> pd.Series:
          """Global numeric summary for a DataArray."""
          x = da
          if load:
              x = x.load()
          # Works for dask-backed arrays too
          s = xr.Dataset({
              'min': x.min(skipna=True),
              'max': x.max(skipna=True),
              'mean': x.mean(skipna=True),
              'std': x.std(skipna=True),
          }).compute()
          return pd.Series({k: float(s[k].values) for k in s.data_vars})

.. topic:: 4) Load + quick inspect (dims, coords, basic stats)

   Load both surface and pressure-level files and print basic metadata.

   .. code-block:: python

      ds_sfc  = open_merra('sfc',  TEST_DAY)
      ds_pres = open_merra('pres', TEST_DAY)
      
      inspect_ds(ds_sfc,  'SFC (one day)')
      inspect_ds(ds_pres, 'PRES (one day)')

.. topic:: 5) Variable-level inspect

   Pick a variable (e.g., `T2M`) and compute global statistics.

   .. code-block:: python

      VAR = 'T2M'  # change if your file uses a different naming
      
      if VAR not in ds_sfc:
          raise KeyError(f"{VAR} not found in ds_sfc. Pick one from the table above.")
      
      da = ds_sfc[VAR]
      print('dims:', da.dims)
      print('shape:', da.shape)
      summarize_da(da)      

.. topic:: 6) Plot a lat-lon map for a variable

   .. code-block:: python

      var = "T2M"
      t = 0
      Z = ds_sfc[var].isel(time=t).values
      plt.contourf(ds_sfc["lon"], ds_sfc["lat"], Z, 100)
      plt.gca().set_aspect("equal")
      plt.title(f"{var} (t={t})")
      plt.show()

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
