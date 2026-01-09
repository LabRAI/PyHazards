Datasets
===================

Summary
-------

PyHazards provides a unified dataset interface for hazard prediction across tabular, temporal, and raster data. Each dataset returns a DataBundle containing splits, feature specs, label specs, and metadata.

Datasets
--------------------

.. list-table::
   :widths: 18 82
   :header-rows: 0
   :class: dataset-list

   * - ``MERRA-2 (NASA GMAO)``
     - Data format: commonly distributed as NetCDF (and also accessible via NASA services); data structure: gridded arrays (tensor-like) with dynamic surface fields, vertical profile fields, and static layers; dimensions: time is the temporal index (hourly), latitude and longitude are the spatial grid axes (about 0.5 deg x 0.625 deg), and vertical variables include an additional pressure-level axis; update frequency: released with a typical latency of a few weeks after real time; tasks: provide core meteorological drivers for weather/climate forecasting (e.g., WxC-style pipelines) and support hazard prediction studies such as wildfire and hurricane by linking hazard outcomes to atmospheric and land-surface conditions; link: `MERRA-2 overview <https://gmao.gsfc.nasa.gov/gmao-products/merra-2/>`_; reference: `Gelaro et al. (2017) <https://journals.ametsoc.org/view/journals/clim/30/14/jcli-d-16-0758.1.xml>`_.
   * - ``ERA5``
     - Data format: commonly distributed as GRIB and NetCDF via Copernicus CDS; data structure: gridded arrays (tensor-like) with single-level fields and optional pressure/model-level fields; dimensions: time is the temporal index (hourly), latitude and longitude are the spatial grid axes (commonly 0.25 deg x 0.25 deg on CDS products), and vertical products include an additional level axis; update frequency: updated daily with a short-latency stream (ERA5T) and later consolidated into the final ERA5; tasks: support weather/climate prediction and benchmarking, and serve as standardized meteorological covariates for hazard prediction such as wildfire danger, heavy-rain/flooding, and hurricane-related environmental conditions; link: `ERA5 single levels <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview>`_; reference: `Hersbach et al. (2020) <https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.3803>`_.
   * - ``NOAA Flood Events (Storm Events Database)``
     - Data format: commonly distributed as tabular text (CSV) and database extracts; data structure: event-based records (not tensors) where each row is an event report with fields rather than gridded axes; dimensions/fields: time fields represent event begin/end timestamps, location fields represent reporting geography (often county/zone and coordinates where available), and impact fields represent reported damages and casualties; update frequency: updated on a rolling basis with a typical delay of about 75–90 days after the end of the data month; tasks: support flood occurrence/impact analysis, event frequency and trend studies, and supervised modeling where the target is event occurrence or impact rather than meteorological state; link: `Storm Events Database <https://www.ncei.noaa.gov/products/storm-events-database>`_; reference: `NCEI metadata record <https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00648>`_.
   * - ``FIRMS (Fire Information for Resource Management System)``
     - Data format: commonly distributed as tabular and GIS-ready formats (CSV, Shapefile/GeoJSON/KML) depending on access method; data structure: event/point detections (not tensors) where each record is a time-stamped fire/hotspot detection; dimensions/fields: time is detection timestamp, location is latitude/longitude (point geometry), and attributes include sensor/product fields such as confidence and radiative measures (product-dependent); update frequency: near-real-time publication refreshed multiple times per day, with detections typically available within hours of satellite overpass; tasks: operational wildfire monitoring/nowcasting and rapid situational awareness, and serving as event labels/targets for wildfire prediction pipelines when combined with meteorology and fuels/land layers; link: `FIRMS portal <https://firms.modaps.eosdis.nasa.gov/>`_; reference: `VIIRS active fire product paper <https://doi.org/10.1016/j.rse.2013.08.008>`_.
   * - ``MTBS (Monitoring Trends in Burn Severity)``
     - Data format: commonly distributed as GIS rasters and vectors (e.g., GeoTIFF and shapefiles/geodatabases); data structure: mapped fire perimeters and burn severity layers (not a single tensor) where products are organized by fire event and can be aggregated by year; dimensions/fields: spatial axes represent 30 m raster grids for severity layers and polygon geometries for perimeters, and time is represented by fire year/event metadata; update frequency: produced on an annual cycle as new fire-year products are completed; tasks: post-fire impact assessment, long-term wildfire regime/trend analysis, and providing training/validation layers for models that predict burn severity/extent or relate severity to meteorology, fuels, and topography; link: `MTBS portal <https://burnseverity.cr.usgs.gov/>`_; reference: `Eidenshink et al. (2007) <https://doi.org/10.4996/fireecology.0301003>`_.
   * - ``LANDFIRE Fuel (USFS / LANDFIRE)``
     - Data format: commonly distributed as GIS rasters (e.g., GeoTIFF) and supporting metadata; data structure: gridded map layers (raster matrices) describing vegetation and fuels at national scale; dimensions/fields: spatial axes represent ~30 m raster grids, time is represented by product version/year (and disturbance layers where applicable), and values encode vegetation/fuel attributes such as type, cover, height, and fuel descriptors; update frequency: released as versioned updates and commonly treated as an annual updating program in applied workflows; tasks: fuels/vegetation characterization for wildfire behavior simulation, landscape-scale wildfire risk assessment, and static covariates for wildfire occurrence/spread models paired with dynamic meteorological predictors; link: `LANDFIRE data access <https://landfire.gov/getdata.php>`_; reference: `LANDFIRE program overview <https://research.fs.usda.gov/firelab/products/dataandtools/landfire-landscape-fire-and-resource-management-planning>`_.

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
