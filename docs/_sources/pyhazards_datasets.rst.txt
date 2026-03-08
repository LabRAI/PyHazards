Datasets
===================

Overview
--------

Use this page to browse the built-in dataset references and the inspection
commands that help you validate local files before training.

Each dataset page answers three practical questions:

1. What is this dataset?
2. How do I obtain or prepare it?
3. What command should I run to check that my files are usable?

Dataset Catalog
---------------

Select a dataset name below to open its reference page.

.. list-table::
   :widths: 30 70
   :header-rows: 1
   :class: dataset-list

   * - Dataset
     - Description

   * - :doc:`MERRA-2 <datasets/merra2>`
     - Global atmospheric reanalysis from NASA GMAO MERRA-2 (`overview <https://gmao.gsfc.nasa.gov/gmao-products/merra-2/>`_), widely used as hourly gridded meteorological drivers for hazard modeling; see `Gelaro et al. (2017) <https://journals.ametsoc.org/view/journals/clim/30/14/jcli-d-16-0758.1.xml>`_.

   * - :doc:`ERA5 <datasets/era5>`
     - ECMWF ERA5 reanalysis served via the `Copernicus CDS <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview>`_, providing hourly single-/pressure-level variables for benchmarks and hazard covariates; see `Hersbach et al. (2020) <https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.3803>`_.

   * - :doc:`NOAA Flood Events <datasets/noaa_flood>`
     - Flood-related event reports from the `NOAA Storm Events Database <https://www.ncei.noaa.gov/products/storm-events-database>`_ (time, location, impacts), commonly used for event-level labeling and impact analysis.

   * - :doc:`FIRMS <datasets/firms>`
     - Near-real-time active fire detections from `NASA FIRMS <https://firms.modaps.eosdis.nasa.gov/>`_ (MODIS/VIIRS), used for operational monitoring and as wildfire occurrence labels; see `Schroeder et al. (2014) <https://doi.org/10.1016/j.rse.2013.08.008>`_.

   * - :doc:`MTBS <datasets/mtbs>`
     - US wildfire perimeters and burn severity layers from `MTBS <https://burnseverity.cr.usgs.gov/>`_ (Landsat-derived), used for post-fire assessment and long-term regime studies; see `Eidenshink et al. (2007) <https://doi.org/10.4996/fireecology.0301003>`_.

   * - :doc:`LANDFIRE <datasets/landfire>`
     - Nationwide fuels and vegetation layers from the `USFS LANDFIRE <https://landfire.gov/>`_ program, often used as static landscape covariates for wildfire behavior and risk modeling; see `the program overview <https://research.fs.usda.gov/firelab/products/dataandtools/landfire-landscape-fire-and-resource-management-planning>`_.

   * - :doc:`WFIGS <datasets/wfigs>`
     - Authoritative incident-level wildfire records from the `U.S. interagency WFIGS <https://data-nifc.opendata.arcgis.com/>`_ ecosystem (ignition, location, status, extent), commonly used as ground-truth labels for wildfire occurrence.

   * - :doc:`FPA-FOD Tabular <datasets/fpa_fod_tabular>`
     - Incident-level tabular dataset derived from FPA-FOD wildfire records for cause or grouped size-class classification, with support for user-provided sqlite/CSV/Parquet inputs and a deterministic micro mode.

   * - :doc:`FPA-FOD Weekly <datasets/fpa_fod_weekly>`
     - Weekly aggregated FPA-FOD sequences for next-week wildfire count forecasting by size group, with chronological splits and optional seasonal time features.

   * - :doc:`GOES-R <datasets/goesr>`
     - High-frequency geostationary multispectral imagery from the `NOAA GOES-R series <https://www.goes-r.gov/>`_, supporting continuous monitoring (e.g., smoke/thermal context) and early detection workflows when paired with fire and meteorology datasets.


Dataset inspection
------------------

PyHazards provides inspection entrypoints so you can validate local files and
produce basic summaries before writing a training loop.

Current end-to-end example:

- **MERRA-2 (merra2)**: download raw files if needed, merge the required
  products, inspect the result, and save outputs.

.. code-block:: bash

   python -m pyhazards.datasets.inspection 20260101


Notes (MERRA-2)
~~~~~~~~~~~~~~~

- Download requires Earthdata credentials:

  .. code-block:: bash

      export EARTHDATA_USERNAME="YOUR_USERNAME"
      export EARTHDATA_PASSWORD="YOUR_PASSWORD"

- Date formats accepted: ``YYYYMMDD`` (e.g., ``20260101``) or ISO ``YYYY-MM-DD``.
- Optional flags commonly used:
  - ``--outdir outputs`` (default: ``outputs`` under repo root)
  - ``--skip-download`` / ``--skip-merge`` for re-running on existing files
  - ``--force-download`` to re-fetch raw files
  - ``--var T2M`` to choose the plotted surface variable (default: ``T2M``)


Simple Dataloader Helper
------------------------

.. literalinclude:: ../../pyhazards/datasets/dataloader/README.md
   :language: markdown


Minimal Inspection Workflow
---------------------------

Use the following pattern when you want to script dataset inspection in a
repeatable way.

.. code-block:: python

   import subprocess

   data = "merra2"
   key = "20260101"

   if data == "merra2":
       cmd = [
           "python", "-m", "pyhazards.datasets.inspection",
           key,
           "--var", "T2M",
           "--outdir", "outputs",
       ]
   else:
       cmd = ["python", "-m", f"pyhazards.datasets.{data}.inspection", key, "--outdir", "outputs"]

   subprocess.run(cmd, check=True)

Inspection CLI Convention
-------------------------

Inspection entrypoints should follow the same basic shape across datasets:

- **Input**: a dataset identifier such as a date or event id.
- **Work**: prepare local files if needed, inspect them, and save lightweight
  artifacts.
- **Output**: figures or tables under ``outputs/``.

Recommended dataset-specific CLI:

.. code-block:: bash

   python -m pyhazards.datasets.<dataset>.inspection <key> --outdir outputs


For Contributors
----------------

If you add a new dataset inspection entrypoint, mirror the existing pattern:

1) parse CLI args (key + outdir + skip/force flags),
2) materialize required local files (download/preprocess),
3) open files and print structure/statistics,
4) generate at least one saved visualization to ``outputs/``.

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
   datasets/fpa_fod_tabular
   datasets/fpa_fod_weekly
   datasets/goesr
