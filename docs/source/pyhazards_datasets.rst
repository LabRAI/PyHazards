Datasets
===================

Summary
-------

PyHazards maintains a curated catalog of commonly used hazard datasets and provides
dataset-specific utilities for **download / preprocessing / inspection / visualization**.

Each dataset page describes: (1) what the dataset is, (2) how to obtain it, and (3) how to
quickly validate local data files via an inspection entrypoint (when available).


Datasets
--------------------

Click a module name to open its merged detail page (dataset summary + short usage example).

.. list-table::
   :widths: 15 85
   :header-rows: 1
   :class: dataset-list

   * - Module
     - Description

   * - :doc:`merra2 <modules/datasets_merra2_inspection>`
     - Global atmospheric reanalysis from NASA GMAO MERRA-2 (`overview <https://gmao.gsfc.nasa.gov/gmao-products/merra-2/>`_), widely used as hourly gridded meteorological drivers for hazard modeling; see `Gelaro et al. (2017) <https://journals.ametsoc.org/view/journals/clim/30/14/jcli-d-16-0758.1.xml>`_. Dataset page: :doc:`merra2 <datasets/merra2>`.

   * - :doc:`era5 <modules/datasets_era5_inspection>`
     - ECMWF ERA5 reanalysis served via the `Copernicus CDS <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview>`_, providing hourly single-/pressure-level variables for benchmarks and hazard covariates; see `Hersbach et al. (2020) <https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.3803>`_. Dataset page: :doc:`era5 <datasets/era5>`.

   * - :doc:`noaa_flood <modules/datasets_noaa_flood_inspection>`
     - Flood-related event reports from the `NOAA Storm Events Database <https://www.ncei.noaa.gov/products/storm-events-database>`_ (time, location, impacts), commonly used for event-level labeling and impact analysis. Dataset page: :doc:`noaa_flood <datasets/noaa_flood>`.

   * - :doc:`firms <modules/datasets_firms_inspection>`
     - Near-real-time active fire detections from `NASA FIRMS <https://firms.modaps.eosdis.nasa.gov/>`_ (MODIS/VIIRS), used for operational monitoring and as wildfire occurrence labels; see `Schroeder et al. (2014) <https://doi.org/10.1016/j.rse.2013.08.008>`_. Dataset page: :doc:`firms <datasets/firms>`.

   * - :doc:`mtbs <modules/datasets_mtbs_inspection>`
     - US wildfire perimeters and burn severity layers from `MTBS <https://burnseverity.cr.usgs.gov/>`_ (Landsat-derived), used for post-fire assessment and long-term regime studies; see `Eidenshink et al. (2007) <https://doi.org/10.4996/fireecology.0301003>`_. Dataset page: :doc:`mtbs <datasets/mtbs>`.

   * - :doc:`landfire <modules/datasets_landfire_inspection>`
     - Nationwide fuels and vegetation layers from the `USFS LANDFIRE <https://landfire.gov/>`_ program, often used as static landscape covariates for wildfire behavior and risk modeling; see `the program overview <https://research.fs.usda.gov/firelab/products/dataandtools/landfire-landscape-fire-and-resource-management-planning>`_. Dataset page: :doc:`landfire <datasets/landfire>`.

   * - :doc:`wfigs <modules/datasets_wfigs_inspection>`
     - Authoritative incident-level wildfire records from the `U.S. interagency WFIGS <https://data-nifc.opendata.arcgis.com/>`_ ecosystem (ignition, location, status, extent), commonly used as ground-truth labels for wildfire occurrence. Dataset page: :doc:`wfigs <datasets/wfigs>`.

   * - :doc:`goesr <modules/datasets_goesr_inspection>`
     - High-frequency geostationary multispectral imagery from the `NOAA GOES-R series <https://www.goes-r.gov/>`_, supporting continuous monitoring (e.g., smoke/thermal context) and early detection workflows when paired with fire and meteorology datasets. Dataset page: :doc:`goesr <datasets/goesr>`.


Dataset inspection
------------------

PyHazards provides dataset inspection entrypoints to quickly validate local files and produce
basic summaries/plots.

Currently implemented:

- **MERRA-2 (merra2)**: one-shot pipeline to **download raw MERRA-2 → merge SFC+PRES → inspect → save plots/tables**.

.. code-block:: bash

   # One command: download (if needed) -> merge -> inspect -> save outputs
   python -m pyhazards.datasets.inspection 20260101


Notes (MERRA-2)
~~~~~~~~~~~~~~~

- Download requires Earthdata credentials via environment variables::

     export EARTHDATA_USERNAME="YOUR_USERNAME"
     export EARTHDATA_PASSWORD="YOUR_PASSWORD"

- Date formats accepted: ``YYYYMMDD`` (e.g., ``20260101``) or ISO ``YYYY-MM-DD``.
- Optional flags commonly used:
  - ``--outdir outputs`` (default: ``outputs`` under repo root)
  - ``--skip-download`` / ``--skip-merge`` for re-running on existing files
  - ``--force-download`` to re-fetch raw files
  - ``--var T2M`` to choose the plotted surface variable (default: ``T2M``)


Example skeleton
----------------

A "nice" skeleton should make it explicit **what data you load** and how it flows into
**inspection/visualization**.

Below is the recommended pattern: set ``data`` to a dataset name (e.g., ``"merra2"`` or ``"mtbs"``)
and run the dataset's inspection entrypoint accordingly.

.. code-block:: python

   import subprocess

   # 1) Choose what dataset you want to load/inspect
   data = "merra2"   # e.g., "merra2", "mtbs", "era5", "firms", "landfire", "wfigs", "goesr" (use accordingly)

   # 2) Choose the dataset key (identifier)
   #    - For MERRA-2, the key is a daily date: "YYYYMMDD" (e.g., "20260101")
   #    - For other datasets (e.g., MTBS), the key could be an event/scene id (to be defined per dataset)
   key = "20260101"

   # 3) Run the inspection pipeline (download/preprocess if needed -> inspect -> visualize -> save outputs)
   if data == "merra2":
       cmd = [
           "python", "-m", "pyhazards.datasets.inspection",
           key,
           "--var", "T2M",           # change variable to plot (e.g., QV2M)
           "--outdir", "outputs",    # output folder under repo root by default
       ]
   else:
       # Convention for other datasets:
       # provide a dataset-specific inspection entrypoint:
       #   python -m pyhazards.datasets.<dataset>.inspection <key> ...
       cmd = ["python", "-m", f"pyhazards.datasets.{data}.inspection", key, "--outdir", "outputs"]

   subprocess.run(cmd, check=True)

   # 4) After running, check outputs/ for saved artifacts (tables + plots).
   #    Example (MERRA-2): CSV tables for variable inventory + a PDF plot for the selected surface variable.


Inspection entrypoints (convention for all datasets)
----------------------------------------------------

Each dataset should expose a minimal inspection entrypoint that supports the same user experience:

- **Input**: a dataset identifier (``key``) such as a date/event id.
- **Work**: download/prepare (if needed) → open files → summarize → visualize.
- **Output**: saved artifacts under ``outputs/`` (tables + figures).

Recommended CLI shape (dataset-specific):

.. code-block:: bash

   # Example convention (to be implemented per dataset):
   python -m pyhazards.datasets.<dataset>.inspection <key> --outdir outputs


Developer note
--------------

If you plan to add inspection for a new dataset, mirror the MERRA-2 inspection pattern:

1) parse CLI args (key + outdir + skip/force flags),
2) materialize required local files (download/preprocess),
3) open files and print structure/statistics,
4) generate at least one saved visualization to ``outputs/``.

.. toctree::
   :maxdepth: 1
   :hidden:

   modules/datasets_merra2_inspection
   modules/datasets_era5_inspection
   modules/datasets_noaa_flood_inspection
   modules/datasets_firms_inspection
   modules/datasets_mtbs_inspection
   modules/datasets_landfire_inspection
   modules/datasets_wfigs_inspection
   modules/datasets_goesr_inspection
