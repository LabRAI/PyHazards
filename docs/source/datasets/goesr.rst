GOES-R
======

High-frequency geostationary satellite observations from NOAA’s GOES-R series, providing continuous
multispectral imagery for monitoring atmospheric and surface processes.

.. admonition:: At-a-glance (Quick Facts)
   :class: note

   .. list-table::
      :widths: 26 74
      :stub-columns: 1

      * - Provider
        - NOAA GOES-R Program
      * - Payload (core)
        - ABI (Advanced Baseline Imager) multispectral imagery
      * - Spatial coverage
        - Western Hemisphere / Americas (geostationary view)
      * - Spatial resolution
        - ~0.5–2 km (band-dependent)
      * - Temporal resolution (typical)
        - **Mode 6 default:** Full Disk 10 min; CONUS 5 min; Mesoscale 1 min
      * - **Data update frequency**
        - **Continuous ingest; new files are added as soon as available.**
          (Scan cadence depends on sector/mode; see above.)
      * - Latency (depends on access route)
        - Cloud/Open-Data: near real time; CLASS subscription can be ~30 min–2 hr after observation
      * - Format
        - NetCDF (ABI Level 1b / Level 2+ products)
      * - Projection / CRS
        - ABI fixed grid (geostationary projection); lat–lon remaps are available for some workflows
      * - Typical role in PyHazards
        - Rapid-refresh imagery for smoke/fire evolution, ignition monitoring, and situational awareness

----

Overview
--------

**GOES-R** refers to NOAA’s current-generation geostationary operational environmental satellites.
The series carries the Advanced Baseline Imager (ABI), which provides rapid-refresh, multispectral
observations over the Americas.

GOES-R data are widely used for real-time monitoring of weather systems, smoke and fire evolution, and
early wildfire detection, particularly where high temporal resolution is critical.

----

Stats
-----

.. list-table::
   :header-rows: 1
   :widths: 22 22 28 28

   * - Spatial
     - Temporal
     - Refresh / update cadence
     - Representation
   * - 0.5–2 km (band-dependent)
     - 1–10 min (sector/mode)
     - Continuous ingest; files appear as soon as available
     - Raster imagery time series (ABI fixed grid)

----

Data Characteristics
--------------------

- **Spatial coverage:** Western Hemisphere (geostationary view)
- **Spatial resolution:** ~0.5–2 km (band-dependent)
- **Temporal resolution:** 1–15 minutes (mode/sector-dependent; typical default Mode 6 cadence below)
- **Typical scan cadence (Mode 6 default):**
  - Full Disk: every 10 minutes
  - CONUS: every 5 minutes
  - Mesoscale: every 1 minute
- **Data structure:** Gridded satellite imagery (raster time series)
- **Data format:** NetCDF
- **Coordinate system:** ABI fixed grid (geostationary projection; lat–lon remaps can be generated)

.. note::
   “Data update frequency” in GOES-R is best interpreted as **(1) observation refresh cadence**
   (sector/mode) + **(2) distribution latency** (varies by access channel). For near-real-time pipelines,
   cloud/open-data mirrors typically publish granules as soon as they are available; some archival/subscription
   services may have longer delays.

----

Variables
---------

GOES-R ABI products include multiple spectral bands and derived fields, such as:

- Visible and infrared radiance and brightness temperature
- Cloud and atmospheric motion indicators
- Fire-related thermal anomalies and contextual imagery
- Smoke, aerosol, and atmospheric structure diagnostics

----

Typical Use Cases
-----------------

- Early detection and monitoring of wildfire ignition and growth
- Analysis of fire and smoke evolution at high temporal resolution
- Integration with ground-truth fire records and reanalysis data
- Real-time and near-real-time hazard situational awareness

----

Access
------

GOES-R data are distributed by NOAA through multiple access points:

- `GOES-R Program <https://www.goes-r.gov/>`_
- `NOAA Open Data Dissemination <https://www.noaa.gov/information-technology/open-data-dissemination>`_

----

Reference
---------

Schmit, T. J., Griffith, P., Gunshor, M. M., *et al.* (2017).
*A closer look at the ABI on the GOES-R series*.
Bulletin of the American Meteorological Society, 98(4), 681–698.
`https://doi.org/10.1175/BAMS-D-15-00230.1 <https://doi.org/10.1175/BAMS-D-15-00230.1>`_
