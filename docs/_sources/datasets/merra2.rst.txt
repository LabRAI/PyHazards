MERRA-2
=======

Global atmospheric reanalysis produced by NASA Global Modeling and Assimilation Office (GMAO),
providing physically consistent meteorological fields for weather, climate, and hazard modeling.

.. admonition:: At-a-glance (Quick Facts)
   :class: note

   .. list-table::
      :widths: 26 74
      :stub-columns: 1

      * - Producer
        - NASA GMAO (GEOS)
      * - Spatial coverage
        - Global
      * - Spatial resolution
        - ~0.5° (lat) × 0.625° (lon)
      * - Temporal resolution
        - Hourly (also 3-hourly / daily / monthly products)
      * - **Data update frequency**
        - **Published monthly; typically available ~2–3 weeks after month end.**
      * - Period of record
        - 1980–present
      * - Format
        - NetCDF4
      * - Grid / CRS
        - Regular latitude–longitude grid
      * - Typical role in PyHazards
        - Core meteorological driver (forcing + covariates)

----

Overview
--------

**MERRA-2 (Modern-Era Retrospective analysis for Research and Applications, Version 2)** is a global
atmospheric reanalysis that assimilates satellite and conventional observations into a numerical
weather prediction system to produce gridded, time-continuous estimates of the atmospheric state.

It is widely used as a meteorological backbone for climate analysis and hazard prediction pipelines,
including wildfire, hurricane, drought, and extreme heat studies.

----

Stats
-----

.. list-table::
   :header-rows: 1
   :widths: 22 22 22 34

   * - Spatial
     - Temporal
     - Update cadence
     - Coverage
   * - 0.5° × 0.625°
     - Hourly
     - Monthly (+ ~2–3 weeks latency)
     - Global, 1980–present

----

Data Characteristics
--------------------

- **Spatial coverage:** Global
- **Spatial resolution:** ~0.5° latitude × 0.625° longitude
- **Temporal resolution:** Hourly (with derived 3-hourly/daily/monthly products)
- **Vertical structure:** Surface fields and multi-level pressure profiles
- **Data format:** NetCDF4
- **Coordinate system:** Regular latitude–longitude grid
- **Data update frequency:** Monthly publication; new data typically appear ~2–3 weeks after month end
  (common operational latency for standard MERRA-2 streams)

----

Variables
---------

MERRA-2 provides a broad set of atmospheric and land-surface variables, including but not limited to:

- Near-surface meteorology (2 m temperature, humidity, wind)
- Surface energy fluxes and precipitation
- Atmospheric pressure-level variables (temperature, winds, geopotential height)
- Land-surface states (soil moisture, skin temperature)

----

Typical Use Cases
-----------------

- Meteorological forcing for wildfire and natural hazard prediction models
- Climate diagnostics and long-term trend analysis
- Environmental covariates for spatiotemporal machine learning models
- Benchmark reanalysis input for weather–climate pipelines (e.g., WxC-style workflows)

----

Access
------

MERRA-2 data can be accessed via NASA Earthdata / GES DISC services:

- `MERRA-2 overview (GMAO) <https://gmao.gsfc.nasa.gov/gmao-products/merra-2/>`_
- `NASA Earthdata <https://earthdata.nasa.gov/>`_

.. note::
   For standard MERRA-2 streams, operational latency is commonly documented as
   ~2–3 weeks after the end of each month (monthly publication cadence).

----

Reference
---------

Gelaro, R., McCarty, W., Suárez, M. J., *et al.* (2017).
*The Modern-Era Retrospective Analysis for Research and Applications, Version 2 (MERRA-2)*.
Journal of Climate, 30(14), 5419–5454.
`https://doi.org/10.1175/JCLI-D-16-0758.1 <https://journals.ametsoc.org/view/journals/clim/30/14/jcli-d-16-0758.1.xml>`_