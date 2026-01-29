MTBS
====

National-scale wildfire burn severity and perimeter dataset produced by the U.S. Geological Survey (USGS) and partners,
supporting post-fire impact assessment and long-term wildfire regime analysis.

.. admonition:: At-a-glance (Quick Facts)
   :class: note

   .. list-table::
      :widths: 26 74
      :stub-columns: 1

      * - Producer
        - USGS + USDA Forest Service (interagency MTBS program)
      * - Spatial coverage
        - United States (CONUS + Alaska + Hawaii + Puerto Rico)
      * - Spatial resolution
        - 30 m (Landsat-based)
      * - Data structure
        - Event-based rasters + vector perimeters (per-fire bundles; mosaics available)
      * - Formats
        - GeoTIFF, Shapefile, File Geodatabase
      * - Projection / CRS
        - Projected coordinate systems (product-dependent; distributed with metadata)
      * - **Data update frequency**
        - **Continuous mapping + quarterly releases (typically Feb / May / Aug / Nov).**
      * - Period of record
        - 1984–near present (inventory grows with releases)
      * - Typical role in PyHazards
        - Post-fire labels / validation targets (extent, severity classes, burn metrics)

----

Overview
--------

**MTBS (Monitoring Trends in Burn Severity)** is a long-term remote sensing program that maps wildfire
perimeters and burn severity across the United States using Landsat imagery. Burn severity is derived
from spectral change metrics and categorized into standardized severity classes.

MTBS is widely used for post-fire ecological assessment, wildfire regime and trend analysis, and as
training or validation data for models that predict burn extent, burn severity, or fire impacts.

----

Stats
-----

.. list-table::
   :header-rows: 1
   :widths: 22 22 26 30

   * - Spatial
     - Temporal
     - Update cadence
     - Representation
   * - 30 m (Landsat)
     - Fire-event / fire-year
     - Continuous mapping; quarterly releases
     - Per-fire rasters + perimeters (vector)

----

Data Characteristics
--------------------

- **Spatial coverage:** United States
- **Spatial resolution:** 30 m (Landsat-based)
- **Temporal coverage:** Fire-event products aggregated by fire year; historical archive from 1984 onward
- **Data structure:** Event-based raster layers and vector perimeters
- **Data format:** GeoTIFF, Shapefile, File Geodatabase
- **Coordinate system:** Projected coordinate systems (product-dependent)
- **Data update frequency:** Continuous mapping with **quarterly releases** (commonly Feb / May / Aug / Nov)

.. note::
   MTBS is not strictly “near-real-time”. Product availability depends on suitable pre-/post-fire Landsat
   imagery and production workflow; recent fire seasons are typically populated progressively across
   quarterly releases.

----

Variables
---------

MTBS products typically include:

- Fire perimeter polygons for individual fire events
- Burn severity raster layers (e.g., dNBR, RdNBR)
- Categorical burn severity classifications
- Fire metadata such as ignition year and fire name

----

Typical Use Cases
-----------------

- Post-fire burn severity and impact assessment
- Long-term wildfire regime and trend analysis
- Model evaluation and validation for fire extent and severity prediction
- Integration with meteorology, fuels, and topography for fire impact studies

----

Access
------

MTBS data products are publicly available through USGS portals:

- `MTBS data portal <https://burnseverity.cr.usgs.gov/>`_
- `USGS MTBS overview <https://www.usgs.gov/programs/mtbs>`_

----

Reference
---------

Eidenshink, J., Schwind, B., Brewer, K., Zhu, Z., Quayle, B., & Howard, S. (2007).
*A project for monitoring trends in burn severity*.
Fire Ecology, 3(1), 3–21.
`https://doi.org/10.4996/fireecology.0301003 <https://doi.org/10.4996/fireecology.0301003>`_
