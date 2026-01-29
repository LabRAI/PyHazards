LANDFIRE
========

National-scale vegetation, fuel, and landscape characterization dataset produced by the U.S. Forest Service,
supporting wildfire behavior modeling and landscape-scale fire risk assessment.

.. admonition:: At-a-glance (Quick Facts)
   :class: note

   .. list-table::
      :widths: 26 74
      :stub-columns: 1

      * - Producer
        - U.S. Forest Service (LANDFIRE Program)
      * - Spatial coverage
        - United States (CONUS + Alaska/Hawaii; some products include PRVI)
      * - Spatial resolution
        - ~30 m (product-dependent)
      * - Data structure
        - Gridded raster layers (static / slowly varying background covariates)
      * - Formats
        - GeoTIFF (commonly distributed); additional GIS packages may be provided
      * - Projection / CRS
        - Projected coordinate systems (product-dependent; distributed with dataset metadata)
      * - **Data update frequency**
        - **Annual/yearly updates (versioned “LF YYYY Update” suites), targeting products current to the previous year.**
      * - Typical role in PyHazards
        - Fuels + vegetation + canopy covariates for behavior/spread/risk modeling

----

Overview
--------

**LANDFIRE (Landscape Fire and Resource Management Planning Tools)** is a geospatial data program that
provides consistent, nationwide maps of vegetation, fuels, and fire regimes across the United States.
Products are derived from remote sensing, field observations, disturbance mapping, and ecological modeling.

LANDFIRE datasets are widely used as static or slowly varying background layers in wildfire modeling,
including fire behavior simulation, fire spread modeling, and wildfire risk assessment pipelines.

----

Stats
-----

.. list-table::
   :header-rows: 1
   :widths: 22 22 26 30

   * - Spatial
     - Update cadence
     - Primary layers
     - Representation
   * - ~30 m
     - Annual (versioned updates)
     - Fuels / vegetation / canopy / fire regime
     - Raster grids (GeoTIFF)

----

Data Characteristics
--------------------

- **Spatial coverage:** United States
- **Spatial resolution:** ~30 m (product-dependent)
- **Temporal coverage:** Versioned releases (annual/yearly update suites)
- **Data structure:** Gridded raster layers (static or slowly varying)
- **Data format:** GeoTIFF
- **Coordinate system:** Projected coordinate systems (product-dependent)
- **Data update frequency:** Annual/yearly updates; “LF YYYY Update” products aim to be current to the
  previous year (reduced-latency update strategy)

----

Variables
---------

LANDFIRE products include multiple thematic layers, such as:

- Fuel models (e.g., Fire Behavior Fuel Models)
- Vegetation type, cover, and height
- Canopy characteristics (canopy cover, base height, bulk density)
- Fire regime and disturbance descriptors (annual disturbance / transitions)

----

Typical Use Cases
-----------------

- Fuel characterization for wildfire behavior and spread modeling
- Landscape-scale wildfire risk and hazard assessment
- Static covariates in machine learning–based wildfire prediction models
- Integration with meteorological and ignition datasets for end-to-end fire modeling

----

Access
------

LANDFIRE data products are publicly accessible via U.S. Forest Service portals:

- `LANDFIRE data access <https://landfire.gov/getdata.php>`_
- `LANDFIRE program overview <https://www.landfire.gov/>`_

----

Reference
---------

Rollins, M. G. (2009).
*LANDFIRE: A nationally consistent vegetation, wildland fire, and fuel assessment*.
International Journal of Wildland Fire, 18(3), 235–249.
`https://doi.org/10.1071/WF08088 <https://doi.org/10.1071/WF08088>`_
