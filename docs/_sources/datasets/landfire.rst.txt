.. _ds-landfire:

LANDFIRE
========

National-scale vegetation and fuel characteristic maps used as static covariates
in wildfire behavior and risk modeling.

LANDFIRE provides gridded datasets describing vegetation type, structure, and
surface and canopy fuels, commonly paired with dynamic meteorological drivers
to support wildfire simulations and hazard assessments.

----------

**Data characteristics**

====================  =========================================
Temporal resolution   Versioned updates (quasi-annual)
Spatial resolution    ~30 m
Spatial coverage      United States
Data structure        Raster map layers
Data format            GeoTIFF
====================  =========================================

----------

**Typical variables**

- Fuel models
- Vegetation type and cover
- Canopy height and density

----------

**Common use cases**

- Wildfire behavior and spread modeling
- Landscape-scale risk assessment
- Static covariates for wildfire occurrence models

----------

**Parameters**

transform (callable, optional)
    A function/transform applied to raster layers, such as reprojection,
    aggregation, or feature encoding.

----------

**Links**

- `LANDFIRE data access <https://landfire.gov/getdata.php>`_
- `LANDFIRE program overview <https://research.fs.usda.gov/firelab/products/dataandtools/landfire-landscape-fire-and-resource-management-planning>`_
