.. _ds-mtbs:

MTBS
====

Burn severity and wildfire perimeter products for the United States, designed
for post-fire assessment and long-term wildfire regime analysis.

Monitoring Trends in Burn Severity (MTBS) provides mapped fire perimeters and
burn severity rasters derived from Landsat imagery, enabling consistent
evaluation of fire impacts across decades.

----------

**Data characteristics**

====================  =========================================
Temporal resolution   Annual (by fire year)
Spatial resolution    ~30 m
Spatial coverage      United States
Data structure        Raster layers and vector perimeters
Data format            GeoTIFF, Shapefile, Geodatabase
====================  =========================================

----------

**Typical attributes**

- Fire perimeters
- Burn severity indices
- Fire year and event metadata

----------

**Common use cases**

- Post-fire impact assessment
- Long-term wildfire trend analysis
- Training and validation data for burn severity models

----------

**Parameters**

transform (callable, optional)
    A function/transform applied to raster or vector products, such as resampling,
    masking, or conversion to model-ready representations.

----------

**Links**

- `MTBS portal <https://burnseverity.cr.usgs.gov/>`_
- `Eidenshink et al. (2007) <https://doi.org/10.4996/fireecology.0301003>`_
