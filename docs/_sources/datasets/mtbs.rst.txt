MTBS
====

National-scale wildfire burn severity and perimeter dataset produced by the U.S. Geological Survey (USGS),
supporting post-fire impact assessment and long-term wildfire regime analysis.

----

Overview
--------

**MTBS (Monitoring Trends in Burn Severity)** is a long-term remote sensing program that maps wildfire
perimeters and burn severity across the United States using Landsat imagery. Burn severity is derived
from spectral change metrics and categorized into standardized severity classes.

MTBS is widely used for post-fire ecological assessment, wildfire regime and trend analysis, and as
training or validation data for models that predict burn extent, burn severity, or fire impacts.

----

Data Characteristics
--------------------

- **Spatial coverage:** United States  
- **Spatial resolution:** 30 m (Landsat-based)  
- **Temporal coverage:** Annual fire-year products  
- **Data structure:** Event-based raster layers and vector perimeters  
- **Data format:** GeoTIFF, Shapefile, File Geodatabase  
- **Coordinate system:** Projected coordinate systems (product-dependent)  

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
