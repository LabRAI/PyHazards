MERRA-2
=======

Global atmospheric reanalysis dataset produced by NASA Global Modeling and Assimilation Office (GMAO),
providing physically consistent meteorological fields for weather, climate, and hazard modeling.

----

Overview
--------

**MERRA-2 (Modern-Era Retrospective analysis for Research and Applications, Version 2)** is a global
atmospheric reanalysis that assimilates satellite and conventional observations into a numerical
weather prediction system to produce gridded, time-continuous estimates of the atmospheric state.

It is widely used as a core meteorological driver in climate analysis and hazard prediction pipelines,
including wildfire, hurricane, drought, and extreme heat studies.

----

Data Characteristics
--------------------

- **Spatial coverage:** Global  
- **Spatial resolution:** ~0.5° latitude × 0.625° longitude  
- **Temporal resolution:** Hourly  
- **Vertical structure:** Surface fields and multi-level pressure profiles  
- **Data format:** NetCDF  
- **Coordinate system:** Regular latitude–longitude grid  

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

MERRA-2 data can be accessed via NASA GMAO and Earthdata services:

- `MERRA-2 overview <https://gmao.gsfc.nasa.gov/gmao-products/merra-2/>`_
- `NASA Earthdata <https://earthdata.nasa.gov/>`_

----

Reference
---------

Gelaro, R., McCarty, W., Suárez, M. J., *et al.* (2017).
*The Modern-Era Retrospective Analysis for Research and Applications, Version 2 (MERRA-2)*.
Journal of Climate, 30(14), 5419–5454.
`https://doi.org/10.1175/JCLI-D-16-0758.1 <https://journals.ametsoc.org/view/journals/clim/30/14/jcli-d-16-0758.1.xml>`_
