ERA5
====

Global atmospheric reanalysis produced by the European Centre for Medium-Range Weather Forecasts (ECMWF),
providing high-resolution meteorological fields for weather, climate, and hazard-related applications.

----

Overview
--------

**ERA5** is the fifth-generation global reanalysis developed by ECMWF under the Copernicus Climate Change
Service (C3S). It combines vast amounts of historical observations with a modern data assimilation system
to generate temporally consistent, high-resolution estimates of the atmospheric state.

ERA5 is widely adopted as a standard meteorological baseline for climate studies and natural hazard
modeling, including wildfire danger assessment, flood risk analysis, and extreme weather attribution.

----

Data Characteristics
--------------------

- **Spatial coverage:** Global  
- **Spatial resolution:** ~0.25° latitude × 0.25° longitude  
- **Temporal resolution:** Hourly  
- **Vertical structure:** Single-level fields and optional pressure/model levels  
- **Data format:** GRIB and NetCDF  
- **Coordinate system:** Regular latitude–longitude grid  

----

Variables
---------

ERA5 provides a comprehensive set of atmospheric and surface variables, including:

- Near-surface meteorology (2 m temperature, dewpoint, wind)
- Precipitation, radiation, and surface fluxes
- Atmospheric pressure-level variables (temperature, winds, geopotential)
- Boundary-layer and land-surface diagnostics

----

Typical Use Cases
-----------------

- Meteorological forcing for wildfire, flood, and extreme weather prediction models
- Climate variability and trend analysis
- Environmental covariates for spatiotemporal machine learning tasks
- Benchmark reanalysis input for weather–climate modeling pipelines

----

Access
------

ERA5 data are distributed via the Copernicus Climate Data Store (CDS):

- `ERA5 single levels <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview>`_
- `Copernicus Climate Data Store <https://cds.climate.copernicus.eu/>`_

----

Reference
---------

Hersbach, H., Bell, B., Berrisford, P., *et al.* (2020).
*The ERA5 global reanalysis*.
Quarterly Journal of the Royal Meteorological Society, 146(730), 1999–2049.
`https://doi.org/10.1002/qj.3803 <https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.3803>`_
