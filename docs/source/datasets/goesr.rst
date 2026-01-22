GOES-R
======

High-frequency geostationary satellite observations from NOAA’s GOES-R series, providing continuous
multispectral imagery for monitoring atmospheric and surface processes.

----

Overview
--------

**GOES-R** refers to the latest generation of NOAA geostationary operational environmental satellites,
including GOES-16, GOES-17, and GOES-18. The series carries the Advanced Baseline Imager (ABI), which
provides rapid-refresh, multispectral observations over the Americas.

GOES-R data are widely used for real-time monitoring of weather systems, smoke and fire evolution, and
early wildfire detection, particularly where high temporal resolution is critical.

----

Data Characteristics
--------------------

- **Spatial coverage:** Western Hemisphere (geostationary view)  
- **Spatial resolution:** ~0.5–2 km (band-dependent)  
- **Temporal resolution:** 1–15 minutes (mode-dependent)  
- **Data structure:** Gridded satellite imagery (raster time series)  
- **Data format:** NetCDF  
- **Coordinate system:** Geostationary projection (with latitude–longitude products available)  

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
- `NOAA Big Data Program <https://www.noaa.gov/information-technology/open-data-dissemination>`_

----

Reference
---------

Schmit, T. J., Griffith, P., Gunshor, M. M., *et al.* (2017).
*A closer look at the ABI on the GOES-R series*.
Bulletin of the American Meteorological Society, 98(4), 681–698.
`https://doi.org/10.1175/BAMS-D-15-00230.1 <https://doi.org/10.1175/BAMS-D-15-00230.1>`_
