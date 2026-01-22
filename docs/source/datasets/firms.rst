FIRMS
=====

Near-real-time global active fire detection dataset provided by NASA, based on satellite thermal anomaly
observations for operational wildfire monitoring and hazard analysis.

----

Overview
--------

**FIRMS (Fire Information for Resource Management System)** is a NASA-operated system that distributes
active fire and thermal anomaly detections derived from satellite sensors, including MODIS and VIIRS.
Each record corresponds to a time-stamped hotspot detection associated with a potential fire event.

FIRMS is widely used for operational wildfire monitoring, rapid situational awareness, and as event-level
labels or targets in wildfire prediction pipelines when combined with meteorological and land-surface data.

----

Data Characteristics
--------------------

- **Spatial coverage:** Global  
- **Spatial resolution:** Sensor-dependent (e.g., ~375 m for VIIRS)  
- **Temporal resolution:** Near real time (multiple updates per day)  
- **Data structure:** Event-based point detections (not gridded tensors)  
- **Data format:** CSV, Shapefile, GeoJSON, KML  
- **Coordinate system:** Geographic latitude–longitude  

----

Variables
---------

Each FIRMS detection record typically includes:

- Detection time and satellite overpass information
- Geographic location (latitude and longitude)
- Fire radiative power (FRP) or thermal anomaly indicators
- Confidence or quality flags (sensor- and product-specific)

----

Typical Use Cases
-----------------

- Operational wildfire monitoring and early detection
- Event labeling for supervised wildfire prediction models
- Spatiotemporal analysis of fire occurrence and activity patterns
- Integration with meteorological and fuel datasets for hazard modeling

----

Access
------

FIRMS data are publicly accessible through NASA Earthdata services:

- `FIRMS portal <https://firms.modaps.eosdis.nasa.gov/>`_
- `NASA Earthdata <https://earthdata.nasa.gov/>`_

----

Reference
---------

Schroeder, W., Oliva, P., Giglio, L., & Csiszar, I. (2014).
*The New VIIRS 375 m active fire detection data product: Algorithm description and initial assessment*.
Remote Sensing of Environment, 143, 85–96.
`https://doi.org/10.1016/j.rse.2013.08.008 <https://doi.org/10.1016/j.rse.2013.08.008>`_
