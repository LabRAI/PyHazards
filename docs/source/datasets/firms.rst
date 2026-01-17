.. _ds-firms:

FIRMS
=====

Near-real-time active fire detections derived from MODIS and VIIRS satellite
sensors, widely used for wildfire monitoring and event labeling.

The Fire Information for Resource Management System (FIRMS) provides point-based
fire and thermal anomaly detections with rapid latency, supporting operational
situational awareness and serving as event-level labels for wildfire prediction
pipelines when combined with meteorological and fuel datasets.

----------

**Data characteristics**

====================  =========================================
Temporal resolution   Sub-daily (near real time)
Spatial resolution    ~375 m (VIIRS), ~1 km (MODIS)
Spatial coverage      Global
Data structure        Event-level point detections
Data format            CSV, Shapefile, GeoJSON, KML
====================  =========================================

----------

**Typical attributes**

- Detection timestamp
- Latitude and longitude
- Sensor and product identifiers
- Fire radiative power and confidence measures

----------

**Common use cases**

- Operational wildfire monitoring
- Event labeling for wildfire occurrence models
- Evaluation of fire prediction and detection systems

----------

**Parameters**

transform (callable, optional)
    A function/transform applied to each detection record before being returned,
    such as filtering, feature normalization, or coordinate transformations.

----------

**Links**

- `FIRMS portal <https://firms.modaps.eosdis.nasa.gov/>`_
- `Schroeder et al. (2014) <https://doi.org/10.1016/j.rse.2013.08.008>`_
