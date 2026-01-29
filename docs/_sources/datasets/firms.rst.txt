FIRMS
=====

Near-real-time global active fire detection dataset provided by NASA, based on satellite thermal anomaly
observations for operational wildfire monitoring and hazard analysis.

.. admonition:: At-a-glance (Quick Facts)
   :class: note

   .. list-table::
      :widths: 26 74
      :stub-columns: 1

      * - Provider
        - NASA LANCE / FIRMS
      * - Data type
        - Event-based **point detections** (vector hotspots; not gridded tensors)
      * - Sensors
        - MODIS (Terra/Aqua) and VIIRS (Suomi NPP / NOAA-20 / NOAA-21) :contentReference[oaicite:1]{index=1}
      * - Spatial coverage
        - Global
      * - Spatial resolution
        - ~375 m (VIIRS), ~1 km (MODIS) :contentReference[oaicite:2]{index=2}
      * - Latency (typical)
        - **< 3 hours globally**; **< 1–30 mins for US/Canada** products :contentReference[oaicite:3]{index=3}
      * - **Data update frequency**
        - **Fire maps refresh ~every 5 minutes; downloadable SHP/KML/CSV refresh ~hourly.**
          NRT detections are later replaced by standard/science-quality data (~3 months). :contentReference[oaicite:4]{index=4}
      * - Formats
        - CSV, Shapefile, GeoJSON, KML (plus web services)
      * - CRS
        - Geographic lat–lon (WGS84) :contentReference[oaicite:5]{index=5}

----

Overview
--------

**FIRMS (Fire Information for Resource Management System)** is a NASA-operated system that distributes
active fire and thermal anomaly detections derived from satellite sensors such as MODIS and VIIRS.
Each record corresponds to a time-stamped hotspot detection associated with a potential fire event.

FIRMS is widely used for operational wildfire monitoring, rapid situational awareness, and as event-level
labels/targets in wildfire prediction pipelines when combined with meteorological and land-surface data.

----

Stats
-----

.. list-table::
   :header-rows: 1
   :widths: 22 22 26 30

   * - Spatial
     - Latency
     - Refresh cadence
     - Representation
   * - 375 m (VIIRS) / 1 km (MODIS)
     - <3h global; <1–30m US/CA
     - Map ~5 min; files ~60 min
     - Point detections (vector)

----

Data Characteristics
--------------------

- **Spatial coverage:** Global
- **Spatial resolution:** Sensor-dependent (e.g., ~375 m for VIIRS; ~1 km for MODIS) :contentReference[oaicite:6]{index=6}
- **Temporal resolution:** Event-based detections (polar-orbiting overpasses; multiple updates/day)
- **Latency:** FIRMS makes NRT detections available within ~3 hours globally; US/Canada streams can be faster (<1–30 mins) :contentReference[oaicite:7]{index=7}
- **Data structure:** Event-based point detections (not gridded tensors)
- **Formats:** CSV, Shapefile, GeoJSON, KML
- **CRS:** Geographic latitude–longitude (WGS84) :contentReference[oaicite:8]{index=8}
- **Data update frequency:** Fire maps update ~every 5 minutes; SHP/KML/CSV downloads update ~every 60 minutes :contentReference[oaicite:9]{index=9}
- **NRT → standard replacement:** NRT detections are replaced by standard/science-quality products when available (~3 months) :contentReference[oaicite:10]{index=10}

----

Variables
---------

Each FIRMS detection record typically includes:

- Detection time and satellite overpass metadata
- Geographic location (latitude, longitude)
- Fire radiative power (FRP) / thermal anomaly indicators (product-dependent)
- Confidence / quality flags (sensor- and product-specific)

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

.. note::
   Some archive / bulk download endpoints require an Earthdata Login, depending on the access route. :contentReference[oaicite:11]{index=11}

----

Reference
---------

Schroeder, W., Oliva, P., Giglio, L., & Csiszar, I. (2014).
*The New VIIRS 375 m active fire detection data product: Algorithm description and initial assessment*.
Remote Sensing of Environment, 143, 85–96.
`https://doi.org/10.1016/j.rse.2013.08.008 <https://doi.org/10.1016/j.rse.2013.08.008>`_