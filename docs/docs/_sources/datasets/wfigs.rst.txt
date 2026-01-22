WFIGS
=====

Authoritative incident-level wildfire records maintained by U.S. interagency fire management systems,
providing official information on real wildfire events across the United States.

----

Overview
--------

**WFIGS (Wildland Fire Incident Geospatial Services)** is an interagency system that aggregates and
distributes geospatial information on active and historical wildland fire incidents. Each record
represents an officially reported wildfire event, rather than a satellite-detected hotspot.

WFIGS is commonly treated as an authoritative source of wildfire ground truth and is widely used to
validate satellite-based fire detections and to label wildfire occurrence in modeling pipelines.

----

Data Characteristics
--------------------

- **Spatial coverage:** United States  
- **Temporal coverage:** Historical and ongoing wildfire incidents  
- **Temporal resolution:** Event-based (ignition, containment, and status updates)  
- **Data structure:** Incident-level event records (point or polygon geometries)  
- **Data format:** GIS services, GeoJSON, Shapefile  
- **Coordinate system:** Geographic and projected systems (service-dependent)  

----

Variables
---------

Typical WFIGS records include:

- Incident name and unique incident identifier
- Ignition date, discovery date, and containment status
- Incident location (point or perimeter geometry)
- Fire size, cause, and management status
- Associated agency and reporting metadata

----

Typical Use Cases
-----------------

- Ground-truth labeling of wildfire occurrence
- Validation of satellite-based fire detection products (e.g., FIRMS, GOES)
- Analysis of wildfire ignition timing and spatial patterns
- Integration with meteorological and fuel datasets for fire modeling studies

----

Access
------

WFIGS data are publicly accessible through U.S. interagency geospatial portals:

- `WFIGS data portal <https://data-nifc.opendata.arcgis.com/>`_
- `National Interagency Fire Center (NIFC) <https://www.nifc.gov/>`_

----

Reference
---------

National Interagency Fire Center (NIFC).
*Wildland Fire Incident Geospatial Services (WFIGS) Documentation*.
`https://data-nifc.opendata.arcgis.com/ <https://data-nifc.opendata.arcgis.com/>`_
