NOAA Flood Events
=================

Event-based flood and severe weather records compiled by the U.S. National Oceanic and Atmospheric
Administration (NOAA), providing historical reports of flood occurrence and impacts.

----

Overview
--------

**NOAA Flood Events** are derived from the NOAA Storm Events Database, which documents the occurrence,
location, timing, and impacts of severe weather events across the United States. Flood-related events
are reported by local weather offices and emergency management agencies.

This dataset is widely used for flood frequency analysis, impact assessment, and as event-level labels
or targets in supervised hazard modeling, particularly when paired with meteorological reanalysis data.

----

Data Characteristics
--------------------

- **Spatial coverage:** United States  
- **Temporal coverage:** Historical records with monthly updates  
- **Temporal resolution:** Event-based (begin and end timestamps)  
- **Data structure:** Tabular event records (not gridded tensors)  
- **Data format:** CSV, database extracts  
- **Coordinate system:** Administrative regions and point locations (where available)  

----

Variables
---------

Typical flood event records include:

- Event start and end time
- Event location (state, county/zone, and coordinates when available)
- Event type and narrative descriptions
- Reported impacts such as property damage, crop damage, injuries, and fatalities

----

Typical Use Cases
-----------------

- Flood occurrence and frequency analysis
- Impact and damage assessment studies
- Supervised learning with flood events as prediction targets
- Integration with meteorological drivers for flood hazard modeling

----

Access
------

NOAA Storm Events data are publicly available via NOAA NCEI:

- `Storm Events Database <https://www.ncei.noaa.gov/products/storm-events-database>`_
- `NOAA NCEI <https://www.ncei.noaa.gov/>`_

----

Reference
---------

NOAA National Centers for Environmental Information (NCEI).
*Storm Events Database Documentation*.
`https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00648 <https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00648>`_
