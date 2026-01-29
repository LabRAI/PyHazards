NOAA Flood Events
=================

Event-based flood and severe weather records compiled by NOAA, providing historical reports of flood
occurrence and impacts across the United States.

.. admonition:: At-a-glance (Quick Facts)
   :class: note

   .. list-table::
      :widths: 26 74
      :stub-columns: 1

      * - Provider
        - NOAA National Centers for Environmental Information (NCEI)
      * - Primary source
        - NOAA National Weather Service (NWS) Storm Data / Storm Events entries
      * - Spatial coverage
        - United States (county/zone-based reporting; points when available)
      * - Temporal resolution
        - Event-based (begin/end timestamps)
      * - **Data update frequency**
        - **Updated monthly; typically published ~75–90 days after the end of a data month (occasionally up to ~120 days).**
      * - Period of record
        - 1950–present (through the most recently processed month)
      * - Data structure
        - Tabular event records (not gridded tensors)
      * - Formats
        - Web query + bulk CSV/DB extracts
      * - Typical role in PyHazards
        - Event labels / targets (flood occurrence + impacts)

----

Overview
--------

**NOAA Flood Events** are derived from the NOAA **Storm Events Database**, which documents the occurrence,
location, timing, and impacts of severe weather events across the United States. Flood-related events
are reported by local weather offices and emergency management partners, then compiled and archived by NCEI.

This dataset is widely used for flood frequency analysis, impact assessment, and as event-level labels
or targets in supervised hazard modeling, particularly when paired with meteorological reanalysis data.

----

Stats
-----

.. list-table::
   :header-rows: 1
   :widths: 22 22 26 30

   * - Spatial
     - Temporal
     - Update cadence
     - Representation
   * - United States
     - Event-based
     - Monthly (typical ~75–90 day latency)
     - Tabular records + optional point coords

----

Data Characteristics
--------------------

- **Spatial coverage:** United States
- **Temporal coverage:** Historical records, appended as new months are processed
- **Temporal resolution:** Event-based (begin and end timestamps)
- **Data structure:** Tabular event records (not gridded tensors)
- **Location representation:** Administrative regions (state/county/zone); point coordinates where available
- **Data update frequency:** Updated monthly; database is typically refreshed within ~75–90 days after the end of a data month
  (rarely longer, up to ~120 days)
- **Data format:** Online query; bulk downloads and database extracts

.. note::
   For “very recent” months, event records may not yet be available due to reporting/validation and monthly publication lag.

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
- `Storm Events bulk download (CSV) <https://www.ncei.noaa.gov/stormevents/ftp.jsp>`_
- `NOAA NCEI <https://www.ncei.noaa.gov/>`_

----

Reference
---------

NOAA National Centers for Environmental Information (NCEI).
*Storm Events Database Documentation*.
`https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00648 <https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00648>`_
