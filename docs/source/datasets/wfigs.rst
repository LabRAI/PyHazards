WFIGS
=====

Authoritative incident-level wildfire records maintained by U.S. interagency fire management systems,
providing official information on real wildfire incidents across the United States.

.. admonition:: At-a-glance (Quick Facts)
   :class: note

   .. list-table::
      :widths: 26 74
      :stub-columns: 1

      * - Provider
        - National Interagency Fire Center (NIFC) Open Data / Interagency WFIGS
      * - Data type
        - **Incident-level event records** with **point + perimeter** geometries (authoritative “ground truth”)
      * - Spatial coverage
        - United States
      * - Temporal coverage
        - Historical + ongoing incidents (continuously appended/updated)
      * - **Data update frequency**
        - **Refreshed from IRWIN every ~5 minutes; perimeter-source changes may take up to ~15 minutes to appear.**
          “Current” layers apply **fall-off** rules to remove stale incidents; “To-Date” layers keep the full year.
      * - Data structure
        - Vector features (incident points / polygons) + attributes and status fields
      * - Formats
        - ArcGIS REST services, GeoJSON, Shapefile (download options vary by layer)
      * - CRS
        - Typically WGS84 geographic (service-dependent)
      * - Typical role in PyHazards
        - Authoritative labels/targets + validation baseline for satellite detections (FIRMS/GOES)

----

Overview
--------

**WFIGS (Wildland Fire Incident Geospatial Services)** is an interagency system that aggregates and
distributes geospatial information on active and historical wildland fire incidents. Each record
represents an officially reported wildfire event (IRWIN-linked incident), rather than a satellite-detected hotspot.

WFIGS is commonly treated as an authoritative source of wildfire ground truth and is widely used to
validate satellite-based fire detections and to label wildfire occurrence in modeling pipelines.

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
   * - U.S.
     - Event-based
     - ~5 min refresh (IRWIN-linked)
     - Incident points + perimeters (vector)

----

Data Characteristics
--------------------

- **Spatial coverage:** United States
- **Temporal coverage:** Historical + ongoing wildfire incidents
- **Temporal resolution:** Event-based (ignition/discovery, containment, status updates)
- **Data structure:** Incident-level event records (point or polygon geometries)
- **Data update frequency:** Refreshed from IRWIN approximately every **5 minutes**; perimeter-source changes may take up to **~15 minutes** to display
- **Current vs To-Date layers:**
  - **Current** layers use **fall-off rules** to remove stale records (based on fire size and last-update time)
  - **To-Date** layers retain the full current-year incident/perimeter set (no fall-off)

.. note::
   WFIGS is “working/operational data”: incident attributes and geometries may change as the incident evolves
   (e.g., perimeter refinements, containment updates, record reconciliation).

----

Variables
---------

Typical WFIGS records include:

- Incident name and unique incident identifier
- Ignition/discovery date, containment status, and last-modified timestamps
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

- `NIFC Open Data (WFIGS layers) <https://data-nifc.opendata.arcgis.com/>`_
- `National Interagency Fire Center (NIFC) <https://www.nifc.gov/>`_

----

Reference
---------

National Interagency Fire Center (NIFC).
*Wildland Fire Incident Geospatial Services (WFIGS)* (Open Data portal documentation and layers).
`https://data-nifc.opendata.arcgis.com/ <https://data-nifc.opendata.arcgis.com/>`_
