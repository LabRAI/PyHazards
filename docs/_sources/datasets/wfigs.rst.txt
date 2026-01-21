.. _ds-wfigs:

WFIGS
=====

Near-real-time wildfire perimeter datasets providing best-available incident
boundaries for operational monitoring and evaluation.

The Wildland Fire Interagency Geospatial Services (WFIGS) program aggregates and
harmonizes perimeter data from multiple incident sources, supporting rapid
situational awareness and perimeter-based analyses.

----------

**Data characteristics**

====================  =========================================
Temporal resolution   Near real time
Spatial resolution    Incident-dependent
Spatial coverage      United States
Data structure        Vector polygons
Data format            ArcGIS Feature Service, Shapefile
====================  =========================================

----------

**Typical attributes**

- Incident identifiers
- Fire perimeters
- Update timestamps

----------

**Common use cases**

- Incident tracking and monitoring
- Perimeter-based model evaluation
- Operational decision support

----------

**Parameters**

transform (callable, optional)
    A function/transform applied to perimeter geometries, such as simplification,
    buffering, or rasterization.

----------

**Links**

- `WFIGS perimeters <https://data-nifc.opendata.arcgis.com/datasets/nifc::wfigs-current-interagency-fire-perimeters/about>`_
