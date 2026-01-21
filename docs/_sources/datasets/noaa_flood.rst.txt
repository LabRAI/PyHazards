.. _ds-noaa-flood:

NOAA Flood
==========

Event-based flood reports derived from NOAA’s Storm Events Database, commonly
used for flood occurrence and impact analysis.

The dataset provides documented flood events with temporal, geographic, and
impact-related attributes, enabling supervised modeling and historical trend
studies when paired with meteorological drivers.

----------

**Data characteristics**

====================  =========================================
Temporal resolution   Event-based
Spatial resolution    County / zone level
Spatial coverage      United States
Data structure        Tabular event records
Data format            CSV
====================  =========================================

----------

**Typical attributes**

- Event start and end time
- Location and affected area
- Damage and casualty estimates

----------

**Common use cases**

- Flood occurrence modeling
- Impact and loss analysis
- Evaluation of flood prediction systems

----------

**Parameters**

transform (callable, optional)
    A function/transform applied to event records, such as filtering,
    feature extraction, or temporal aggregation.

----------

**Links**

- `Storm Events Database <https://www.ncei.noaa.gov/products/storm-events-database>`_
