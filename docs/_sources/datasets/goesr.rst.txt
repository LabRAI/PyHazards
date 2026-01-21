.. _ds-goesr:

GOES-R
======

Geostationary Lightning Mapper (GLM) observations providing high-frequency
lightning activity measurements over the Americas.

GOES-R GLM detects lightning events, groups, and flashes at sub-minute cadence,
offering proxies for convective intensity and ignition-related processes in
hazard modeling.

----------

**Data characteristics**

====================  =========================================
Temporal resolution   ~20 seconds
Spatial resolution    ~10 km (storm scale)
Spatial coverage      Americas (geostationary)
Data structure        Event-based lightning detections
Data format            NetCDF
====================  =========================================

----------

**Typical attributes**

- Lightning event time and location
- Flash and group identifiers
- Radiometric measurements

----------

**Common use cases**

- Convective activity monitoring
- Lightning–wildfire ignition analysis
- Severe weather characterization

----------

**Parameters**

transform (callable, optional)
    A function/transform applied to lightning events, such as temporal aggregation
    or spatial binning.

----------

**Links**

- `GLM metadata <https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc%3AC01527>`_
- `Goodman et al. (2013) <https://doi.org/10.1016/j.atmosres.2013.01.006>`_
