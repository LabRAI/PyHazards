.. _ds-merra2:

MERRA2
======

Global atmospheric reanalysis produced by NASA GMAO, providing hourly gridded
meteorological fields widely used as environmental drivers in climate and
hazard modeling.

MERRA-2 (Modern-Era Retrospective analysis for Research and Applications, Version 2)
assimilates satellite and in-situ observations into a global atmospheric model
to generate consistent, long-term records of surface and atmospheric states.
It is commonly used as a backbone meteorological dataset for wildfire,
flooding, and other natural hazard studies.

----------

**Data characteristics**

====================  =========================================
Temporal resolution   Hourly
Spatial resolution    ~0.5° × 0.625° (lat × lon)
Spatial coverage      Global
Vertical structure    Surface + pressure-level fields
Data format            NetCDF
====================  =========================================

----------

**Typical variables**

- Near-surface meteorology (temperature, humidity, wind, pressure)
- Radiation and energy fluxes
- Precipitation diagnostics
- Atmospheric profiles on pressure levels

----------

**Common use cases**

- Environmental drivers for hazard prediction models
- Climate and reanalysis benchmarking
- Coupling with event-based datasets (e.g., wildfire perimeters, flood reports)
- Long-term trend and variability analysis

----------

**Parameters**

transform (callable, optional)
    A function/transform applied to each time slice or spatial sample before
    being returned. This can be used for normalization, feature selection,
    or conversion to tensors.

----------

**Links**

- `MERRA-2 overview <https://gmao.gsfc.nasa.gov/gmao-products/merra-2/>`_
- `Gelaro et al. (2017) <https://journals.ametsoc.org/view/journals/clim/30/14/jcli-d-16-0758.1.xml>`_
