.. _ds-era5:

ERA5
====

High-resolution global atmospheric reanalysis produced by ECMWF and distributed
via the Copernicus Climate Data Store (CDS), providing standardized meteorological
fields widely used in weather, climate, and hazard modeling.

ERA5 combines numerical weather prediction with extensive data assimilation to
produce a consistent global reanalysis from 1940 to present. Owing to its high
spatial resolution and frequent updates, ERA5 is commonly used as a reference
meteorological dataset for wildfire danger assessment, flood modeling, extreme
weather analysis, and environmental benchmarking.

----------

**Data characteristics**

====================  =========================================
Temporal resolution   Hourly
Spatial resolution    ~0.25° × 0.25° (lat × lon)
Spatial coverage      Global
Vertical structure    Single-level and pressure/model-level fields
Data format            GRIB and NetCDF
====================  =========================================

----------

**Typical variables**

- Near-surface meteorology (2m temperature, dew point, wind, surface pressure)
- Precipitation and snowfall
- Radiation and surface energy fluxes
- Atmospheric state variables on pressure and model levels

----------

**Common use cases**

- Meteorological covariates for hazard prediction models
- Weather and climate benchmarking
- Extreme event analysis (heatwaves, heavy rainfall, wind extremes)
- Coupling with event-based hazard datasets (e.g., wildfire perimeters, flood reports)

----------

**Parameters**

transform (callable, optional)
    A function/transform applied to each temporal or spatial sample before
    being returned. This can be used for normalization, regridding, variable
    selection, or conversion to tensor-based representations.

----------

**Links**

- `ERA5 single levels (CDS) <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview>`_
- `Hersbach et al. (2020) <https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.3803>`_
