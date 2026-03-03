pyhazards.datasets.era5.inspection
==================================

Description
-----------

CLI entrypoint for ERA5 file inspection. It validates local NetCDF files, prints discovered dimensions/variables,
and falls back to HDF5-level inspection when optional xarray NetCDF backends are unavailable.

Example usage
-------------

.. code-block:: bash

   python -m pyhazards.datasets.era5.inspection --path pyhazards/data/era5_subset --max-vars 10

