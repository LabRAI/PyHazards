era5
====

Description
-----------

ERA5 is the ECMWF reanalysis distributed via Copernicus CDS and is widely used as a meteorological
baseline for hazard modeling. This module is the dataset-level inspection entrypoint for local ERA5
NetCDF files.

For full background and publication details, see :doc:`ERA5 dataset page </datasets/era5>`.

Note on alignment: use ``pyhazards.datasets.era5.inspection`` for consistent dataset inspection.
For HydroGraphNet training/validation, ``load_hydrograph_data`` is the model-specific adapter that
converts ERA5 NetCDF files into a graph ``DataBundle``.

Example of how to use it
------------------------

.. code-block:: bash

   # Inspect local ERA5 NetCDF files
   python -m pyhazards.datasets.era5.inspection --path pyhazards/data/era5_subset --max-vars 10

.. code-block:: python

   import subprocess

   # Step 1: inspect dataset files (canonical dataset entrypoint)
   subprocess.run(
       [
           "python", "-m", "pyhazards.datasets.era5.inspection",
           "--path", "pyhazards/data/era5_subset",
           "--max-vars", "10",
       ],
       check=True,
   )
