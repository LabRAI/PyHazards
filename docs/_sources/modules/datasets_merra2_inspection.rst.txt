merra2
======

Description
-----------

MERRA-2 is a global atmospheric reanalysis from NASA GMAO, commonly used as hourly gridded
meteorological forcing for hazard modeling. This module is the callable inspection entrypoint for
MERRA-2 data in PyHazards and runs the one-shot inspect pipeline.

For full background and publication details, see :doc:`MERRA-2 dataset page </datasets/merra2>`.

Example of how to use it
------------------------

.. code-block:: bash

   # End-to-end MERRA-2 inspection workflow for one date key
   python -m pyhazards.datasets.merra2.inspection 20260101

.. code-block:: python

   import subprocess

   subprocess.run(
       [
           "python", "-m", "pyhazards.datasets.merra2.inspection",
           "20260101",
       ],
       check=True,
   )
