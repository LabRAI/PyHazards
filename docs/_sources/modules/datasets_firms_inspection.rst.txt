firms
=====

Description
-----------

FIRMS provides near-real-time active fire detections (MODIS/VIIRS), often used for wildfire
occurrence labeling and operational monitoring. This module is the callable inspection entrypoint
for quick local path and file inventory validation.

For full background and publication details, see :doc:`FIRMS dataset page </datasets/firms>`.

Example of how to use it
------------------------

.. code-block:: bash

   python -m pyhazards.datasets.firms.inspection --path /path/to/firms_data --max-items 10

.. code-block:: python

   import subprocess

   subprocess.run(
       [
           "python", "-m", "pyhazards.datasets.firms.inspection",
           "--path", "/path/to/firms_data",
           "--max-items", "10",
       ],
       check=True,
   )
