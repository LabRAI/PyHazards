noaa_flood
==========

Description
-----------

NOAA Flood Events are event-level flood records from the NOAA Storm Events Database, commonly used
for impact labeling and event analysis. This module is the callable inspection entrypoint for quick
local path and file inventory validation.

For full background and publication details, see :doc:`NOAA Flood dataset page </datasets/noaa_flood>`.

Example of how to use it
------------------------

.. code-block:: bash

   python -m pyhazards.datasets.noaa_flood.inspection --path /path/to/noaa_flood_data --max-items 10

.. code-block:: python

   import subprocess

   subprocess.run(
       [
           "python", "-m", "pyhazards.datasets.noaa_flood.inspection",
           "--path", "/path/to/noaa_flood_data",
           "--max-items", "10",
       ],
       check=True,
   )
