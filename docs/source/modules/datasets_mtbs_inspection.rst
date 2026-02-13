mtbs
====

Description
-----------

MTBS provides U.S. wildfire perimeters and burn severity products (Landsat-derived), widely used
for post-fire assessment and long-term wildfire analysis. This module is the callable inspection
entrypoint for quick local path and file inventory validation.

For full background and publication details, see :doc:`MTBS dataset page </datasets/mtbs>`.

Example of how to use it
------------------------

.. code-block:: bash

   python -m pyhazards.datasets.mtbs.inspection --path /path/to/mtbs_data --max-items 10

.. code-block:: python

   import subprocess

   subprocess.run(
       [
           "python", "-m", "pyhazards.datasets.mtbs.inspection",
           "--path", "/path/to/mtbs_data",
           "--max-items", "10",
       ],
       check=True,
   )
