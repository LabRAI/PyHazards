landfire
========

Description
-----------

LANDFIRE provides national fuels and vegetation layers, commonly used as static landscape
covariates for wildfire behavior and risk modeling. This module is the callable inspection
entrypoint for quick local path and file inventory validation.

For full background and publication details, see :doc:`LANDFIRE dataset page </datasets/landfire>`.

Example of how to use it
------------------------

.. code-block:: bash

   python -m pyhazards.datasets.landfire.inspection --path /path/to/landfire_data --max-items 10

.. code-block:: python

   import subprocess

   subprocess.run(
       [
           "python", "-m", "pyhazards.datasets.landfire.inspection",
           "--path", "/path/to/landfire_data",
           "--max-items", "10",
       ],
       check=True,
   )
