goesr
=====

Description
-----------

GOES-R provides high-frequency geostationary multispectral imagery, useful for continuous hazard
monitoring and context features (e.g., smoke/thermal signals). This module is the callable
inspection entrypoint for quick local path and file inventory validation.

For full background and publication details, see :doc:`GOES-R dataset page </datasets/goesr>`.

Example of how to use it
------------------------

.. code-block:: bash

   python -m pyhazards.datasets.goesr.inspection --path /path/to/goesr_data --max-items 10

.. code-block:: python

   import subprocess

   subprocess.run(
       [
           "python", "-m", "pyhazards.datasets.goesr.inspection",
           "--path", "/path/to/goesr_data",
           "--max-items", "10",
       ],
       check=True,
   )
