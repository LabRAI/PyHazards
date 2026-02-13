wfigs
=====

Description
-----------

WFIGS provides authoritative U.S. interagency incident-level wildfire records (ignition, location,
status, extent), commonly used as ground-truth wildfire labels. This module is the callable
inspection entrypoint for quick local path and file inventory validation.

For full background and publication details, see :doc:`WFIGS dataset page </datasets/wfigs>`.

Example of how to use it
------------------------

.. code-block:: bash

   python -m pyhazards.datasets.wfigs.inspection --path /path/to/wfigs_data --max-items 10

.. code-block:: python

   import subprocess

   subprocess.run(
       [
           "python", "-m", "pyhazards.datasets.wfigs.inspection",
           "--path", "/path/to/wfigs_data",
           "--max-items", "10",
       ],
       check=True,
   )
