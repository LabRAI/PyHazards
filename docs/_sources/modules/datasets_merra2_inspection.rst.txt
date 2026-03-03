pyhazards.datasets.merra2.inspection
====================================

Description
-----------

CLI entrypoint for the MERRA-2 inspection workflow in PyHazards. It wraps the full pipeline in
``pyhazards.datasets.inspection`` (download/preprocess/inspect/outputs).

Example usage
-------------

.. code-block:: bash

   # Full pipeline for one date key
   python -m pyhazards.datasets.merra2.inspection 20260101

