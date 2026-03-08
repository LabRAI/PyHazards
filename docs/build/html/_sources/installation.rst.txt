Installation
============

Use this page to install PyHazards, verify that the package imports correctly,
and choose the right setup path for local use or contribution. PyHazards
supports Python 3.8 through 3.12 and installs with ``pip``.

Requirements
------------

- Python ``>=3.8, <3.13``
- PyTorch ``>=2.3, <3.0``

Install from PyPI
-----------------

Install from PyPI:

.. code-block:: bash

    pip install pyhazards

GPU Install
-----------

If you plan to run on GPU, install a matching PyTorch build first and then
install PyHazards.

Example for CUDA 12.6:

.. code-block:: bash

    pip install torch --index-url https://download.pytorch.org/whl/cu126
    pip install pyhazards

Install from Source
-------------------

Use an editable install when you are contributing code or documentation:

.. code-block:: bash

    git clone https://github.com/LabRAI/PyHazards.git
    cd PyHazards
    python -m pip install -e .

Verify the Installation
-----------------------

Run a small import check to confirm that the package is available in the
environment:

.. code-block:: bash

    python -c "import pyhazards; print(pyhazards.__version__)"

You should see the installed package version printed to stdout.

Next Steps
----------

- Continue to :doc:`quick_start` for the first end-to-end workflow.
- See :doc:`implementation` if you are setting up a contributor workflow.
