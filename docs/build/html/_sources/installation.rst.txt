Installation
============

PyHazard requires Python 3.8+ and can be installed using pip. We recommend using a conda environment for installation.

Installing PyHazard
----------------

To get started with PyHazard, set up your environment by installing the required dependencies:

.. code-block:: bash

    pip install -r requirements.txt

Ensure you have Python installed (version 3.8 or higher recommended) along with the necessary libraries listed
in `requirements.txt`.

Specifically, using following command to install `dgl 2.2.1` and ensure your `pytorch==2.3.0`.

For CPU
~~~~~~~~~~~~~

.. code-block:: bash

    pip install dgl==2.2.1 -f https://data.dgl.ai/wheels/torch-2.3/repo.html

For GPU (cuda 12.1)
~~~~~~~~~~~~~

.. code-block:: bash

    pip install dgl==2.2.1 -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html



Requirements
------------

- Python >= 3.8
- PyTorch == 2.3
- torch-geometric >= 2.6.0
- dgl == 2.2.1
- CUDA 12.1
