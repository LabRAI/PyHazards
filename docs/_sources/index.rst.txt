.. image:: _static/logo.png
   :alt: PyHazards Icon
   :width: 260px
   :align: center

.. image:: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fpypi.org%2Fpypi%2Fpyhazards%2Fjson&query=%24.info.version&prefix=v&label=PyPI
   :target: https://pypi.org/project/pyhazards
   :alt: PyPI Version

.. image:: https://img.shields.io/github/actions/workflow/status/LabRAI/PyHazards/ci.yml?branch=main
   :target: https://github.com/LabRAI/PyHazards/actions/workflows/ci.yml
   :alt: Build Status

.. image:: https://img.shields.io/badge/license-MIT-green
   :target: https://github.com/LabRAI/PyHazards/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/badge/downloads-check%20PyPI-blue
   :target: https://pypi.org/project/pyhazards
   :alt: PyPI Downloads

.. image:: https://img.shields.io/github/issues/LabRAI/PyHazards
   :target: https://github.com/LabRAI/PyHazards
   :alt: Issues

.. image:: https://img.shields.io/github/issues-pr/LabRAI/PyHazards
   :target: https://github.com/LabRAI/PyHazards
   :alt: Pull Requests

.. image:: https://img.shields.io/github/stars/LabRAI/PyHazards
   :target: https://github.com/LabRAI/PyHazards
   :alt: Stars

.. image:: https://img.shields.io/github/forks/LabRAI/PyHazards
   :target: https://github.com/LabRAI/PyHazards
   :alt: GitHub forks

.. image:: _static/github.svg
   :target: https://github.com/LabRAI/PyHazards
   :alt: GitHub

----

Overview
--------

PyHazards is a Python library for hazard-focused machine learning. It provides a
consistent interface for dataset inspection, model construction, experiment
execution, and hazard-specific extensions.

The documentation is organized around two common workflows:

1. **Use the library**: install PyHazards, inspect data, build a model, and run a
   short experiment.
2. **Extend the library**: add a dataset, register a model, and keep the docs and
   smoke tests aligned with the existing project structure.

What You Can Do with PyHazards
------------------------------

- Inspect hazard datasets through consistent CLI entrypoints and dataset pages.
- Build registered models by name instead of wiring architectures manually.
- Train, evaluate, and predict with a shared ``Trainer`` interface.
- Publish hazard-specific model docs and examples through the library's catalog
  workflow.

Minimal Example
---------------

Build a registered model in one step:

.. code-block:: python

    from pyhazards.models import build_model

    model = build_model(
        name="hydrographnet",
        task="regression",
        node_in_dim=2,
        edge_in_dim=3,
        out_dim=1,
    )

Start Here
----------

If you are new to PyHazards, use the docs in this order:

1. :doc:`installation` for environment setup and verification.
2. :doc:`quick_start` for the first runnable workflow.
3. :doc:`pyhazards_datasets` and :doc:`pyhazards_models` to browse the built-in
   dataset and model references.

Documentation Map
-----------------

- :doc:`installation`: install from PyPI or source and verify the package.
- :doc:`quick_start`: run one small end-to-end example.
- :doc:`pyhazards_datasets`: browse datasets and inspection commands.
- :doc:`pyhazards_models`: browse the public model catalog and registry usage.
- :doc:`interactive_map`: open the companion wildfire map at
  ``https://rai-fire.com/``.
- :doc:`implementation`: contributor-oriented guidance for adding datasets and
  models.

For Contributors
----------------

PyHazards is registry-driven. If you plan to contribute a new dataset or model,
start with :doc:`implementation` and then use the reference pages for the
relevant subsystem.

How to Cite
-----------

If you use PyHazards in your research, please cite:

.. code-block:: bibtex

   @misc{pyhazards2025,
     title        = {PyHazards: An Open-Source Library for AI-Powered Hazard Prediction},
     author       = {Cheng, Xueqi and Xu, Yangshuang and Xu, Runyang and Schneier, Lex and Kodudula, Sharan Kumar Reddy and Hsu, Deyang and Shen, Dacheng and Dong, Yushun},
     year         = {2025},
     howpublished = {\url{https://github.com/LabRAI/PyHazards}},
     note         = {GitHub repository}
   }


.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   installation
   quick_start

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   pyhazards_datasets
   pyhazards_models
   interactive_map
   pyhazards_engine
   pyhazards_metrics
   pyhazards_utils

.. toctree::
   :maxdepth: 2
   :caption: Additional Information
   :hidden:

   implementation
   cite
   references
   team
