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

PyHazards is an open-source Python library for AI-based natural hazard
modeling, providing unified interfaces for datasets, models, training
pipelines, and evaluation. It is designed for researchers, practitioners, and
contributors who need a consistent way to inspect data, build hazard models,
run experiments, and extend the library.

Core Capabilities
-----------------

- Inspect hazard datasets through consistent command-line inspection workflows,
  dataset reference pages, and lightweight validation steps before training.
- Build registered models through a unified model registry with stable builder
  interfaces across hazard scenarios.
- Train, evaluate, and predict with shared engine interfaces so experiments,
  smoke tests, and model comparisons follow a consistent workflow.

Getting Started
---------------

If you are new to PyHazards, use the documentation in this order:

1. :doc:`installation` - set up the environment and verify the package.
2. :doc:`quick_start` - run a first end-to-end workflow.
3. :doc:`pyhazards_datasets` and :doc:`pyhazards_models` - explore supported
   data sources and model implementations.

Illustrative Example
~~~~~~~~~~~~~~~~~~~~

The example below shows how to instantiate a registered model through the
unified model registry.

.. code-block:: python

    from pyhazards.models import build_model

    model = build_model(
        name="hydrographnet",
        task="regression",
        node_in_dim=2,
        edge_in_dim=3,
        out_dim=1,
    )

Documentation Guide
-------------------

- :doc:`installation`: set up PyHazards from PyPI or source and verify the
  environment.
- :doc:`quick_start`: run the shortest end-to-end workflow in the library.
- :doc:`pyhazards_datasets`: browse the hazard-grouped dataset catalog, linked
  sources, and inspection or registry entry points.
- :doc:`pyhazards_models`: browse the public model catalog and registry usage.
- :doc:`interactive_map`: open the wildfire-only companion map at
  ``https://rai-fire.com/``.
- :doc:`pyhazards_benchmarks`: run hazard-specific benchmark evaluators through
  the shared runner.
- :doc:`appendix_a_coverage`: audit which planned methods, benchmarks, and
  datasets are actually implemented today.
- :doc:`pyhazards_configs`: load or author reproducible experiment YAML files.
- :doc:`pyhazards_reports`: export JSON, Markdown, and CSV benchmark summaries.
- :doc:`implementation`: use the contributor guide for adding datasets, models,
  and public catalog updates.

For Contributors
----------------

PyHazards is registry-driven and uses dataset cards, model cards, and benchmark
cards to generate the public catalogs. If you plan to contribute a dataset or
model, start with :doc:`implementation` and then use the dataset, model, and
benchmark reference pages to keep registry wiring, smoke tests, and generated
docs aligned with the library workflow. Use :doc:`appendix_a_coverage` when
you need the audited gap list for the remaining roadmap work.

Citation
--------

If you use PyHazards in your research, please cite:

.. code-block:: bibtex

   @misc{pyhazards2025,
     title        = {PyHazards: An Open-Source Library for AI-Powered Hazard Prediction},
     author       = {Cheng et al.},
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
   pyhazards_benchmarks
   pyhazards_configs
   pyhazards_reports
   pyhazards_engine
   pyhazards_metrics
   pyhazards_utils

.. toctree::
   :maxdepth: 2
   :caption: Additional Information
   :hidden:

   implementation
   appendix_a_coverage
   cite
   references
   team
