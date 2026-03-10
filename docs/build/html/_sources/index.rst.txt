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

.. image:: https://img.shields.io/badge/Slack-RAI%20Lab%20Channel-4A154B?logo=slack&logoColor=white
   :target: https://rai-lab-workspace.slack.com/archives/C0AKAJCTY4F
   :alt: Slack Channel

----

Overview
--------

PyHazards is an open-source Python library for AI-based natural hazard
modeling, providing unified interfaces for datasets, models, training
pipelines, benchmarks, and evaluation. It is designed for researchers,
practitioners, and contributors who need a consistent way to inspect hazard
data, build models, run benchmark-aligned experiments, and extend the library.

Core Capabilities
-----------------

- Inspect hazard datasets through consistent command-line inspection workflows,
  dataset reference pages, and linked benchmark adapters before training.
- Build registered models through a unified model registry with stable builder
  interfaces across hazard scenarios.
- Run benchmark-aligned smoke tests and experiment configs through shared
  benchmark, config, and report layers.
- Train, evaluate, and predict with shared engine interfaces so experiments and
  model comparisons follow a consistent workflow.

Hazard Coverage
---------------

- Wildfire workflows cover incident records, active-fire detections, fuels,
  burn products, danger forecasting, weekly forecasting, and spread baselines.
- Earthquake workflows cover waveform picking, dense-grid forecasting, and
  linked benchmark ecosystems for picking and forecasting.
- Flood workflows cover streamflow and inundation tasks with public benchmark
  adapters and model implementations.
- Tropical Cyclone workflows cover track-and-intensity forecasting with shared
  storm benchmark coverage and linked ecosystem pages.

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
- :doc:`pyhazards_models`: browse the public model catalog and benchmark-linked
  model coverage.
- :doc:`pyhazards_benchmarks`: run hazard-specific benchmark evaluators through
  the shared runner.
- :doc:`pyhazards_configs`: load or author reproducible experiment YAML files.
- :doc:`pyhazards_reports`: export JSON, Markdown, and CSV benchmark summaries.
- :doc:`pyhazards_engine`: use the shared training and inference runtime.
- :doc:`pyhazards_metrics`: review metrics used across benchmark and training
  paths.
- :doc:`pyhazards_utils`: access lower-level utility modules.
- :doc:`interactive_map`: open the wildfire-only companion map at
  ``https://rai-fire.com/``.
- :doc:`appendix_a_coverage`: audit which planned methods, benchmarks, and
  datasets are actually implemented today.
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

Community and Activity
----------------------

Use the `RAI Lab Slack channel <https://rai-lab-workspace.slack.com/archives/C0AKAJCTY4F>`_
for project discussion and coordination.

.. image:: https://api.star-history.com/svg?repos=LabRAI/PyHazards&type=Date&from=2026-01-01
   :target: https://www.star-history.com/#LabRAI/PyHazards&Date
   :alt: PyHazards Star History

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
   pyhazards_benchmarks
   pyhazards_configs
   pyhazards_reports
   pyhazards_engine
   pyhazards_metrics
   pyhazards_utils
   interactive_map

.. toctree::
   :maxdepth: 2
   :caption: Additional Information
   :hidden:

   implementation
   appendix_a_coverage
   cite
   references
   team
