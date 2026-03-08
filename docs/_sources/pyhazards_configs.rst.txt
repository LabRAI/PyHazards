Configs
===================

Overview
--------

Use the configs layer when you want reproducible experiment specifications for
benchmark runs, smoke tests, and hazard-specific model comparisons.

What This Page Covers
---------------------

- ``pyhazards.configs`` dataclasses and YAML loading helpers
- hazard-scoped smoke configs under ``pyhazards/configs/<hazard>/``
- the shared structure for benchmark, dataset, model, and report settings

Typical Usage
-------------

.. code-block:: python

    from pyhazards.configs import load_experiment_config

    config = load_experiment_config("pyhazards/configs/flood/hydrographnet_smoke.yaml")
    print(config.benchmark.hazard_task)
    print(config.model.name)

Config Layout
-------------

Each experiment config contains four sections:

- ``benchmark``: which evaluator to run and which hazard task to score
- ``dataset``: which registered dataset to load and with which parameters
- ``model``: which registered model to build and with which parameters
- ``report``: where to write JSON, Markdown, or CSV outputs

Next step: pair this page with :doc:`pyhazards_benchmarks` when you want to
match configs to implemented evaluation paths, and with
:doc:`pyhazards_reports` when you want to export benchmark outputs.
