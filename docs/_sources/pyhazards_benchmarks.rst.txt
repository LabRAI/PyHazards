Benchmarks
===================

Overview
--------

Use the benchmarks layer when you want hazard-specific evaluation contracts that
stay stable across datasets and models.

What This Page Covers
---------------------

- ``pyhazards.benchmarks`` registries and benchmark classes
- the shared ``BenchmarkRunner`` workflow
- hazard-task specific evaluation paths for earthquake, wildfire, flood, and tropical cyclone experiments

Typical Usage
-------------

.. code-block:: python

    from pyhazards.configs import load_experiment_config
    from pyhazards.engine import BenchmarkRunner

    config = load_experiment_config("pyhazards/configs/earthquake/phasenet_smoke.yaml")
    summary = BenchmarkRunner().run(config)
    print(summary.metrics)

Command-Line Entry Point
------------------------

Use ``python scripts/run_benchmark.py --help`` to inspect the shared benchmark
runner CLI and list available hazard tasks.

Next step: pair this page with :doc:`pyhazards_configs` when you want to author
or compare benchmark configs.
