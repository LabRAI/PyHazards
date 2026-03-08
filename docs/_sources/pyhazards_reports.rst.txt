Reports
===================

Overview
--------

Use the reports layer when you want benchmark outputs exported in structured
formats that are easy to archive, compare, and publish.

What This Page Covers
---------------------

- ``pyhazards.reports`` exporters for JSON, CSV, and Markdown summaries
- how benchmark metrics and metadata are written to disk
- where report paths appear in ``BenchmarkRunSummary``

Typical Usage
-------------

.. code-block:: python

    from pyhazards.configs import load_experiment_config
    from pyhazards.engine import BenchmarkRunner

    config = load_experiment_config("pyhazards/configs/tc/hurricast_smoke.yaml")
    summary = BenchmarkRunner().run(config, output_dir="reports/tc_demo")
    print(summary.report_paths)

Why It Matters
--------------

The reports layer keeps hazard comparisons reproducible by exporting the same
metric and config snapshot structure across benchmark runs.

Next step: pair this page with :doc:`pyhazards_benchmarks` when you want to
inspect the evaluator contracts behind those report files.
