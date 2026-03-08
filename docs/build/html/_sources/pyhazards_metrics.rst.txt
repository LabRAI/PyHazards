Metrics
===================

Overview
--------

PyHazards includes small, task-oriented metric classes that accumulate
predictions and targets across a full split.

Core Classes
------------

- ``MetricBase``: shared interface with ``update``, ``compute``, and ``reset``.
- ``ClassificationMetrics``: basic classification metrics such as accuracy.
- ``RegressionMetrics``: MAE and RMSE style regression summaries.
- ``SegmentationMetrics``: segmentation-oriented aggregation.

Usage
-----

.. code-block:: python

    from pyhazards.metrics import ClassificationMetrics

    metrics = [ClassificationMetrics()]
    # pass to Trainer or update metrics directly

Use this page together with :doc:`pyhazards_engine` if you want a consistent
train/evaluate workflow.
