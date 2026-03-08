Metrics
===================

Summary
-------

Built-in metrics cover common tasks and are designed to aggregate predictions/targets over a full split.

Core classes
------------

- ``MetricBase`` — abstract interface with ``update``, ``compute``, ``reset``.
- ``ClassificationMetrics`` — accuracy.
- ``RegressionMetrics`` — MAE, RMSE.
- ``SegmentationMetrics`` — pixel accuracy (extend to IoU/Dice as needed).

Usage
-----

.. code-block:: python

    from pyhazards.metrics import ClassificationMetrics

    metrics = [ClassificationMetrics()]
    # pass to Trainer or call directly
