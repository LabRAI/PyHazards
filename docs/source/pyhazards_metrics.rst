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

Hydrological Metrics
--------------------

``pyhazards.metrics.hydrology`` ports NeuralHydrology's evaluation metrics (BSD-3-Clause) for daily
discharge series: ``nse``, ``mse``, ``rmse``, ``kge``, ``alpha_nse``, ``beta_kge``, ``beta_nse``,
``pearson_r``, the flow-duration-curve biases ``fdc_fhv``, ``fdc_fms`` and ``fdc_flv``, and the peak
metrics ``mean_peak_timing``, ``missed_peaks`` and ``mean_absolute_percentage_peak_error``. Missing values
are skipped as in NeuralHydrology. ``calculate_metrics`` scores one basin and ``aggregate_basin_metrics``
reports the median and mean over basins; the flood benchmark (``flood.streamflow``) uses both.

.. code-block:: python

    import numpy as np
    from pyhazards.metrics.hydrology import calculate_metrics

    obs = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    sim = np.array([1.2, 2.5, 2.1, 4.0, 4.4])
    print(calculate_metrics(obs, sim, metrics=["nse", "kge"]))

Inundation Metrics
------------------

``pyhazards.metrics.inundation`` scores water-depth predictions for the flood benchmark
(``flood.inundation``): ``pixel_mae`` and ``rmse`` (depth errors over all cells), ``iou`` / ``f1`` of the wet
cells, and, per event as in the UrbanFloodCast evaluation (``DNO/utils25.py``), ``critical_success_index``
at 1 / 10 / 50 cm, ``relative_l2`` (FNO's relative L2 error per variable, averaged over variables),
``nash_sutcliffe`` and ``pearson_correlation`` over all predicted variables. ``rollout_rmse`` gives the depth
RMSE at every step of an autoregressive mesh rollout (HydroGraphNet's ``inference.py``). The definitions
are checked against the official code (oracle tests).

.. code-block:: python

    import torch
    from pyhazards.metrics.inundation import inundation_metrics

    pred = torch.rand(2, 32, 32, 24, 3)  # (events, Sy, Sx, steps, depth / qx / qy)
    target = torch.rand(2, 32, 32, 24, 3)
    print(inundation_metrics(pred, target, depth_index=0))

Usage
-----

.. code-block:: python

    from pyhazards.metrics import ClassificationMetrics

    metrics = [ClassificationMetrics()]
    # pass to Trainer or update metrics directly

Use this page together with :doc:`pyhazards_engine` if you want a consistent
train/evaluate workflow.
