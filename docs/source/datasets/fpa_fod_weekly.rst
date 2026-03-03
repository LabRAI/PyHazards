FPA-FOD Weekly
==============

Weekly wildfire forecasting dataset derived from FPA-FOD incident records, aggregated into next-week
count targets by size group.

Modular
-------

``fpa_fod_weekly``

Overview
--------

``fpa_fod_weekly`` builds rolling lookback windows from weekly incident counts and predicts the next
week's counts for the five grouped size classes ``A/B/C/D/EFG``.

The implementation supports user-provided ``.sqlite`` / ``.db`` / ``.csv`` / ``.parquet`` inputs and
includes a deterministic ``micro=True`` mode for smoke tests.

PyHazards Usage
---------------

Load through the dataset registry:

.. code-block:: python

   from pyhazards.datasets import load_dataset

   data = load_dataset(
       "fpa_fod_weekly",
       micro=True,
       features="counts+time",
       lookback_weeks=12,
   ).load()

   train = data.get_split("train")
   print(train.inputs.shape, train.targets.shape)

Inspection entrypoint:

.. code-block:: bash

   python -m pyhazards.datasets.fpa_fod_weekly.inspection --micro --lookback-weeks 12

Notes
-----

- ``features="counts"`` uses only the five weekly count channels.
- ``features="counts+time"`` adds sinusoidal week-of-year features for seasonality.
- Splits are chronological rather than stratified to preserve the forecasting setting.
- Returned split tensors follow the standard ``DataBundle`` contract:
  ``inputs`` shape ``(N, T, D)`` and float ``targets`` shape ``(N, 5)``.
