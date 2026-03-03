FPA-FOD Tabular
===============

Incident-level tabular dataset derived from the Fire Program Analysis Fire-Occurrence Database (FPA-FOD),
packaged for wildfire cause or size classification in PyHazards.

Overview
--------

``fpa_fod_tabular`` converts one wildfire incident record into one feature vector and supports two
classification targets:

- ``task="cause"`` for incident cause prediction.
- ``task="size"`` for size-class prediction.

The implementation supports user-provided ``.sqlite`` / ``.db`` / ``.csv`` / ``.parquet`` inputs and
ships a deterministic ``micro=True`` mode so CI and local smoke tests do not require the full dataset.

PyHazards Usage
---------------

Load through the dataset registry:

.. code-block:: python

   from pyhazards.datasets import load_dataset

   data = load_dataset(
       "fpa_fod_tabular",
       task="cause",
       micro=True,
       normalize=True,
   ).load()

   train = data.get_split("train")
   print(train.inputs.shape, train.targets.shape)

Inspection entrypoint:

.. code-block:: bash

   python -m pyhazards.datasets.fpa_fod_tabular.inspection --task cause --micro

Notes
-----

- ``region="US"`` uses all states present in the source table.
- ``region="CA"`` restricts to California incidents only.
- ``cause_mode="paper5"`` keeps the five consolidated cause groups used by the FPA-FOD tabular model path.
- Returned split tensors follow the standard ``DataBundle`` contract:
  ``inputs`` shape ``(N, D)`` and integer ``targets`` shape ``(N,)``.
