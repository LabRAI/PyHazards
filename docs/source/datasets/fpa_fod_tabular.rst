FPA FOD — Tabular Classification Dataset
========================================

What it is
----------

This dataset provides a tabular supervised-learning view derived from FPA FOD wildfire records.
One sample corresponds to one incident (or one processed record).

Data source and licensing
-------------------------

The raw FPA FOD dataset is large and must be obtained by the user. PyHazards does not ship the raw data
due to licensing/size constraints.

How to provide data
-------------------

1. Obtain the FPA FOD sqlite (user-provided).
2. Place it at the path expected by the dataset builder (see project README/scripts).
3. Run the preprocess/build step to produce processed dataset artifacts.

Returned tensors (contract)
---------------------------

A single sample returns:
- ``x``: float tensor of shape ``[input_dim]``
- ``y``: integer class id (scalar)

A batch returns:
- ``x``: ``[B, input_dim]``
- ``y``: ``[B]``

Micro dataset for CI
--------------------

For CI/testing: set ``micro=True`` to use deterministic synthetic data that matches the schema and shapes,
so tests run without requiring the raw FPA FOD sqlite.
