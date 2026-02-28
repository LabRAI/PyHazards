FPA FOD — Weekly Forecasting Dataset
====================================

What it is
----------

This dataset provides weekly sequences for forecasting wildfire activity.
One sample corresponds to a sequence of weekly feature vectors.

Data source and licensing
-------------------------

The raw FPA FOD dataset must be obtained by the user. PyHazards does not ship the raw data due to
licensing/size constraints.

Returned tensors (contract)
---------------------------

A single sample returns:
- ``x``: float tensor of shape ``[T, input_dim]``
- ``y``: float tensor of shape ``[out_dim]`` (or scalar)

A batch returns:
- ``x``: ``[B, T, input_dim]``
- ``y``: ``[B, out_dim]``

Micro dataset for CI
--------------------

For CI/testing: set ``micro=True`` to use deterministic synthetic data (small B and T) to validate shapes,
dtypes, and a trainer smoke run, without requiring the raw FPA FOD sqlite.
