Engine
===================

Overview
--------

Use the engine when you want a shared interface for training, evaluation, and
prediction without rewriting the loop for every hazard task.

Core modules
------------

- ``pyhazards.engine.trainer``: the ``Trainer`` class with ``fit``,
  ``evaluate``, and ``predict``.
- ``pyhazards.engine.distributed``: distributed-strategy helpers.
- ``pyhazards.engine.inference``: inference utilities for large grids or
  sliding-window style workflows.

Typical Usage
-------------

.. code-block:: python

    import torch
    from pyhazards.engine import Trainer
    from pyhazards.metrics import ClassificationMetrics
    from pyhazards.models import build_model

    model = build_model(name="mlp", task="classification", in_dim=16, out_dim=2)
    trainer = Trainer(model=model, metrics=[ClassificationMetrics()], mixed_precision=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    trainer.fit(data_bundle, optimizer=optimizer, loss_fn=loss_fn, max_epochs=10)
    results = trainer.evaluate(data_bundle, split="test")
    preds = trainer.predict(data_bundle, split="test")

Device and Distributed Notes
----------------------------

- ``Trainer(strategy="auto")`` uses DDP when multiple GPUs are available; otherwise runs single-device.
- ``mixed_precision=True`` enables AMP when on CUDA.
- Device selection is handled via ``pyhazards.utils.hardware.auto_device`` by default.

Next step: pair this page with :doc:`pyhazards_metrics` and
:doc:`pyhazards_utils` when you want to customize evaluation or device behavior.
