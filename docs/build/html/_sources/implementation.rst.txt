Implementation Guide
====================

Use this guide when you want to extend PyHazards itself. It is written for
contributors who are adding new datasets, new models, smoke tests, catalog
cards, or documentation updates for the public site.

This page explains the public contributor workflow. For repository operations
and maintainer automation details, also see ``.github/IMPLEMENTATION.md``.

Who This Guide Is For
---------------------

This guide assumes you already know Python and PyTorch, but you have not yet
worked inside the PyHazards codebase. It is most useful when you are doing one
of the following:

- adding a new dataset loader or dataset inspection entrypoint,
- porting a paper or external implementation into ``pyhazards.models``,
- updating the public dataset or model catalogs and generated documentation,
- preparing a pull request that should be easy to review and merge.

If you only want to install the library and run a first example, use
:doc:`installation` and :doc:`quick_start` instead.

Repository Mental Model
-----------------------

PyHazards is organized around a small set of extension points:

- ``pyhazards.datasets`` contains dataset abstractions, the dataset registry,
  and inspection entrypoints for supported data sources.
- ``pyhazards.models`` contains model builders, reusable components, and the
  model registry used by ``build_model(...)``.
- ``pyhazards.engine`` contains the shared training and evaluation workflow.
- ``pyhazards/dataset_cards`` contains YAML cards used to generate the public
  dataset catalog and per-dataset documentation pages.
- ``pyhazards/model_cards`` contains YAML cards used to generate the public
  model tables and per-model documentation pages.
- ``docs/source`` contains handwritten Sphinx pages, while the committed
  ``docs/`` directory contains the rendered HTML published on GitHub Pages.

There are three separate layers to keep in mind:

1. registry availability:
   a dataset or model can be constructed from Python once it is registered;
2. catalog visibility:
   a public dataset or model only appears on the website when it also has a
   matching catalog card;
3. published website output:
   GitHub Pages only changes after the rendered HTML in ``docs/`` is rebuilt.

Typical Contribution Workflow
-----------------------------

Most changes should follow the same sequence:

1. decide whether you are extending a dataset, a model, or both;
2. implement the code in ``pyhazards/datasets`` or ``pyhazards/models``;
3. register the new entrypoint so it is discoverable from the library API;
4. add or update smoke-test coverage for the new behavior;
5. update the relevant docs source and, for public datasets or models, the
   matching catalog cards;
6. run the smallest local validation commands that match the change;
7. rebuild the published docs HTML if the website output changed;
8. open a pull request with the required metadata and validation notes.

Treat code, validation, generated docs, and published docs as one contribution.
A public dataset or model implementation is not complete if users cannot
discover it or if the website catalog still describes the old state of the
library.

Adding a Dataset
----------------

Datasets are built around ``Dataset`` and ``DataBundle``. A dataset subclass
implements ``_load()`` and returns train/validation/test splits plus feature and
label metadata.

The minimum pattern looks like this:

.. code-block:: python

    import torch
    from pyhazards.datasets import (
        DataBundle,
        DataSplit,
        Dataset,
        FeatureSpec,
        LabelSpec,
        register_dataset,
    )

    class MyHazardDataset(Dataset):
        name = "my_hazard"

        def _load(self) -> DataBundle:
            x = torch.randn(1000, 16)
            y = torch.randint(0, 2, (1000,))

            splits = {
                "train": DataSplit(x[:800], y[:800]),
                "val": DataSplit(x[800:900], y[800:900]),
                "test": DataSplit(x[900:], y[900:]),
            }

            return DataBundle(
                splits=splits,
                feature_spec=FeatureSpec(
                    input_dim=16,
                    description="Example tabular hazard features.",
                ),
                label_spec=LabelSpec(
                    num_targets=2,
                    task_type="classification",
                    description="Binary hazard label.",
                ),
            )

    register_dataset(MyHazardDataset.name, MyHazardDataset)

Keep the following expectations in mind when you add a dataset:

- use ``DataBundle`` to make split names, feature dimensions, and target
  semantics explicit;
- keep the builder/import path lightweight so the dataset can be imported
  without triggering heavy side effects;
- register the dataset with ``register_dataset(...)`` so
  ``load_dataset(name=...)`` can construct it;
- if the dataset belongs in the public catalog, add or update a card in
  ``pyhazards/dataset_cards`` and regenerate the dataset docs;
- prefer clear metadata over implicit conventions, especially when a model
  depends on shapes, channels, graph structure, or task type.

Dataset Inspection Entry Points
-------------------------------

PyHazards also includes inspection modules under ``pyhazards.datasets`` for
supported external data sources. If you add a new dataset family, keep the
inspection module consistent with the existing ones:

- it should be importable as ``python -m pyhazards.datasets.<name>.inspection``;
- ``--help`` should exit cleanly;
- argument parsing should work without requiring optional plotting or network
  dependencies at import time;
- if the dataset belongs in the public dataset table, its inspection workflow
  should be stable enough for ``scripts/verify_table_entries.py``.

The goal is simple: users should be able to discover the dataset from the docs,
inspect it from the command line, and load it from Python through the registry.

Dataset Cards and Generated Docs
--------------------------------

Public datasets are documented through cards in ``pyhazards/dataset_cards``.
These cards are the source of truth for the public dataset catalog and the
generated per-dataset detail pages.

A typical dataset card includes:

- the public display name and hazard family,
- a one-sentence summary and source role,
- provider, geometry, cadence, and period-of-record metadata,
- the primary source or product reference,
- the inspection command when the dataset is inspection-first,
- the registry name and example when it is public through
  ``load_dataset(...)``,
- related model and benchmark links when those cross-links help users navigate
  the library.

After updating dataset cards, refresh the generated docs:

.. code-block:: bash

    python scripts/render_dataset_docs.py

Use the ``--check`` mode when you want to confirm the generated files are
already up to date:

.. code-block:: bash

    python scripts/render_dataset_docs.py --check

Adding a Model
--------------

Models are registered builders that can be constructed through:

.. code-block:: python

    from pyhazards.models import build_model

    model = build_model(name="<model_name>", task="<task>", **kwargs)

When you port a paper or external repository into PyHazards, define the library
contract first. Your builder should:

- accept ``task: str``,
- accept the shape and hyperparameter arguments needed to construct the model,
- return an ``nn.Module``,
- validate unsupported tasks early with a clear error,
- accept ``**kwargs`` so extra configuration keys do not break the call path.

The minimum pattern looks like this:

.. code-block:: python

    from __future__ import annotations

    import torch
    import torch.nn as nn
    from pyhazards.models import register_model


    class MyModel(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.ndim != 2:
                raise ValueError(f"Expected input of shape (batch, features), got {tuple(x.shape)}")
            return self.net(x)


    def my_model_builder(
        task: str,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        **kwargs,
    ) -> nn.Module:
        _ = kwargs
        if task.lower() not in {"classification", "regression"}:
            raise ValueError(f"MyModel does not support task={task!r}")
        return MyModel(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim)


    register_model(
        "my_model",
        my_model_builder,
        defaults={"hidden_dim": 128},
    )

In practice, good model ports also include:

- a short paper-to-library mapping from the original repository into the new
  PyHazards module and builder kwargs;
- explicit input-shape validation in ``forward()`` so integration failures are
  easy to diagnose;
- clear task handling when the same architecture can be used for different
  objectives;
- minimal defaults in the registry so ``build_model(...)`` is predictable.

Match the Forward Signature to the Data Path
--------------------------------------------

PyHazards supports more than one input style. Some models work with plain tensor
pairs, while others expect mappings, graph batches, or custom dataset objects.
Make that contract explicit.

As a rule:

- if your model expects ``Tensor -> Tensor``, keep the shape assumptions simple
  and document them in the model card;
- if your model expects graph or structured inputs, prefer dataset and collate
  behavior that produces the mapping your ``forward()`` already consumes;
- use ``FeatureSpec``, ``LabelSpec``, and split metadata to record dimensions,
  channels, and task semantics instead of burying them in comments.

Porting Training Logic
----------------------

Do not copy an upstream training loop into PyHazards unless the architecture
truly depends on custom runtime behavior. In most cases you should:

- keep the architecture inside ``nn.Module``,
- keep custom losses or helper blocks close to the model implementation,
- use ``pyhazards.engine.Trainer`` for fit, evaluate, and predict workflows,
- document intentional differences from the paper repository in the pull request.

If the PyHazards port changes preprocessing, outputs, or optimization behavior,
state that clearly in the PR's parity notes. Review is much faster when the
intended differences are explicit.

Model Cards and Generated Docs
------------------------------

Public models are documented through cards in ``pyhazards/model_cards``. A model
card is not optional when you want a model to appear on the website.

A typical card includes:

- the public model name and display name,
- the hazard family used for the model table,
- the source file and builder name,
- a short summary and description,
- the paper citation or technical reference,
- supported tasks,
- one runnable example,
- a synthetic smoke-test specification.

For example:

.. code-block:: yaml

    model_name: my_model
    display_name: My Model
    hazard: Flood
    source_file: pyhazards/models/my_model.py
    builder_name: my_model_builder
    summary: >
      Short description of the public model entrypoint.
    paper:
      title: Example paper title
      url: https://example.com/paper
    tasks:
      - regression
    smoke_test:
      task: regression
      build_kwargs:
        in_dim: 16
        out_dim: 1
      input:
        kind: tensor
        shape: [4, 16]
      expected_output:
        kind: tensor
        shape: [4, 1]

Model cards drive the generated pages in :doc:`pyhazards_models`. They also
control public visibility:

- if a model is registered but has no card, it can still be used from Python but
  it will not appear in the public model tables;
- if a card sets ``include_in_public_catalog: false``, the implementation stays
  in the library but is hidden from the public catalog;
- if the hazard name in the card is new, the generated model page creates a new
  hazard section automatically.

After updating a card, refresh the generated docs:

.. code-block:: bash

    python scripts/render_model_docs.py

Use the ``--check`` mode when you want to confirm the generated files are
already up to date:

.. code-block:: bash

    python scripts/render_model_docs.py --check

Validation Workflow
-------------------

Run the smallest set of checks that covers your change. The core validation
commands in this repository are:

.. code-block:: bash

    python -c "import pyhazards; print(pyhazards.__version__)"
    python scripts/render_dataset_docs.py --check
    python scripts/render_model_docs.py --check
    python scripts/verify_table_entries.py

Use them for the following purposes:

- ``python -c "import pyhazards; print(pyhazards.__version__)"``
  verifies that the package still imports cleanly;
- ``python scripts/render_dataset_docs.py --check``
  verifies that generated dataset docs and catalog pages are in sync with the
  current dataset cards;
- ``python scripts/render_model_docs.py --check``
  verifies that generated model docs and catalog pages are in sync with the
  current model cards;
- ``python scripts/verify_table_entries.py``
  exercises dataset inspection entrypoints and runs smoke tests for cataloged
  public models.

When you changed a specific model, also run the model-scoped smoke test:

.. code-block:: bash

    python scripts/smoke_test_models.py --models <model_name>

This uses the model card's smoke-test spec, so it is the fastest way to confirm
that a new public model can build and run with synthetic inputs.

If your change touched the model catalog or its generation logic, also run:

.. code-block:: bash

    python -m pytest tests/test_model_catalog.py

If you changed runtime behavior in the training path and you have the required
hardware available, run the broader smoke path described in ``test.py`` as well.

Preparing a Model Pull Request
------------------------------

Model PRs should make the implementation easy to review against the original
paper or upstream repository. The PR template asks for a few specific fields for
that reason:

- ``Model Summary`` should describe the architecture and public API you are
  adding, not just the file names you changed;
- ``Hazard Scenario`` should name the model table that owns the entry, and it
  should explicitly call out when the PR introduces a new hazard family;
- ``Registry Name`` should list the exact ``build_model(name=...)`` entrypoints
  added or changed in the PR;
- ``Paper / Source`` should link the scientific paper, source repository, or
  technical reference that the implementation follows;
- ``Smoke Test`` should list the commands you ran or point to the card's
  smoke-test specification;
- ``Parity Notes`` should explain intentional differences from the upstream
  implementation, especially around preprocessing, outputs, or objectives.

PR automation can only help when this metadata is present and accurate. A
catalog-backed model PR is expected to include the implementation, the registry
wiring, the model card, the smoke-test path, and refreshed generated docs.

Registration, Catalog, and Published HTML
-----------------------------------------

It is easy to update one layer of the repo and forget the others. Keep this
distinction in mind:

- code registration makes a dataset or model usable from Python;
- dataset cards make a public dataset discoverable in the generated docs;
- model cards make a public model discoverable in the generated docs;
- Sphinx source updates change the documentation source tree;
- rebuilding ``docs/`` updates the committed HTML published on GitHub Pages.

If the website output changed, rebuild the site locally:

.. code-block:: bash

    cd docs
    sphinx-build -b html source build/html
    cp -r build/html/* .

That final copy step matters in this repository because the published website is
served from the committed ``docs/`` directory, not from ``docs/source``.

Common Mistakes
---------------

These are the issues that most often block review:

- the new dataset or model exists in code but was never registered;
- a public dataset changed, but ``pyhazards/dataset_cards`` or the generated
  dataset docs were not updated;
- a public model was implemented without a matching card in
  ``pyhazards/model_cards``;
- generated docs were not refreshed after the model card changed;
- ``docs/source`` was updated but the committed ``docs/`` HTML was not rebuilt;
- the builder does not validate unsupported tasks or accepts the wrong shape
  arguments for the intended use;
- a hidden or internal model was accidentally left visible in the public
  catalog;
- an inspection module imports optional heavy dependencies at module import time,
  which breaks ``python -m ... --help`` in clean environments.

Contributor Checklist
---------------------

Before you open a pull request, confirm all of the following:

- the implementation lives in the correct dataset or model module;
- the new entrypoint is registered and can be constructed from the public API;
- task handling and input-shape validation are clear and actionable;
- public datasets have a complete card when they belong in the public catalog;
- generated dataset docs are refreshed and pass ``render_dataset_docs.py --check``;
- public models have a complete card with a runnable smoke-test spec;
- generated model docs are refreshed and pass ``render_model_docs.py --check``;
- dataset inspection entrypoints and public tables pass
  ``scripts/verify_table_entries.py``;
- the published docs HTML in ``docs/`` was rebuilt if the visible website output
  changed;
- the pull request explains the source paper, registry name, hazard scenario,
  smoke-test commands, and parity notes.

Next Steps
----------

After you finish a contributor-oriented change:

- browse the public catalogs in :doc:`pyhazards_datasets` and
  :doc:`pyhazards_models` to confirm the new entry is discoverable;
- use :doc:`quick_start` to check that the user path still feels coherent;
- keep ``.github/IMPLEMENTATION.md`` and this page aligned when the repository
  workflow changes.
