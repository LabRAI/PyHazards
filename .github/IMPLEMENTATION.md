# Model Implementation Guide for Collaborators

This guide explains how to port a model from an original paper repository into PyHazards with minimal friction and maximum reproducibility.

## 1. Start from a paper-to-library mapping

Before coding, build a short mapping table from the original repo:

- paper module/class name -> new `pyhazards/models/<model_name>.py` class
- paper training inputs/targets -> PyHazards `DataBundle` split format
- paper config keys -> builder kwargs/defaults in `register_model(...)`
- paper loss/metrics -> PyTorch loss and optional `pyhazards.metrics` usage

This avoids ad-hoc ports and makes review easier.

## 2. Define the PyHazards model contract first

In PyHazards, models are built with:

```python
from pyhazards.models import build_model
model = build_model(name="<model_name>", task="<task>", **kwargs)
```

Your builder must:

- accept `task: str`
- accept model hyperparameters (for example, `in_dim`, `hidden_dim`)
- return `nn.Module`
- validate unsupported tasks early with clear errors

For portability, always include `**kwargs` in the builder signature so extra config keys do not break the call path.

## 3. Implement the model module

Create `pyhazards/models/<model_name>.py` and include:

1. main model class inheriting `nn.Module`
2. optional helper blocks/losses
3. builder function `<model_name>_builder(...)`

Use explicit input-shape checks in `forward()` (existing models do this) so failures are actionable.

Template:

```python
from __future__ import annotations
import torch
import torch.nn as nn


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
            raise ValueError(f"Expected (B, F), got {tuple(x.shape)}")
        return self.net(x)


def my_model_builder(task: str, in_dim: int, out_dim: int, hidden_dim: int = 128, **kwargs) -> nn.Module:
    _ = kwargs
    if task.lower() not in {"classification", "regression"}:
        raise ValueError(f"MyModel does not support task='{task}'")
    return MyModel(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim)
```

## 4. Register the model in the registry

Edit `pyhazards/models/__init__.py`:

1. import the class and builder
2. add symbols to `__all__`
3. call `register_model(...)` with stable defaults

Example:

```python
from .my_model import MyModel, my_model_builder

register_model(
    "my_model",
    my_model_builder,
    defaults={"hidden_dim": 128},
)
```

If you skip registration, `build_model(name="my_model", ...)` will fail.

## 5. Match data format to your forward signature

`Trainer` supports two input patterns:

- tensor pairs: `inputs` and `targets` as tensors
- dataset objects: `inputs` as `torch.utils.data.Dataset` (recommended for graph/structured inputs)

For complex models (for example graph models), return dict-like batches from your dataset/collate function so `model(batch_dict)` works directly.

Use `DataBundle` metadata to make construction explicit:

- `FeatureSpec(input_dim=..., channels=...)`
- `LabelSpec(task_type="classification|regression|segmentation", num_targets=...)`

## 6. Port training logic carefully

Do not copy the paper repo training loop verbatim unless required. In most cases:

- keep model logic inside `nn.Module`
- use `pyhazards.engine.Trainer` for fit/evaluate/predict
- keep custom losses as separate classes in the model module

If the paper model needs custom multi-output behavior, document output shape and expected loss computation in the PR.

## 7. Add a reproducible smoke test

At minimum, verify:

1. model builds from registry
2. one forward pass succeeds with realistic tensor shapes
3. one short `Trainer.fit(...)` + `evaluate(...)` run works

Use existing examples (`test.py`, `pyhazards/models/hydrographnet.py`) as reference for strict shape checks and integration behavior.

## 8. Document the new model

Update docs so users can discover and run it:

1. add or update `pyhazards/model_cards/<model_name>.yaml`
2. keep the paper citation, usage snippet, and smoke-test spec in that card
3. set `include_in_public_catalog: false` in the card when a model should stay implemented but not appear in the public model table
4. run `python scripts/render_model_docs.py` if you want to preview the generated pages locally
5. when you need the published GitHub Pages site updated locally too, run:
   ```bash
   cd docs
   sphinx-build -b html source build/html
   cp -r build/html/* .
   ```

The model page and per-model docs are generated automatically from the card, including new
hazard-scenario tables when needed. Keep the card focused on I/O contract, supported tasks,
and one runnable example.

## 9. Recommended collaborator workflow

For each new paper model contribution:

1. open an issue with paper link + proposed API (`name`, `task`, required kwargs)
2. submit PR with model file, registry wiring, and smoke-test commands
3. include a short “paper parity note” listing intentional differences from the original repo (for example, optimizer, scheduler, or preprocessing changes)
4. complete the PR template so the automation bot can match the described model to the implementation

This keeps implementations reviewable and scientifically traceable.

## 10. Pre-PR checklist

- [ ] model file added under `pyhazards/models/`
- [ ] builder validates task and returns `nn.Module`
- [ ] model registered in `pyhazards/models/__init__.py`
- [ ] `pyhazards/model_cards/<model_name>.yaml` added or updated
- [ ] `build_model(name=..., task=...)` works
- [ ] forward pass shape checks and error messages are clear
- [ ] minimal train/eval smoke test executed
- [ ] PR template sections completed with paper/source, smoke-test, and parity notes

## 11. Automation setup

The PR automation added in `.github/workflows/` expects:

- repository workflow permissions that allow `contents: write` and `pull-requests: write`

Once configured, the workflow does the following for catalog-backed model PRs:

1. validate the PR against the model contract and smoke-test spec
2. comment with actionable blockers when the implementation is not ready
3. merge passing PRs automatically
4. regenerate the model page and module docs on the resulting push
5. rebuild the committed `docs/` HTML site so GitHub Pages reflects the new catalog
