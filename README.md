# PyHazards

[![PyPI - Version](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fpypi.org%2Fpypi%2Fpyhazards%2Fjson&query=%24.info.version&prefix=v&label=PyPI)](https://pypi.org/project/pyhazards)
[![Build Status](https://img.shields.io/github/actions/workflow/status/LabRAI/PyHazards/ci.yml?branch=main)](https://github.com/LabRAI/PyHazards/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/LabRAI/PyHazards/blob/main/LICENSE)
[![PyPI - Downloads](https://img.shields.io/badge/downloads-check%20PyPI-blue)](https://pypi.org/project/pyhazards)
[![Issues](https://img.shields.io/github/issues/LabRAI/PyHazards)](https://github.com/LabRAI/PyHazards/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/LabRAI/PyHazards)](https://github.com/LabRAI/PyHazards/pulls)
[![Stars](https://img.shields.io/github/stars/LabRAI/PyHazards)](https://github.com/LabRAI/PyHazards)
[![GitHub forks](https://img.shields.io/github/forks/LabRAI/PyHazards)](https://github.com/LabRAI/PyHazards)

## Introduction

PyHazards is a Python framework for AI-powered hazard prediction and risk assessment. It provides a hazard-first API for loading data, building models, running end-to-end experiments, and extending with your own modules.

## Core Components

Use this as the minimal path: install the package, load one implemented dataset, build one implemented model, then run one end-to-end validation test.

- **Datasets**: Unified interfaces for tabular, temporal, raster, and graph-style hazard data through `DataBundle`.
- **Models**: Built-in hazard models plus reusable backbones/heads via a registry-driven model API.
- **Engine**: `Trainer` for fit/evaluate/predict workflows with mixed precision and distributed options.
- **Metrics and Utilities**: Classification/regression/segmentation metrics, hardware helpers, and reproducibility tools.

## Install

Install from PyPI. If you plan to run on GPU, install a compatible PyTorch build first.

```bash
pip install pyhazards
```

Optional CUDA setup:

```bash
export PYHAZARDS_DEVICE=cuda:0
```

## Load Data

Use the dataset module entrypoint for consistent dataset inspection/loading across datasets.

```bash
python -m pyhazards.datasets.era5.inspection --path pyhazards/data/era5_subset --max-vars 10
```

## Load Model

Build one implemented model from the registry (this example uses the wildfire model).

Example using `wildfire_aspp`:

```python
from pyhazards.models import build_model

print("[Step 2/3] Building model...")
model = build_model(
    name="wildfire_aspp",
    task="segmentation",
    in_channels=12,
)
print("[Step 2/3] Model built.")
print(type(model).__name__)
```

## Full Test

Validation example: load the same ERA5-based hydrograph subset and run one epoch with `hydrographnet`.
This uses `load_hydrograph_data` as the model-specific adapter after dataset inspection.

```python
import torch
from pyhazards.data.load_hydrograph_data import load_hydrograph_data
from pyhazards.datasets import graph_collate
from pyhazards.engine import Trainer
from pyhazards.models import build_model

data = load_hydrograph_data("pyhazards/data/era5_subset", max_nodes=50)

model = build_model(
    name="hydrographnet",
    task="regression",
    node_in_dim=2,
    edge_in_dim=3,
    out_dim=1,
)

trainer = Trainer(model=model, mixed_precision=False)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

print("[Step 3/3] Running one training epoch...")
trainer.fit(
    data,
    optimizer=optimizer,
    loss_fn=loss_fn,
    max_epochs=1,
    batch_size=1,
    collate_fn=graph_collate,
)

print("[Step 3/3] Evaluating on train split...")
metrics = trainer.evaluate(
    data,
    split="train",
    batch_size=1,
    collate_fn=graph_collate,
)
print(metrics)
```

## Quick Verification (`test.py`)

Run the built-in GPU smoke test:

```bash
python test.py
```

`test.py` is a validation/smoke test only. It verifies pipeline correctness and integration, not final benchmark performance.

It prints step-by-step status (dataset load, model build, forward pass, train/eval) and ends with:

```text
PASS: end-to-end implementation is working.
```

## Custom Module

Use this when you want to add your own dataset/model implementation into PyHazards.

To upload and use your own data/model modules:

1. Upload your raw data files to your project path and write a dataset loader that returns a `DataBundle`.
2. Register your model with `register_model` and a builder function that returns an `nn.Module`.
3. Build with `build_model(...)` and train/evaluate through `Trainer`.

Implementation details:

- [Implementation Guideline](.github/IMPLEMENTATION.md)
- [Contributors Guideline](.github/CONTRIBUTING.md)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=LabRAI/PyHazards&type=Date&from=2026-01-01)](https://www.star-history.com/#LabRAI/PyHazards&Date)

## How to Cite

If you use PyHazards in your research, please cite:

```bibtex
@software{pyhazards2025,
  title={PyHazards: A Python Framework for AI-Powered Hazard Prediction},
  author={Cheng, Xueqi},
  year={2025},
  url={https://github.com/LabRAI/PyHazards}
}
```

## Documentation

Full documentation is available at: [https://labrai.github.io/PyHazards](https://labrai.github.io/PyHazards)

## License

[MIT License](LICENSE)

## Contact

For questions or contributions, please contact xc25@fsu.edu.
