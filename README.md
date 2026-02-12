# PyHazards

[![PyPI - Version](https://img.shields.io/pypi/v/pyhazards)](https://pypi.org/project/pyhazards)
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

- **Datasets**: Unified interfaces for tabular, temporal, raster, and graph-style hazard data through `DataBundle`.
- **Models**: Built-in hazard models plus reusable backbones/heads via a registry-driven model API.
- **Engine**: `Trainer` for fit/evaluate/predict workflows with mixed precision and distributed options.
- **Metrics and Utilities**: Classification/regression/segmentation metrics, hardware helpers, and reproducibility tools.

## Install

```bash
pip install pyhazards
```

Optional CUDA setup:

```bash
export PYHAZARDS_DEVICE=cuda:0
```

## Load Data

Example using the implemented ERA5 flood subset loader:

```python
from pyhazards.data.load_hydrograph_data import load_hydrograph_data

data = load_hydrograph_data(
    era5_path="pyhazards/data/era5_subset",
    max_nodes=50,
)
print(data.feature_spec)
print(data.label_spec)
print(list(data.splits.keys()))  # ["train"]
```

## Load Model

Example using `wildfire_aspp`:

```python
from pyhazards.models import build_model

model = build_model(
    name="wildfire_aspp",
    task="segmentation",
    in_channels=12,
)
print(type(model).__name__)
```

## Full Test

Short end-to-end example using real ERA5 data and an implemented flood model:

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

trainer.fit(
    data,
    optimizer=optimizer,
    loss_fn=loss_fn,
    max_epochs=1,
    batch_size=1,
    collate_fn=graph_collate,
)

metrics = trainer.evaluate(
    data,
    split="train",
    batch_size=1,
    collate_fn=graph_collate,
)
print(metrics)
```

## Custom Module

To upload and use your own data/model modules:

1. Upload your raw data files to your project path and write a dataset loader that returns a `DataBundle`.
2. Register your model with `register_model` and a builder function that returns an `nn.Module`.
3. Build with `build_model(...)` and train/evaluate through `Trainer`.

Implementation details:

- [Implementation Guideline](.github/IMPLEMENTATION.md)
- [Contributors Guideline](.github/CONTRIBUTING.md)

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
