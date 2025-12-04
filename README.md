# PyHazards

[![PyPI - Version](https://img.shields.io/pypi/v/pyhazards)](https://pypi.org/project/pyhazards)
[![Build Status](https://img.shields.io/github/actions/workflow/status/LabRAI/PyHazards/docs.yml)](https://github.com/LabRAI/PyHazards/actions)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/LabRAI/PyHazard/blob/main/LICENSE)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/pyhazards)](https://pypi.org/project/pyhazards)
[![Issues](https://img.shields.io/github/issues/LabRAI/PyHazards)](https://github.com/LabRAI/PyHazards/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/LabRAI/PyHazards)](https://github.com/LabRAI/PyHazards/pulls)
[![Stars](https://img.shields.io/github/stars/LabRAI/PyHazards)](https://github.com/LabRAI/PyHazards)
[![GitHub forks](https://img.shields.io/github/forks/LabRAI/PyHazards)](https://github.com/LabRAI/PyHazards)

PyHazards is a Python framework for AI-powered hazard prediction and risk assessment. It provides a modular, hazard-first architecture for building, training, and deploying machine learning models to predict and analyze natural hazards (earthquake, wildfire, flood, hurricane, landslide, etc.).

## Features

- **Hazard-First Design**: Unified dataset interface for tabular, temporal, and raster data
- **Simple Models**: Ready-to-use MLP/CNN/temporal encoders with task heads (classification, regression, segmentation)
- **Trainer API**: Fit/evaluate/predict with optional mixed precision and multi-GPU (DDP) support
- **Metrics**: Built-in classification/regression/segmentation metrics
- **Extensible**: Registries for datasets, models, transforms, and pipelines

## Installation

PyHazards supports both CPU and GPU environments. Make sure you have Python installed (version >= 3.8, <3.13).

### Base Installation

Install the core package:

```bash
pip install pyhazards
```

This will install PyHazards with minimal dependencies.

Python 3.8 and PyTorch (CUDA 12.6 example)
-----------------------------------------

If you need a specific PyTorch build (e.g., CUDA 12.6), install PyTorch first, then install PyHazards:

```bash
# Example for CUDA 12.6 wheels
pip install torch --index-url https://download.pytorch.org/whl/cu126

pip install pyhazards
```

## Quick Start

Here's a simple example to get started with PyHazards using a toy tabular dataset:

```python
import torch
from pyhazards.datasets import DataBundle, DataSplit, Dataset, FeatureSpec, LabelSpec
from pyhazards.models import build_model
from pyhazards.engine import Trainer
from pyhazards.metrics import ClassificationMetrics

class ToyHazard(Dataset):
    def _load(self):
        x = torch.randn(500, 16)
        y = torch.randint(0, 2, (500,))
        splits = {
            "train": DataSplit(x[:350], y[:350]),
            "val": DataSplit(x[350:425], y[350:425]),
            "test": DataSplit(x[425:], y[425:]),
        }
        return DataBundle(
            splits=splits,
            feature_spec=FeatureSpec(input_dim=16, description="toy features"),
            label_spec=LabelSpec(num_targets=2, task_type="classification"),
        )

data = ToyHazard().load()
model = build_model(name="mlp", task="classification", in_dim=16, out_dim=2)
trainer = Trainer(model=model, metrics=[ClassificationMetrics()], mixed_precision=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

trainer.fit(data, optimizer=optimizer, loss_fn=loss_fn, max_epochs=5)
results = trainer.evaluate(data, split="test")
print(results)
```

### Using CUDA

To use CUDA for GPU acceleration, set the environment variable:

```shell
export PYHAZARDS_DEVICE=cuda:0
```

Or specify the device in your code:

```python
from pyhazards.utils import set_device

set_device("cuda:0")
```

## Documentation

Full documentation is available at: [https://labrai.github.io/PyHazards](https://labrai.github.io/PyHazards)

## Contributing

We welcome contributions! Please see our:
- [Implementation Guideline](.github/IMPLEMENTATION.md) - For implementing new models
- [Contributors Guideline](.github/CONTRIBUTING.md) - For contributing to the project

## Citation

If you use PyHazards in your research, please cite:

```bibtex
@software{pyhazards2025,
  title={PyHazards: A Python Framework for AI-Powered Hazard Prediction},
  author={Cheng, Xueqi},
  year={2025},
  url={https://github.com/LabRAI/PyHazards}
}
```

## License

[MIT License](LICENSE)

## Contact

For questions or contributions, please contact xc25@fsu.edu.
