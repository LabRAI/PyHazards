# PyHazards

[![PyPI - Version](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fpypi.org%2Fpypi%2Fpyhazards%2Fjson&query=%24.info.version&prefix=v&label=PyPI)](https://pypi.org/project/pyhazards)
[![Build Status](https://img.shields.io/github/actions/workflow/status/LabRAI/PyHazards/ci.yml?branch=main)](https://github.com/LabRAI/PyHazards/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/LabRAI/PyHazards/blob/main/LICENSE)

PyHazards is an open-source Python library for AI-based natural hazard modeling. It provides unified interfaces for datasets, models, benchmarks, training pipelines, and evaluation across wildfire, earthquake, flood, and tropical-cyclone workflows.

It is designed for researchers, practitioners, and contributors who need a consistent way to inspect hazard data, run benchmark-aligned experiments, and extend the library with new datasets or model implementations.

## What PyHazards Provides

- **Dataset catalog and inspection workflows** for public hazard datasets, benchmark adapters, and shared forcing sources.
- **Registry-driven model implementations** spanning wildfire, earthquake, flood, and tropical-cyclone tasks.
- **Shared benchmark, config, and report layers** for reproducible smoke tests and evaluation workflows.
- **Reusable training and inference utilities** through a common engine API.

## Hazard Coverage

- **Wildfire**: incident records, active-fire detections, fuels and burn products, danger forecasting, weekly forecasting, and spread baselines.
- **Earthquake**: waveform-picking baselines, forecasting adapters, and shared picking/forecasting benchmark paths.
- **Flood**: streamflow and inundation models with benchmark-aligned adapter datasets and evaluation coverage.
- **Tropical Cyclone**: track-and-intensity forecasting baselines plus linked benchmark ecosystems.

## Installation

Install PyHazards from PyPI. If you need GPU execution, install a compatible PyTorch build first.

```bash
pip install pyhazards
```

Optional device selection:

```bash
export PYHAZARDS_DEVICE=cuda:0
```

## Quick Start

Inspect one dataset source:

```bash
python -m pyhazards.datasets.era5.inspection --path pyhazards/data/era5_subset --max-vars 10
```

Build one registered model:

```python
from pyhazards.models import build_model

model = build_model(
    name="hydrographnet",
    task="regression",
    node_in_dim=2,
    edge_in_dim=3,
    out_dim=1,
)
print(type(model).__name__)
```

Run the GPU smoke path when CUDA is available:

```bash
python test.py
```

## Documentation

Full documentation: [https://labrai.github.io/PyHazards](https://labrai.github.io/PyHazards)

Recommended path:

1. [Installation](https://labrai.github.io/PyHazards/installation.html)
2. [Quick Start](https://labrai.github.io/PyHazards/quick_start.html)
3. [Datasets](https://labrai.github.io/PyHazards/pyhazards_datasets.html)
4. [Models](https://labrai.github.io/PyHazards/pyhazards_models.html)
5. [Benchmarks](https://labrai.github.io/PyHazards/pyhazards_benchmarks.html)

## Contributing

PyHazards uses generated dataset, model, and benchmark catalogs. If you are extending the library:

- use the public contributor guide in [docs/source/implementation.rst](docs/source/implementation.rst)
- use the maintainer workflow notes in [.github/IMPLEMENTATION.md](.github/IMPLEMENTATION.md)
- follow the repository contribution process in [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md)

## Citation

If you use PyHazards in your research, please cite:

```bibtex
@misc{pyhazards2025,
  title        = {PyHazards: An Open-Source Library for AI-Powered Hazard Prediction},
  author       = {Cheng et al.},
  year         = {2025},
  howpublished = {\url{https://github.com/LabRAI/PyHazards}},
  note         = {GitHub repository}
}
```

## License

[MIT License](LICENSE)
