<p align="center">
  <img src="docs/source/_static/logo.png" alt="PyHazards logo" width="170" />
</p>

<h1 align="center">PyHazards</h1>

<p align="center">
  Open-source Python library for AI-based natural hazard modeling
</p>

<p align="center">
  <a href="https://github.com/LabRAI/PyHazards">
    <img src="docs/source/_static/github.svg" alt="GitHub" width="26" />
  </a>
  <a href="https://github.com/LabRAI/PyHazards/stargazers">
    <img src="https://img.shields.io/github/stars/LabRAI/PyHazards?label=stars" alt="GitHub stars" />
  </a>
  <a href="https://github.com/LabRAI/PyHazards/network/members">
    <img src="https://img.shields.io/github/forks/LabRAI/PyHazards?label=forks" alt="GitHub forks" />
  </a>
  <a href="https://pypi.org/project/pyhazards">
    <img src="https://img.shields.io/badge/PyPI-library-3775A9?logo=pypi&logoColor=white" alt="PyPI library" />
  </a>
  <a href="https://pypi.org/project/pyhazards">
    <img src="https://img.shields.io/badge/downloads-check%20PyPI-blue" alt="PyPI downloads" />
  </a>
</p>

<p align="center">
  <a href="https://pypi.org/project/pyhazards">
    <img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fpypi.org%2Fpypi%2Fpyhazards%2Fjson&query=%24.info.version&prefix=v&label=PyPI" alt="PyPI version" />
  </a>
  <a href="https://github.com/LabRAI/PyHazards/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/LabRAI/PyHazards/ci.yml?branch=main" alt="Build status" />
  </a>
  <a href="https://github.com/LabRAI/PyHazards/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License" />
  </a>
  <a href="https://rai-lab-workspace.slack.com/archives/C0AKAJCTY4F">
    <img src="https://img.shields.io/badge/Slack-RAI%20Lab%20Channel-4A154B?logo=slack&logoColor=white" alt="Slack channel" />
  </a>
</p>

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

## Community and Activity

Join the PyHazards discussion channel on Slack:

- [RAI Lab Slack Channel](https://rai-lab-workspace.slack.com/archives/C0AKAJCTY4F)

Project star history:

[![Star History Chart](https://api.star-history.com/svg?repos=LabRAI/PyHazards&type=Date&from=2026-01-01)](https://www.star-history.com/#LabRAI/PyHazards&Date)

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
