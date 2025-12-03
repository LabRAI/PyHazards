.. raw:: html

   <div style="margin-top: 50px; text-align: center;">
     <img src="_static/logo.pdf" alt="PyHazard Icon" style="width: 600px; height: auto;">
   </div>

.. image:: https://img.shields.io/pypi/v/PyHazard
   :target: https://pypi.org/project/PyHazard
   :alt: PyPI Version

.. image:: https://img.shields.io/github/actions/workflow/status/LabRAI/PyHazard/docs.yml
   :target: https://github.com/LabRAI/PyHazard/actions
   :alt: Build Status

.. image:: https://img.shields.io/github/license/LabRAI/PyHazard.svg
   :target: https://github.com/LabRAI/PyHazard/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/pypi/dm/pyhazard
   :target: https://github.com/LabRAI/PyHazard
   :alt: PyPI Downloads

.. image:: https://img.shields.io/github/issues/LabRAI/PyHazard
   :target: https://github.com/LabRAI/PyHazard
   :alt: Issues

.. image:: https://img.shields.io/github/issues-pr/LabRAI/PyHazard
   :target: https://github.com/LabRAI/PyHazard
   :alt: Pull Requests

.. image:: https://img.shields.io/github/stars/LabRAI/PyHazard
   :target: https://github.com/LabRAI/PyHazard
   :alt: Stars

.. image:: https://img.shields.io/github/forks/LabRAI/PyHazard
   :target: https://github.com/LabRAI/PyHazard
   :alt: GitHub forks

.. image:: _static/github.svg
   :target: https://github.com/LabRAI/PyHazard
   :alt: GitHub

----

**PyHazard** is a comprehensive Python framework for AI-powered hazard prediction and risk assessment. Built on PyTorch, PyTorch Geometric, and DGL, the library provides a modular and extensible architecture for building, training, and deploying machine learning models to predict and analyze natural hazards and environmental risks.

**PyHazard is designed for:**

- **Modular Architecture**: Easy-to-extend framework for implementing custom hazard prediction models
- **Graph Neural Networks**: Built-in support for graph-based data representation and GNN models
- **Flexible Datasets**: Unified dataset interface supporting both DGL and PyTorch Geometric
- **GPU Acceleration**: Full CUDA support for training and inference
- **Extensible Models**: Base classes for implementing custom prediction models
- **Production Ready**: Type hints, proper error handling, and comprehensive testing

**Quick Start Example:**

Basic Usage Example:

.. code-block:: python

    import pyhazard
    from pyhazard.datasets import Cora
    from pyhazard.models.nn import GCN

    # Load a dataset
    dataset = Cora(api_type='pyg')

    # Initialize a model
    model = GCN(
        input_dim=dataset.num_features,
        hidden_dim=64,
        output_dim=dataset.num_classes
    )

    # Your training and prediction code here
    # ...

Core Components
---------------

**Datasets**
   PyHazard provides a unified dataset interface that supports both DGL and PyTorch Geometric formats.
   Built-in datasets include Cora, CiteSeer, PubMed, and more.

**Models**
   Extensible model architecture with built-in neural network backbones including GCN, GAT, and more.
   Easy to implement custom models by extending base classes.

**Utilities**
   Helper functions for device management, metrics calculation, and data conversion between DGL and PyTorch Geometric.

How to Cite
-----------

If you use PyHazard in your research, please cite:

.. code-block:: bibtex

   @software{pyhazard2025,
     title={PyHazard: A Python Framework for AI-Powered Hazard Prediction},
     author={Cheng, Xueqi},
     year={2025},
     url={https://github.com/LabRAI/PyHazard}
   }


.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   installation
   quick_start

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   pyhazard_datasets
   pyhazard_utils

.. toctree::
   :maxdepth: 2
   :caption: Additional Information
   :hidden:

   implementation
   cite
   team


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
