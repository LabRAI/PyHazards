pyhazards.models package
========================

Model Category
--------------

Wildfire
~~~~~~~~

.. list-table::
   :widths: 30 70
   :header-rows: 1
   :class: dataset-list

   * - Model
     - Description
   * - :doc:`CNN-ASPP </modules/models_wildfire_aspp>`
     - Explainable CNN-ASPP model for next-day wildfire spread prediction.
   * - :doc:`DNN-LSTM-AutoEncoder </modules/models_wildfire_fpa>`
     - Two-stage FPA-FOD framework with a DNN stage for cause/size prediction and an LSTM + autoencoder stage for weekly wildfire forecasting.

Flood
~~~~~

.. list-table::
   :widths: 30 70
   :header-rows: 1
   :class: dataset-list

   * - Model
     - Description
   * - :doc:`HydroGraphNet </modules/models_hydrographnet>`
     - Physics-informed graph neural network for flood forecasting on irregular meshes.

Submodules
----------

pyhazards.models.backbones module
----------------------------------

.. automodule:: pyhazards.models.backbones
   :members:
   :undoc-members:
   :show-inheritance:

pyhazards.models.heads module
------------------------------

.. automodule:: pyhazards.models.heads
   :members:
   :undoc-members:
   :show-inheritance:

pyhazards.models.builder module
--------------------------------

.. automodule:: pyhazards.models.builder
   :members:
   :undoc-members:
   :show-inheritance:

pyhazards.models.registry module
---------------------------------

.. automodule:: pyhazards.models.registry
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: pyhazards.models
   :members:
   :undoc-members:
   :show-inheritance:
