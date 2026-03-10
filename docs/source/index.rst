.. title:: PyHazards

.. image:: _static/logo.png
   :alt: PyHazards Icon
   :width: 220px
   :align: center
   :class: landing-hero-logo

.. raw:: html

   <h2 class="landing-hero-title">PyHazards</h2>

.. raw:: html

   <div class="home-cta-row">
     <a class="home-cta-button home-cta-secondary" href="quick_start.html">Quick Start</a>
     <a class="home-cta-button home-cta-secondary" href="pyhazards_models.html">Browse Models</a>
     <a class="home-cta-button home-cta-secondary" href="pyhazards_benchmarks.html">Browse Benchmarks</a>
   </div>

Overview
--------

PyHazards brings together public dataset catalogs, registry-based models,
benchmark families, experiment configs, and shared training or reporting
workflows across wildfire, earthquake, flood, and tropical cyclone tasks.

It is designed for researchers and practitioners who need one coherent library
for reproducing baselines, comparing methods, and extending hazard-ML
workflows without rebuilding the software stack for each hazard family.

.. grid:: 1 2 4 4
   :gutter: 2
   :class-container: catalog-grid home-kicker-grid home-hero-stats

   .. grid-item-card:: Hazard Families
      :class-card: catalog-stat-card

      .. container:: catalog-stat-value

         4

      .. container:: catalog-stat-note

         Wildfire, earthquake, flood, and tropical cyclone workflows under one library.

   .. grid-item-card:: Public Datasets
      :class-card: catalog-stat-card

      .. container:: catalog-stat-value

         20

      .. container:: catalog-stat-note

         Curated dataset pages covering forcing sources and hazard-specific benchmark adapters.

   .. grid-item-card:: Implemented Models
      :class-card: catalog-stat-card

      .. container:: catalog-stat-value

         24

      .. container:: catalog-stat-note

         Public implemented baselines and variants surfaced through the model catalog.

   .. grid-item-card:: Benchmark Families
      :class-card: catalog-stat-card

      .. container:: catalog-stat-value

         4

      .. container:: catalog-stat-note

         Shared evaluator families with linked ecosystems, smoke configs, and reports.

Start Here
----------

.. container:: home-section-note

   Use one of these four paths to move from overview to action quickly.

.. grid:: 1 1 2 4
   :gutter: 2
   :class-container: catalog-recommend-grid home-link-grid

   .. grid-item-card:: Quick Start
      :class-card: catalog-detail-card

      Run the first benchmark-aware workflow and verify the package.

      **Open:** :doc:`Quick Start <quick_start>`

   .. grid-item-card:: Browse Datasets
      :class-card: catalog-detail-card

      Explore forcing sources, benchmark adapters, and inspection entrypoints.

      **Open:** :doc:`Datasets <pyhazards_datasets>`

   .. grid-item-card:: Browse Models
      :class-card: catalog-detail-card

      Compare implemented baselines, variants, and benchmark-linked model detail pages.

      **Open:** :doc:`Models <pyhazards_models>`

   .. grid-item-card:: Browse Benchmarks
      :class-card: catalog-detail-card

      Compare hazard benchmark families, ecosystem mappings, and smoke coverage.

      **Open:** :doc:`Benchmarks <pyhazards_benchmarks>`

Why PyHazards
-------------

.. grid:: 1 1 2 4
   :gutter: 2
   :class-container: catalog-grid home-pillar-grid

   .. grid-item-card:: Unified Datasets
      :class-card: catalog-detail-card

      Public datasets, forcing sources, and inspection surfaces are documented through one hazard-first catalog.

   .. grid-item-card:: Benchmark-aligned Evaluation
      :class-card: catalog-detail-card

      Shared benchmark families, smoke configs, and report exports make model comparisons more reproducible.

   .. grid-item-card:: Registry-based Models
      :class-card: catalog-detail-card

      Baselines and adapters are exposed through a consistent build surface instead of one-off scripts.

   .. grid-item-card:: Shared Training and Inference
      :class-card: catalog-detail-card

      One engine layer supports training, evaluation, prediction, and benchmark execution across hazard tasks.

Hazard Coverage
---------------

.. container:: home-section-note

   PyHazards spans four hazard families with public datasets, models, and benchmark pages designed to work together.

.. grid:: 1 1 2 4
   :gutter: 2
   :class-container: catalog-recommend-grid home-hazard-grid

   .. grid-item-card:: Wildfire
      :class-card: catalog-detail-card

      Danger forecasting, weekly forecasting, spread baselines, fuels, burn products, and active-fire sources.

      **Explore:** :doc:`Datasets <pyhazards_datasets>` | :doc:`Models <pyhazards_models>`

   .. grid-item-card:: Earthquake
      :class-card: catalog-detail-card

      Waveform picking, dense-grid forecasting adapters, and linked benchmark ecosystems for phase-picking workflows.

      **Explore:** :doc:`Models <pyhazards_models>` | :doc:`Benchmarks <pyhazards_benchmarks>`

   .. grid-item-card:: Flood
      :class-card: catalog-detail-card

      Streamflow and inundation baselines with benchmark-backed datasets, configs, and evaluation coverage.

      **Explore:** :doc:`Datasets <pyhazards_datasets>` | :doc:`Benchmarks <pyhazards_benchmarks>`

   .. grid-item-card:: Tropical Cyclone
      :class-card: catalog-detail-card

      Track-and-intensity forecasting baselines plus shared benchmark ecosystems and experimental weather-model adapters.

      **Explore:** :doc:`Models <pyhazards_models>` | :doc:`Benchmarks <pyhazards_benchmarks>`

Featured Example
----------------

.. container:: home-section-note

   Run a benchmark-aligned smoke configuration with one command, then move into the full Quick Start for model building and training workflows.

.. code-block:: bash

   python scripts/run_benchmark.py --config pyhazards/configs/flood/hydrographnet_smoke.yaml

.. container:: catalog-link-row

   **Next step:** :doc:`Quick Start <quick_start>` for the first full workflow, or :doc:`Models <pyhazards_models>` to browse benchmark-linked baselines.

Explore the Docs
----------------

.. grid:: 1 1 2 3
   :gutter: 2
   :class-container: catalog-recommend-grid home-link-grid

   .. grid-item-card:: Installation
      :class-card: catalog-detail-card

      Set up PyHazards from PyPI or source and verify the environment.

      **Open:** :doc:`installation`

   .. grid-item-card:: Quick Start
      :class-card: catalog-detail-card

      Run the shortest end-to-end workflow in the library.

      **Open:** :doc:`quick_start`

   .. grid-item-card:: Datasets
      :class-card: catalog-detail-card

      Browse hazard-grouped dataset cards, detail pages, and inspection entrypoints.

      **Open:** :doc:`pyhazards_datasets`

   .. grid-item-card:: Models
      :class-card: catalog-detail-card

      Compare implemented models, variants, and benchmark-linked detail pages.

      **Open:** :doc:`pyhazards_models`

   .. grid-item-card:: Benchmarks
      :class-card: catalog-detail-card

      Review benchmark families, ecosystem mappings, and smoke-config coverage.

      **Open:** :doc:`pyhazards_benchmarks`

   .. grid-item-card:: Reports and Configs
      :class-card: catalog-detail-card

      Load reproducible experiment YAML files and export benchmark summaries.

      **Open:** :doc:`pyhazards_configs` | :doc:`pyhazards_reports`

For Contributors
----------------

PyHazards is registry-driven and uses dataset cards, model cards, and benchmark
cards to generate the public catalogs. If you plan to extend the library, use
:doc:`implementation` for the contributor workflow and :doc:`appendix_a_coverage`
for the audited gap list behind the current roadmap work.

Citation
--------

If you use PyHazards in your research, please cite:

.. code-block:: bibtex

   @misc{pyhazards2025,
     title        = {PyHazards: An Open-Source Library for AI-Powered Hazard Prediction},
     author       = {Cheng et al.},
     year         = {2025},
     howpublished = {\url{https://github.com/LabRAI/PyHazards}},
     note         = {GitHub repository}
   }

Community
---------

Use the `RAI Lab Slack channel <https://rai-lab-workspace.slack.com/archives/C0AKAJCTY4F>`_
for project discussion and coordination.


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

   pyhazards_datasets
   pyhazards_models
   pyhazards_benchmarks
   pyhazards_configs
   pyhazards_reports
   pyhazards_engine
   pyhazards_metrics
   pyhazards_utils
   interactive_map

.. toctree::
   :maxdepth: 2
   :caption: Additional Information
   :hidden:

   implementation
   appendix_a_coverage
   cite
   references
   team
