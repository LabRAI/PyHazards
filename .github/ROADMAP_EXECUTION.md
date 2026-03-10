# PyHazards Roadmap Execution Package

This file turns `pyhazard_plan.pdf` into a repo-adjusted multi-agent execution
package that can be handed directly to parallel agents.

## Repo-adjusted roadmap

The PDF roadmap assumes a greenfield architecture. This repository already has
working dataset and model registries, model cards, generated model docs, and a
published `docs/` site. Use that existing structure as the foundation.

Apply these adjustments before execution:

1. Keep the current dataset and model registries and extend them.
   Do not replace `register_dataset(...)`, `register_model(...)`, or
   `build_model(name, task, **kwargs)`.
2. Introduce hazard taxonomy at the benchmark and config layer.
   Keep current low-level model task labels such as `classification`,
   `regression`, and `segmentation`.
3. Keep `Dataset.load()` and `DataBundle` as the runtime dataset contract.
   Treat `download`, `prepare`, and split helpers as hazard-specific adapters,
   not mandatory methods on every existing dataset class.
4. Add `pyhazards/benchmarks`, `pyhazards/configs`, and `pyhazards/reports`
   incrementally.
5. Keep public model discovery driven by `pyhazards/model_cards/*.yaml` and
   `pyhazards/model_catalog.py`.
6. Keep top-level docs and published `docs/` artifacts owned by the integrator.

## Current audited baseline

Use `docs/source/appendix_a_coverage.rst` as the checked-in audit of the
current repo against Appendix A in `pyhazard_plan.pdf`.

Key corrections from the audit:

- Wildfire is the largest gap. None of the Appendix A wildfire baselines are
  implemented yet. The current FPA-FOD family and `CNN-ASPP` stay public, but
  they are non-core variants and must not be counted as finished Appendix A
  coverage.
- Earthquake has the main model adapters (`PhaseNet`, `EQTransformer`, `GPD`,
  `EQNet`), but the SeisBench / pick-benchmark / pyCSEP / AEFA benchmark-data
  stack is still missing.
- Flood has the main model adapters (`NeuralHydrology`, `FloodCast`,
  `UrbanFloodCast`), but the Caravan / WaterBench / FloodCastBench /
  HydroBench benchmark-data stack is still missing.
- Storm has the main model adapters, but `TCBench`, `IBTrACS`, and
  `TropiCycloneNet-Dataset` are still missing. `GraphCast`, `Pangu`, and
  `FourCastNet` remain experimental wrappers and must not be counted as
  completed core baselines.
- Synthetic datasets remain smoke fixtures only. They are not evidence that the
  Appendix A benchmark adapters are finished.

## Staged delivery plan

Run the corrective roadmap as six waves:

### Wave 0: Truthful portfolio and audit enforcement

Goal:
- keep the public catalog, benchmark page, and roadmap docs aligned with the
  Appendix A audit,
- separate `core`, `variant`, and `experimental` entries,
- prevent future merges from over-counting same-paper variants or lightweight
  wrappers as finished coverage.

Exit criteria:
- generated docs show truthful status sections,
- Appendix A coverage page is in sync,
- CI checks the generated Appendix A page,
- worker agents use the audited gap list as their task queue.

### Wave 1: Shared contracts

Goal:
- land the shared task taxonomy,
- add benchmark core contracts and runner entrypoints,
- add report exporters,
- add config schema support,
- keep the existing public API intact.

Exit criteria:
- shared benchmark registry exists,
- one dummy benchmark path can run end to end,
- shared tests for task taxonomy, benchmark registry, and report export pass.

### Wave 2: First vertical slices

Goal:
- land one credible benchmark path per hazard family on top of Wave 1.

Hazard deliverables:
- Earthquake: first picking evaluator plus one credible baseline.
- Wildfire: first danger evaluator plus one credible baseline.
- Flood: first streamflow evaluator plus one credible baseline.
- Storm: first shared `tc.track_intensity` evaluator plus Hurricast.

Exit criteria:
- each hazard agent passes its owned tests,
- each hazard agent returns a registration manifest,
- each hazard family has at least one smoke config.

### Wave 3: Breadth expansion

Goal:
- add the remaining 2-3 baselines per hazard family from the PDF plan.

Exit criteria:
- all added baselines have model cards,
- smoke configs exist,
- evaluator contracts remain shared rather than duplicated inside adapters.

### Wave 4: Foundation-weather adapters

Goal:
- land GraphCast, Pangu-Weather, and FourCastNet style storm adapters only
  after the shared TC evaluator is stable.

Exit criteria:
- foundation adapters remain wrapper-style integrations,
- storm evaluator remains the single scoring entrypoint,
- dependencies are marked experimental where appropriate.

### Wave 5: Integration and release polish

Goal:
- finalize shared registry wiring,
- update generated docs and published docs,
- align CI and smoke-test scripts,
- write release-quality docs and examples.

Exit criteria:
- CI passes on `main`,
- docs build cleanly,
- generated model docs are in sync,
- the published `docs/` site reflects the merged work.

## Ownership model

Use five worker agents plus one integrator. Worker ownership must not overlap.

### Agent 1: Core Platform

Own:
- `pyhazards/tasks.py`
- `pyhazards/benchmarks/__init__.py`
- `pyhazards/benchmarks/base.py`
- `pyhazards/benchmarks/registry.py`
- `pyhazards/benchmarks/runner.py`
- `pyhazards/benchmarks/schemas.py`
- `pyhazards/configs/__init__.py`
- `pyhazards/configs/_schema.py`
- `pyhazards/reports/**`
- `pyhazards/engine/runner.py`
- `scripts/run_benchmark.py`
- `tests/test_tasks.py`
- `tests/test_benchmark_registry.py`
- `tests/test_benchmark_runner.py`
- `tests/test_report_exports.py`

Do not edit:
- `pyhazards/__init__.py`
- `pyhazards/datasets/__init__.py`
- `pyhazards/datasets/registry.py`
- `pyhazards/models/__init__.py`
- `pyhazards/models/registry.py`
- `pyhazards/models/builder.py`
- `pyhazards/model_catalog.py`
- `docs/**`
- `.github/**`

### Agent 2: Earthquake

Own:
- `pyhazards/datasets/earthquake/**`
- `pyhazards/benchmarks/earthquake.py`
- `pyhazards/configs/earthquake/**`
- `pyhazards/models/wavecastnet.py`
- `pyhazards/models/phasenet.py`
- `pyhazards/models/eqtransformer.py`
- `pyhazards/models/gpd.py`
- `pyhazards/models/eqnet.py`
- `pyhazards/model_cards/wavecastnet.yaml`
- `pyhazards/model_cards/phasenet.yaml`
- `pyhazards/model_cards/eqtransformer.yaml`
- `pyhazards/model_cards/gpd.yaml`
- `pyhazards/model_cards/eqnet.yaml`
- `tests/test_earthquake_*.py`

Do not edit:
- shared registries and package `__init__` files
- `pyhazards/model_catalog.py`
- `docs/**`
- `.github/**`

### Agent 3: Wildfire

Own:
- `pyhazards/benchmarks/wildfire.py`
- `pyhazards/configs/wildfire/**`
- `pyhazards/models/wildfire_*.py`
- `pyhazards/models/cnn_aspp.py`
- `pyhazards/model_cards/wildfire_*.yaml`
- `pyhazards/datasets/firms/**`
- `pyhazards/datasets/mtbs/**`
- `pyhazards/datasets/landfire/**`
- `pyhazards/datasets/wfigs/**`
- `pyhazards/datasets/fpa_fod.py`
- `pyhazards/datasets/fpa_fod_tabular/**`
- `pyhazards/datasets/fpa_fod_weekly/**`
- `tests/test_wildfire_*.py`
- `tests/test_fpa_fod_*.py`

Do not edit:
- shared registries and package `__init__` files
- `pyhazards/model_catalog.py`
- `docs/**`
- `.github/**`

### Agent 4: Flood

Own:
- `pyhazards/datasets/flood/**`
- `pyhazards/datasets/noaa_flood/**`
- `pyhazards/benchmarks/flood.py`
- `pyhazards/configs/flood/**`
- `pyhazards/models/hydrographnet.py`
- `pyhazards/models/neuralhydrology_*.py`
- `pyhazards/models/floodcast.py`
- `pyhazards/models/urbanfloodcast.py`
- `pyhazards/model_cards/hydrographnet.yaml`
- `pyhazards/model_cards/neuralhydrology_*.yaml`
- `pyhazards/model_cards/floodcast.yaml`
- `pyhazards/model_cards/urbanfloodcast.yaml`
- `pyhazards/data/load_hydrograph_data.py`
- `tests/test_flood_*.py`

Do not edit:
- shared registries and package `__init__` files
- `pyhazards/model_catalog.py`
- `docs/**`
- `.github/**`

### Agent 5: Storm

Own:
- `pyhazards/datasets/tc/**`
- `pyhazards/benchmarks/tc.py`
- `pyhazards/configs/tc/**`
- `pyhazards/models/hurricast.py`
- `pyhazards/models/tropicalcyclone_mlp.py`
- `pyhazards/models/tropicyclonenet.py`
- `pyhazards/models/saf_net.py`
- `pyhazards/models/tcif_fusion.py`
- `pyhazards/models/graphcast_tc.py`
- `pyhazards/models/pangu_tc.py`
- `pyhazards/models/fourcastnet_tc.py`
- `pyhazards/model_cards/hurricast.yaml`
- `pyhazards/model_cards/tropicalcyclone_mlp.yaml`
- `pyhazards/model_cards/tropicyclonenet.yaml`
- `pyhazards/model_cards/saf_net.yaml`
- `pyhazards/model_cards/tcif_fusion.yaml`
- `pyhazards/model_cards/graphcast_tc.yaml`
- `pyhazards/model_cards/pangu_tc.yaml`
- `pyhazards/model_cards/fourcastnet_tc.yaml`
- `tests/test_tc_*.py`

Do not edit:
- shared registries and package `__init__` files
- `pyhazards/model_catalog.py`
- `docs/**`
- `.github/**`

### Integrator

Own exclusively:
- `pyhazards/__init__.py`
- `pyhazards/datasets/__init__.py`
- `pyhazards/datasets/registry.py`
- `pyhazards/models/__init__.py`
- `pyhazards/models/registry.py`
- `pyhazards/models/builder.py`
- `pyhazards/model_catalog.py`
- `scripts/render_model_docs.py`
- `scripts/smoke_test_models.py`
- `scripts/verify_table_entries.py`
- `.github/workflows/**`
- `.github/PULL_REQUEST_TEMPLATE.md`
- `.github/IMPLEMENTATION.md`
- `docs/source/*.rst`
- `docs/source/api/**`
- `docs/source/modules/**`
- `docs/**`

Workers must not edit those files. They must instead return:
- registration manifests,
- model-card manifests,
- smoke-config names,
- doc notes for the integrator.

## Worker prompts

### Agent 1 prompt

```text
You are Agent 1 for the PyHazards roadmap.

Mission:
Implement the shared platform layer required by the roadmap without breaking the
current dataset/model registry API.

You own:
- pyhazards/tasks.py
- pyhazards/benchmarks/{__init__.py,base.py,registry.py,runner.py,schemas.py}
- pyhazards/configs/{__init__.py,_schema.py}
- pyhazards/reports/**
- pyhazards/engine/runner.py
- scripts/run_benchmark.py
- tests/test_tasks.py
- tests/test_benchmark_registry.py
- tests/test_benchmark_runner.py
- tests/test_report_exports.py

Do not edit:
- pyhazards/__init__.py
- pyhazards/models/__init__.py
- pyhazards/datasets/__init__.py
- pyhazards/models/registry.py
- pyhazards/datasets/registry.py
- pyhazards/model_catalog.py
- docs/**
- .github/**

Implementation requirements:
- Keep the current build_model(name, task, **kwargs) contract intact.
- Add benchmark-level hazard taxonomy separately from low-level model task
  values.
- Define benchmark contracts with evaluate(), aggregate_metrics(), and
  export_report().
- Add shared report exporters and a benchmark runner entrypoint.
- Add shared config schema support without requiring a whole-repo dataclass
  rewrite.
- Make the core layer stable enough that hazard agents can implement evaluators
  without inventing their own contracts.

Required outputs:
- code in owned files
- tests in owned files
- a short contract note for hazard agents
- a summary of any interface assumptions the integrator must expose publicly

Validation:
- python -m pytest tests/test_tasks.py tests/test_benchmark_registry.py tests/test_benchmark_runner.py tests/test_report_exports.py
- python scripts/run_benchmark.py --help

Escalate if:
- you need changes in shared current registry files or top-level docs
- the current engine lifecycle blocks a benchmark runner without changing shared exports
```

### Agent 2 prompt

```text
You are Agent 2 for the PyHazards roadmap.

Mission:
Own the earthquake workstream across the staged roadmap: first the vertical
slice, then breadth expansion.

You own:
- pyhazards/datasets/earthquake/**
- pyhazards/benchmarks/earthquake.py
- pyhazards/configs/earthquake/**
- pyhazards/models/wavecastnet.py
- pyhazards/models/phasenet.py
- pyhazards/models/eqtransformer.py
- pyhazards/models/gpd.py
- pyhazards/models/eqnet.py
- pyhazards/model_cards/wavecastnet.yaml
- pyhazards/model_cards/phasenet.yaml
- pyhazards/model_cards/eqtransformer.yaml
- pyhazards/model_cards/gpd.yaml
- pyhazards/model_cards/eqnet.yaml
- tests/test_earthquake_*.py

Do not edit:
- shared registries
- pyhazards/model_catalog.py
- docs/**
- .github/**

Implementation requirements:
- Treat wavecastnet as an existing starting asset, not a new baseline.
- Land the first working slice around earthquake picking first.
- Standardize waveform tensor and metadata assumptions inside your owned module set.
- Implement one evaluator path for picking before breadth expansion.
- Add later baselines only after the first evaluator and one baseline pass
  cleanly.
- Return a registration manifest for every dataset and model you add or change.

Required outputs:
- owned code and tests
- model-card files for public baselines
- config files for smoke and fuller runs
- a registration manifest for the integrator

Validation:
- python -m pytest tests/test_earthquake_*.py

Escalate if:
- a required change touches shared registries, top-level docs, or generated docs
- the benchmark contract from Agent 1 is insufficient for picking or forecasting reports
```

### Agent 3 prompt

```text
You are Agent 3 for the PyHazards roadmap.

Mission:
Own the wildfire workstream across the staged roadmap: danger first, then
spread, then breadth.

You own:
- pyhazards/benchmarks/wildfire.py
- pyhazards/configs/wildfire/**
- pyhazards/models/wildfire_*.py
- pyhazards/models/cnn_aspp.py
- pyhazards/model_cards/wildfire_*.yaml
- pyhazards/datasets/firms/**
- pyhazards/datasets/mtbs/**
- pyhazards/datasets/landfire/**
- pyhazards/datasets/wfigs/**
- pyhazards/datasets/fpa_fod.py
- pyhazards/datasets/fpa_fod_tabular/**
- pyhazards/datasets/fpa_fod_weekly/**
- tests/test_wildfire_*.py
- tests/test_fpa_fod_*.py

Do not edit:
- shared registries
- pyhazards/model_catalog.py
- docs/**
- .github/**

Implementation requirements:
- Treat existing wildfire assets as the seed state.
- Separate wildfire.danger and wildfire.spread at the benchmark and config layer.
- Land one danger baseline before expanding the spread stack.
- Standardize raster/tile output conventions for spread evaluators.
- Keep physics-style simulator integration as an external adapter, not a deep
  vendor import.
- Return a registration manifest for every dataset and model you add or change.

Required outputs:
- owned code and tests
- smoke and full configs under pyhazards/configs/wildfire/
- model cards for public baselines
- a registration manifest for the integrator

Validation:
- python -m pytest tests/test_fpa_fod_datasets.py tests/test_fpa_fod_models.py tests/test_fpa_fod_trainer_smoke.py tests/test_wildfire_*.py

Escalate if:
- shared registry wiring is required
- generic dataset helpers outside your ownership must change
```

### Agent 4 prompt

```text
You are Agent 4 for the PyHazards roadmap.

Mission:
Own the flood workstream across the staged roadmap: streamflow first, then
inundation, then breadth.

You own:
- pyhazards/datasets/flood/**
- pyhazards/datasets/noaa_flood/**
- pyhazards/benchmarks/flood.py
- pyhazards/configs/flood/**
- pyhazards/models/hydrographnet.py
- pyhazards/models/neuralhydrology_*.py
- pyhazards/models/floodcast.py
- pyhazards/models/urbanfloodcast.py
- pyhazards/model_cards/hydrographnet.yaml
- pyhazards/model_cards/neuralhydrology_*.yaml
- pyhazards/model_cards/floodcast.yaml
- pyhazards/model_cards/urbanfloodcast.yaml
- pyhazards/data/load_hydrograph_data.py
- tests/test_flood_*.py

Do not edit:
- shared registries
- pyhazards/model_catalog.py
- docs/**
- .github/**

Implementation requirements:
- Treat hydrographnet as an existing starting asset.
- Expose flood.streamflow and flood.inundation as separate benchmark and config
  tracks.
- Land streamflow evaluation first.
- Use adapter-style integration for NeuralHydrology-family baselines.
- Add inundation evaluation only after streamflow is stable.
- Return a registration manifest for every dataset and model you add or change.

Required outputs:
- owned code and tests
- smoke and full configs under pyhazards/configs/flood/
- model cards for public baselines
- a registration manifest for the integrator

Validation:
- python -m pytest tests/test_flood_*.py

Escalate if:
- shared registry wiring is required
- generic shared datasets outside your ownership must change
```

### Agent 5 prompt

```text
You are Agent 5 for the PyHazards roadmap.

Mission:
Own the combined hurricane and tropical-cyclone workstream across the staged
roadmap, including the late foundation-adapter phase.

You own:
- pyhazards/datasets/tc/**
- pyhazards/benchmarks/tc.py
- pyhazards/configs/tc/**
- pyhazards/models/hurricast.py
- pyhazards/models/tropicalcyclone_mlp.py
- pyhazards/models/tropicyclonenet.py
- pyhazards/models/saf_net.py
- pyhazards/models/tcif_fusion.py
- pyhazards/models/graphcast_tc.py
- pyhazards/models/pangu_tc.py
- pyhazards/models/fourcastnet_tc.py
- pyhazards/model_cards/hurricast.yaml
- pyhazards/model_cards/tropicalcyclone_mlp.yaml
- pyhazards/model_cards/tropicyclonenet.yaml
- pyhazards/model_cards/saf_net.yaml
- pyhazards/model_cards/tcif_fusion.yaml
- pyhazards/model_cards/graphcast_tc.yaml
- pyhazards/model_cards/pangu_tc.yaml
- pyhazards/model_cards/fourcastnet_tc.yaml
- tests/test_tc_*.py

Do not edit:
- shared registries
- pyhazards/model_catalog.py
- docs/**
- .github/**

Implementation requirements:
- Implement one shared storm-centric evaluator for track and intensity first.
- Treat hurricane as basin presets layered on top of the shared TC module.
- Land Hurricast before the broader TC stack.
- Delay GraphCast, Pangu, and FourCastNet adapters until the shared TC evaluator
  is stable.
- Keep foundation-model integrations as external-field adapters plus extraction
  and evaluation wrappers.
- Return a registration manifest for every dataset and model you add or change.

Required outputs:
- owned code and tests
- smoke and full configs under pyhazards/configs/tc/
- model cards for public baselines
- a registration manifest for the integrator

Validation:
- python -m pytest tests/test_tc_*.py

Escalate if:
- shared registry wiring is required
- the benchmark contract from Agent 1 is insufficient for track and intensity reports
```

## Integrator prompt

```text
You are the integrator agent for the PyHazards roadmap.

Mission:
Merge the staged outputs from Agents 1-5 into the existing repo without
breaking the current public API, docs generation flow, or CI.

You exclusively own:
- pyhazards/__init__.py
- pyhazards/datasets/__init__.py
- pyhazards/datasets/registry.py
- pyhazards/models/__init__.py
- pyhazards/models/registry.py
- pyhazards/models/builder.py
- pyhazards/model_catalog.py
- scripts/render_model_docs.py
- scripts/smoke_test_models.py
- scripts/verify_table_entries.py
- .github/workflows/**
- .github/PULL_REQUEST_TEMPLATE.md
- .github/IMPLEMENTATION.md
- docs/source/*.rst
- docs/source/api/**
- docs/source/modules/**
- docs/**

Responsibilities:
- merge Agent 1 first and expose shared contracts publicly only where needed
- wire dataset and model registrations from worker manifests
- keep existing build_model(name, task, **kwargs) behavior intact
- update model catalog generation and smoke scripts for new public baselines
- update top-level docs and implementation guidance
- regenerate model docs and published docs
- run final validation
- resolve conflicts without letting worker agents edit shared choke points directly

Final validation:
- python -c "import pyhazards; print(pyhazards.__version__)"
- python -m pytest tests
- python scripts/render_model_docs.py
- python scripts/render_model_docs.py --check
- python scripts/verify_table_entries.py
- python scripts/smoke_test_models.py
- cd docs && sphinx-build -b html source build/html

Do not:
- redesign worker-owned implementations unless integration requires it
- drop model cards or smoke specs from worker branches
- merge foundation adapters before the TC evaluator is stable
```

## Dependency and merge order

1. Phase A:
   - Agent 1 only.
   - Integrator merges Agent 1 after its tests pass.
2. Phase B:
   - Agents 2-5 branch from the integrated Phase A commit and work in parallel.
   - Integrator merges in this order:
     - Earthquake
     - Wildfire
     - Flood
     - Storm
3. Phase C:
   - Hazard agents continue breadth expansion on fresh branches from updated
     `main`.
   - Integrator merges in the same order:
     - Earthquake
     - Wildfire
     - Flood
     - Storm
4. Phase D:
   - Agent 5 alone handles foundation-weather adapters.
   - Integrator merges the storm foundation branch last.
5. Phase E:
   - Integrator finalizes shared docs, CI, packaging, generated docs, and
     release polish.

## Validation by role

### Worker validation

- Agent 1:
  - `python -m pytest tests/test_tasks.py tests/test_benchmark_registry.py tests/test_benchmark_runner.py tests/test_report_exports.py`
  - `python scripts/run_benchmark.py --help`
- Agent 2:
  - `python -m pytest tests/test_earthquake_*.py`
- Agent 3:
  - `python -m pytest tests/test_fpa_fod_datasets.py tests/test_fpa_fod_models.py tests/test_fpa_fod_trainer_smoke.py tests/test_wildfire_*.py`
- Agent 4:
  - `python -m pytest tests/test_flood_*.py`
- Agent 5:
  - `python -m pytest tests/test_tc_*.py`

### Integrator validation

- `python -c "import pyhazards; print(pyhazards.__version__)"`
- `python -m pytest tests`
- `python scripts/render_model_docs.py`
- `python scripts/render_model_docs.py --check`
- `python scripts/verify_table_entries.py`
- `python scripts/smoke_test_models.py`
- `cd docs && sphinx-build -b html source build/html`

## Likely conflict points

### Shared registries and package exports

Risk:
- every hazard wants to touch `__init__` files and registries.

Avoidance:
- only the integrator edits those files,
- workers return registration manifests instead.

### Model catalog and generated docs

Risk:
- public model additions naturally collide in generated docs and catalog logic.

Avoidance:
- workers edit only model cards and their own model files,
- integrator runs `render_model_docs.py` and rebuilds the published site.

### Top-level docs and implementation guidance

Risk:
- every hazard can generate docs requests that overlap the same pages.

Avoidance:
- workers provide doc notes only,
- integrator owns `docs/source/*.rst`, `docs/source/api/**`,
  `docs/source/modules/**`, and `docs/**`.

### Shared dataset helpers

Risk:
- hazard agents may attempt to patch common helpers when adding loaders.

Avoidance:
- add new hazard-specific loaders under owned directories first,
- escalate before touching shared generic helpers.

### Foundation-weather adapters

Risk:
- GraphCast and Pangu style integrations can destabilize the whole roadmap if
  merged too early.

Avoidance:
- hard phase gate,
- no foundation adapter merges before the shared TC evaluator is stable on
  `main`.

## How to use this package

1. Decide the current wave.
2. Start the agents allowed in that wave.
3. Give each worker only the prompt from this file plus the current base commit.
4. Require a manifest of registrations, model cards, configs, tests, and open
   issues from every worker.
5. Hand the manifests and branches to the integrator.
6. Require the integrator to run the full validation set before merge or push.
