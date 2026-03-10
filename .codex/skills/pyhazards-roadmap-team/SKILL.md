---
name: pyhazards-roadmap-team
description: Use when asked to turn `pyhazard_plan.pdf` into a staged multi-agent execution plan for this repository, or to run the PyHazards hazard-expansion roadmap with fixed worker ownership, copy-paste agent prompts, phase gates, and an integrator workflow instead of re-deriving the team split.
---

# PyHazards Roadmap Team

## Overview

Use the checked-in execution package in `.github/ROADMAP_EXECUTION.md` as the
source of truth for the long-range PyHazards roadmap, and use
`docs/source/appendix_a_coverage.rst` as the audited coverage baseline for
what is still missing.

Do not re-split ownership or redesign the phase order unless the user
explicitly asks to change the roadmap.

## Fast Start

Open these files first:

- `.github/ROADMAP_EXECUTION.md`
- `pyhazard_plan.pdf`
- `docs/source/appendix_a_coverage.rst`
- `.github/IMPLEMENTATION.md`
- `pyhazards/models/__init__.py`
- `pyhazards/datasets/__init__.py`
- `pyhazards/model_catalog.py`
- `scripts/render_model_docs.py`
- `scripts/smoke_test_models.py`
- `scripts/verify_table_entries.py`

## Workflow

1. Identify the wave the user wants to execute.
   If unspecified, start from the earliest incomplete wave in
   `.github/ROADMAP_EXECUTION.md`, using the Appendix A coverage page to avoid
   counting variants or experimental wrappers as finished baseline work.

2. Keep file ownership fixed.
   Worker agents must not edit shared choke-point files. Only the integrator
   owns registries, top-level docs, generated docs, and workflows.

3. Hand out the exact agent prompt from `.github/ROADMAP_EXECUTION.md`.
   Do not paraphrase unless the user asks for a smaller or larger team.

4. Require manifests from worker agents.
   Every worker should return:
   - changed files
   - registration changes
   - model-card changes
   - config names
   - tests run
   - open issues

5. Use the integrator workflow for merge and validation.
   The integrator must run the full validation set listed in the execution
   package before merge or push.

## Guardrails

- Keep the current `build_model(name, task, **kwargs)` contract intact.
- Keep public model discovery driven by model cards and generated docs.
- Keep `catalog_status` truthful so same-paper variants and experimental
  wrappers do not inflate Appendix A coverage.
- Keep `docs/` published artifacts integrator-owned.
- Delay storm foundation adapters until the shared TC evaluator is stable.
- If a worker needs a shared-file change, escalate to the integrator instead of
  patching around ownership.
