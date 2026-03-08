---
name: pyhazards-pr-maintainer
description: Review and process pull requests for the PyHazard/PyHazards repository, especially model contribution PRs. Use when asked to inspect a new PR, decide whether the implementation matches the described model or paper, run the catalog smoke checks, update generated model tables/docs, merge a good PR, or draft an actionable blocker comment for a weak PR without rediscovering the repo workflow.
---

# PyHazards PR Maintainer

## Overview

Use the model-catalog workflow that already exists in this repo instead of re-deriving the review process from scratch.
Start from the established scripts and generated-doc pipeline, then inspect only the PR-specific files that the scripts flag.

## Fast Start

Open these files first and treat them as the source of truth for model PR handling:

- `scripts/review_model_pr.py`
- `scripts/smoke_test_models.py`
- `scripts/render_model_docs.py`
- `scripts/verify_table_entries.py`
- `pyhazards/model_catalog.py`
- `pyhazards/model_cards/*.yaml`
- `.github/workflows/model-pr-validation.yml`
- `.github/workflows/model-pr-bot.yml`
- `.github/workflows/model-docs-sync.yml`
- `.github/IMPLEMENTATION.md`
- `docs/README.md`

Do not begin with broad repo discovery unless one of those files is missing or broken.

## Workflow

1. Identify whether the PR is a model PR.
   Use the changed files, the PR description, and `pyhazards/model_cards/*.yaml`.
   If it is not a model PR, say so and fall back to normal review.

2. Run the existing automated review path first.
   If GitHub event data and a base SHA are available, run:
   ```bash
   python scripts/review_model_pr.py \
     --event "$GITHUB_EVENT_PATH" \
     --base-sha "<base_sha>" \
     --report-json /tmp/model-pr-review.json \
     --report-md /tmp/model-pr-review.md
   ```
   If you are reviewing locally without GitHub event payloads, inspect the current diff and run the targeted checks below instead.

3. For touched models, use the cataloged checks instead of inventing ad hoc smoke tests.
   Run:
   ```bash
   python scripts/smoke_test_models.py --models <model_name>
   python scripts/render_model_docs.py --check
   python -c "import pyhazards; print(pyhazards.__version__)"
   ```
   When you changed model cards or model code, also run:
   ```bash
   python scripts/render_model_docs.py
   python scripts/verify_table_entries.py
   python -m pytest tests/test_model_catalog.py
   ```
   If the change should be visible on the published docs site, rebuild the committed HTML too:
   ```bash
   cd docs
   sphinx-build -b html source build/html
   cp -r build/html/* .
   ```
   Do not stop after updating `docs/source/`; this repo publishes the committed `docs/` HTML artifacts.

4. Decide whether to fix or block.
   Patch the PR yourself when the issue is localized and the correct change is clear.
   Write a blocker comment only when the implementation is materially off-spec and fixing it would take substantial time.
   Reuse the report structure from `scripts/review_model_pr.py` so the contributor gets concrete action items.

5. If the PR passes, keep docs aligned and merge when the user asked for processing rather than pure review.
   Generated model tables come from `pyhazards/model_cards/*.yaml`, so a new hazard scenario appears as a new section automatically after `python scripts/render_model_docs.py`.
   Do not hand-edit `docs/source/pyhazards_models.rst` or the generated module pages.
   If a model should remain implemented but not appear in the public catalog, set `include_in_public_catalog: false` in its model card instead of editing generated docs.

6. Report back concisely.
   Include:
   - whether the PR passed or was blocked
   - which model names were touched
   - which commands/tests were run
   - whether merge happened
   - any remaining permission or workflow limitations

## Review Standard

- Treat `pyhazards/model_catalog.py` plus the model cards as the source of truth for model-table/docs behavior.
- Require a valid builder contract: `task` support, `**kwargs`, registry wiring, and explicit shape validation.
- Require a smoke test that matches the card.
- Prefer targeted inspection of touched model files over reading unrelated modules.
- Keep blocker comments technical and specific; do not be vague.

## Operational Notes

- The GitHub Actions bot comments on blocked model PRs, merges passing ones, and syncs generated model docs on push.
- The published site is driven by committed files under `docs/`, so after catalog changes you must think in two stages: generate `docs/source`, then build/copy the HTML site artifacts.
- Email notification is intentionally not part of the workflow.
- If GitHub posting or merging is unavailable in the current environment, still run the local review path and return the exact comment or merge recommendation to the user.
