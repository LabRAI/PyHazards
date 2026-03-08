<!-- pyhazards:model-pr -->

## PR Type
- [ ] Model contribution
- [ ] Non-model change

## Model Summary
Describe the model architecture, intended public API, and what was ported from the paper or source repository.
If this is not a model PR, write `N/A`.

## Hazard Scenario
State the hazard family that should own the model table entry (for example, Wildfire or Flood).
If this introduces a new hazard scenario, name it explicitly here.

## Registry Name
List the `build_model(name=...)` entrypoints added or changed in this PR.

## Paper / Source
Link the paper, upstream repository, or technical reference used for the implementation.

## Smoke Test
Document the smoke test command(s) you ran locally, or reference the updated
`pyhazards/model_cards/<model_name>.yaml` smoke-test spec.

## Parity Notes
List any intentional differences from the original implementation, especially if optimizer,
preprocessing, outputs, or training objectives changed.
