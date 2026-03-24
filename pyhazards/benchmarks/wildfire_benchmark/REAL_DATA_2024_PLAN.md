# Wildfire Benchmark Real-Data Plan (2024 v1)

## Goal

Run the first real-data wildfire benchmark in `my-copy` using the 2024 data pack that is already available locally.

This plan treats 2024 as the first stable benchmark year because:
- the local 2024 label and weather coverage is already present;
- the benchmark contract already uses 2024 splits;
- we can start with a fair, single-year method comparison before moving to 2025 generalization.

## Benchmark Year

- Benchmark year: `2024`
- Primary task: `Track-O`
- Task definition: predict grid-level or aggregated wildfire occurrence probability `P(y=1 | x)`

## Real-Data 2024 Dataset Pack

### Core inputs

1. **Fire labels / fire history**
- Path: `/home/runyang/ryang/firms/combine`
- Format: daily CSV files such as `2024-01-10.csv`
- Role: primary occurrence label source and lagged fire-history source

2. **Dynamic weather / land-surface forcing**
- Path: `/home/runyang/output2024`
- Format: Prithvi-WxC predicted NetCDF files such as `pred_20240101_18.nc`
- Available channels observed in sample files:
  - `T2M`, `QV2M`, `TQV`, `U10M`, `V10M`, `GWETROOT`, `TS`, `LAI`, `EFLUX`, `HFLUX`, `SWGNT`, `SWTNT`, `LWGAB`, `LWGEM`
- Role: main dynamic feature source

3. **Static fuels / vegetation**
- Path: `/home/runyang/ryang/landfire_fbfm40`
- Role: static fuel and vegetation background

### Recommended v1 optional inputs

4. **Perimeters**
- Path: `/home/runyang/ryang/WFIGS_Perimeters/history_2024`
- Role: event extent, perimeter-derived context, perimeter proximity features

5. **Human activity proxy**
- Candidate paths:
  - `/home/runyang/ryang/WRC_Housing_Density`
  - `/home/runyang/ryang/LandScan_Global_2024`
- Role: ignition proxy and exposure context

## Minimum Real-Data Feature Packs by Model Family

### Classical / Trees
Use aggregated tabular features.

Required:
- FIRMS labels / lagged fire counts
- aggregated weather features from `output2024`
- LANDFIRE static fuel features

Recommended:
- WFIGS perimeter proximity
- housing / population

### Deep Learning
Use raster or raster-sequence tensors.

Required:
- FIRMS rasterized labels
- `output2024` weather tensors
- LANDFIRE static channels

Recommended:
- fire-history channels from FIRMS
- WFIGS perimeter channels or masks

### Satellite Remote Sensing
Use raster tensors with wildfire-specific spatial observations.

Required for v1:
- FIRMS labels / fire-history
- `output2024`
- LANDFIRE

Recommended for later v2:
- GOES FDCF
- HMS Smoke

### Physics / Simulators
Required:
- weather
- fuels
- perimeter or ignition initialization

This group should not block the first real-data benchmark if data conversion takes longer.

### Foundation Models
- `prithvi_wxc`: prioritize weather sequence tensors from `output2024`
- `prithvi_eo_2_tl`, `prithvi_burnscars`: use raster sequences plus static channels and fire-history context

### LLM / MLLM
Do not block v1 on raw NetCDF ingestion.
Use summarized products, rendered maps, metadata, and benchmark-derived inputs after the core benchmark is stable.

## Data Processing Strategy

### Step 1: Build a canonical benchmark grid and date index
- Use `output2024/pred_20240101_18.nc` as a canonical weather grid reference.
- Create a canonical daily date list from `2024-01-01` through `2024-12-31`.
- Align all dynamic inputs to that grid and daily calendar.

### Step 2: Build labels
- Read FIRMS daily CSV files from `/home/runyang/ryang/firms/combine`.
- Rasterize or aggregate them onto the benchmark grid.
- Create:
  - `y_t`: binary occurrence label for day `t`
  - optional lagged fire-history channels from prior days

### Step 3: Build dynamic weather tensors
- Read `output2024/pred_*.nc`
- Daily aggregate if multiple files per day are used
- Select the 14 current channels as the default dynamic pack

### Step 4: Build static tensors
- Reproject or sample LANDFIRE fuels to the benchmark grid
- Add optional human-activity layers if needed

### Step 5: Materialize cached benchmark-ready arrays
Recommended cache layout:

```text
/home/runyang/my-copy/data_cache/wildfire_2024_v1/
  dates.txt
  labels/
    2024-01-01.npy
  met/
    2024-01-01.npy
  static/
    fuel.npy
    housing.npy
    population.npy
  metadata/
    grid.json
    vars.json
```

## Train / Val / Test Protocol

Use the current benchmark contract split:
- Train: `2024-01-01` to `2024-09-30`
- Val: `2024-10-01` to `2024-10-31`
- Test: `2024-11-01` to `2024-12-31`

Rules:
- fit all normalization statistics on train only;
- no future covariates relative to the prediction target;
- store fixed split files for reproducibility.

## Training Recommendations

### Phase A: real-data dry run
Use one seed first.
- Seed: `42`
- Purpose: verify data loading, training loop, output schema, and metric computation

### Phase B: final benchmark runs
Use multi-seed reporting.
- Seeds: `42, 52, 62, 72, 82`
- Report: `mean ± std`

### Classical / Trees
- `logistic_regression`: native binary objective
- `random_forest`: `predict_proba`
- `xgboost`: binary objective, several hundred rounds allowed
- `lightgbm`: binary objective, several hundred rounds allowed

### Deep models
- task: binary occurrence probability
- output: one logit per grid cell / tile / target unit
- loss: `BCEWithLogitsLoss`
- recommended initial schedule:
  - `max_epochs = 120` or higher
  - early stopping monitor: `val_auprc`
  - `patience = 20 to 30`
  - `min_delta = 1e-4`
- current smoke settings are not sufficient for convergence claims

## GPU Policy

Real-data deep-model training should use GPU, not CPU.

Record in `experiment_setting.json`:
- device
- gpu id
- gpu name
- total memory if available

Recommended policy:
- classical models may remain on CPU unless GPU versions are explicitly used
- deep models should default to `cuda:<id>`

## Output Layout

All new real-data benchmark artifacts should be written under:

```text
/home/runyang/my-copy/runs/wildfire_benchmark/real/
```

Recommended run layout:

```text
runs/wildfire_benchmark/real/track_o_2024_real_v1/
  benchmark_contract_snapshot.json
  benchmark_summary.json
  experiment_templates.json
  <model_name>/
    model_template.json
    model_summary.json
    seed_42/
      experiment_setting.json
      history.csv
      loss_curve.png
      metrics.json
```

## Required Per-Seed Outputs

For every model and seed:
- `experiment_setting.json`
- `history.csv`
- `loss_curve.png`
- `metrics.json`

### history.csv should include
At minimum:
- step column (`epoch`, `round`, `iteration`, or `tree_count`)
- `train_loss`
- `val_loss`
- optional learning-rate column when applicable

### loss_curve.png should show
- train loss vs step
- validation loss vs step
- clear title with model name and train unit

## Evaluation Protocol

### Primary metrics
- `AUPRC`

### Secondary metrics
- `AUROC`

### Reliability metrics
- `Brier`
- `NLL`
- `ECE`

### Temporal consistency metrics
- `mean_day_to_day_change`
- `normalized_consistency_score`

### Reporting rules
- report mean and std across seeds for final benchmark numbers
- include train/val loss curves
- log best step
- log converged step

## Recommended Execution Order

1. Build cache from FIRMS + `output2024` + LANDFIRE
2. Run `seed=42` dry run on 4 representative models:
   - `logistic_regression`
   - `xgboost`
   - `unet`
   - `convlstm`
3. Validate output artifacts and metric computation
4. Expand to the rest of the main benchmark roster
5. Add remote-sensing / foundation / simulator tracks afterwards

## Immediate Implementation Notes

- The current `track_o_2024_v1.json` still points to `/home/runyang/ryang/firms_download/combine`.
- The locally verified combined FIRMS label directory is `/home/runyang/ryang/firms/combine`.
- The first real-data contract should use the verified local path.

