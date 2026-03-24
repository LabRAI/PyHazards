from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

from .artifacts import (
    build_experiment_setting_from_run_output,
    build_model_template,
    mean_std,
    plot_loss_curve,
    write_history_csv,
    write_json,
)
from .catalog import load_contract, load_model_catalog, parse_seed_list, select_models
from .layout import WILDFIRE_RUNS_ROOT, prepare_run_paths


def run_smoke_batch(
    *,
    adapter_factory: Callable[[Dict[str, Any], Dict[str, Any], Dict[str, int]], Any],
    run_name: str | None = None,
    track: str = "smoke",
    catalog_kind: str = "main",
    catalog_path: str | Path | None = None,
    contract_path: str | Path | None = None,
    source_tier: str = "all",
    models: str | List[str] | None = None,
    seeds: str | List[int] | None = None,
    limit_models: int = 0,
    step_limits: Dict[str, int] | None = None,
) -> Path:
    contract = load_contract(contract_path)
    catalog = load_model_catalog(catalog_kind, catalog_path)
    selected_models = select_models(catalog, source_tier=source_tier, models=models, limit_models=limit_models)
    if not selected_models:
        raise ValueError("No wildfire benchmark models selected.")

    seed_list = parse_seed_list(seeds)
    step_limits = step_limits or {"epoch": 60, "round": 300, "iteration": 250, "tree": 300}
    run_name = run_name or f"smoke_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_root = WILDFIRE_RUNS_ROOT / track / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "benchmark_contract_snapshot.json", contract)

    benchmark_rows: List[Dict[str, Any]] = []
    templates_index: Dict[str, Dict[str, Any]] = {}

    for model_spec in selected_models:
        model_name = str(model_spec["name"])
        model_root = run_root / model_name
        model_root.mkdir(parents=True, exist_ok=True)

        template = build_model_template(contract, model_spec)
        templates_index[model_name] = template
        write_json(model_root / "model_template.json", template)

        metric_pool: Dict[str, List[float]] = {}
        per_seed_rows: List[Dict[str, Any]] = []

        for seed in seed_list:
            paths = prepare_run_paths(track=track, run_name=run_name, model_name=model_name, seed=int(seed), create=True)
            adapter = adapter_factory(model_spec, contract, step_limits)
            run_output = adapter.run(seed=int(seed))

            write_history_csv(paths.history_csv_path, run_output.history)
            plot_loss_curve(run_output.history, run_output.train_unit, paths.loss_curve_path, f"{model_spec['display_name']} ({run_output.train_unit})")

            if hasattr(adapter, 'build_experiment_setting'):
                experiment_setting = adapter.build_experiment_setting(seed=int(seed), run_output=run_output)
            else:
                experiment_setting = build_experiment_setting_from_run_output(
                    contract=contract,
                    model_spec=model_spec,
                    seed=int(seed),
                    run_output=run_output,
                )
            write_json(paths.experiment_setting_path, experiment_setting)
            write_json(paths.metrics_path, run_output.metrics)

            for key, value in run_output.metrics.items():
                metric_pool.setdefault(key, []).append(float(value))
            per_seed_rows.append(
                {
                    'seed': int(seed),
                    'best_step': int(run_output.best_step),
                    'converged_step': int(run_output.converged_step),
                    'train_unit': run_output.train_unit,
                    **run_output.metrics,
                }
            )

        metric_stats = {k: mean_std(v) for k, v in metric_pool.items()}
        write_json(
            model_root / 'model_summary.json',
            {
                'model': {
                    'name': model_name,
                    'display_name': model_spec['display_name'],
                    'group': model_spec['group'],
                    'source_tier': model_spec['source_tier'],
                    'train_unit': model_spec['train_unit'],
                },
                'mode': contract['mode'],
                'n_seeds': len(seed_list),
                'seeds': seed_list,
                'metrics_mean_std': metric_stats,
                'per_seed': per_seed_rows,
            },
        )
        benchmark_rows.append(
            {
                'name': model_name,
                'display_name': model_spec['display_name'],
                'group': model_spec['group'],
                'source_tier': model_spec['source_tier'],
                'train_unit': model_spec['train_unit'],
                'auprc_mean': metric_stats.get('auprc', {}).get('mean'),
                'auprc_std': metric_stats.get('auprc', {}).get('std'),
                'auroc_mean': metric_stats.get('auroc', {}).get('mean'),
                'auroc_std': metric_stats.get('auroc', {}).get('std'),
                'brier_mean': metric_stats.get('brier', {}).get('mean'),
                'nll_mean': metric_stats.get('nll', {}).get('mean'),
                'ece_mean': metric_stats.get('ece', {}).get('mean'),
                'normalized_consistency_score_mean': metric_stats.get('normalized_consistency_score', {}).get('mean'),
            }
        )

    write_json(
        run_root / 'benchmark_summary.json',
        {
            'benchmark': {
                'name': contract['benchmark_name'],
                'contract_version': contract['contract_version'],
                'mode': contract['mode'],
                'task': contract['task'],
                'generated_at': datetime.now().isoformat(),
                'note': 'Adapter-level smoke run.',
                'contract_path': str(contract_path) if contract_path else 'pyhazards/configs/wildfire_benchmark/track_o_2024_v1.json',
                'catalog_kind': catalog_kind,
            },
            'models_selected': [m['name'] for m in selected_models],
            'n_models': len(selected_models),
            'seeds': seed_list,
            'rows': benchmark_rows,
        },
    )

    templates_payload = {
        'template_version': 'track_o_model_template_v1',
        'generated_at': datetime.now().isoformat(),
        'models': templates_index,
    }
    write_json(run_root / 'experiment_templates.json', templates_payload)
    if catalog_kind == 'main':
        write_json(run_root / 'experiment_templates_22.json', templates_payload)
    return run_root
