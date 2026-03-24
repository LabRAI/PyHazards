from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from pyhazards.datasets.wildfire import (
    WildfireTrackO2024RasterDataset,
    WildfireTrackO2024TabularDataset,
    WildfireTrackO2024TemporalDataset,
)
from pyhazards.models import build_model
from pyhazards.models.convlstm import ConvLSTMTrackOConfig, train_convlstm_track_o
from pyhazards.models.unet import UNetTrackOConfig, train_unet_track_o
from pyhazards.models.unet import binary_ece, normalized_consistency_score
from pyhazards.utils.hardware import auto_device

from .artifacts import build_model_template, mean_std, plot_loss_curve, write_history_csv, write_json
from .catalog import load_contract, load_model_catalog
from .experiment_settings import build_default_experiment_setting
from .layout import WILDFIRE_RUNS_ROOT, prepare_run_paths


REPRESENTATIVE_MODELS = ("logistic_regression", "random_forest", "xgboost", "lightgbm", "unet", "convlstm")


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _positive_class_prob(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    pred = model(x)
    arr = _to_numpy(pred)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr[:, 1].astype(np.float32)
    return arr.reshape(-1).astype(np.float32)


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float32).reshape(-1)
    y_prob = np.clip(y_prob, 1e-7, 1.0 - 1e-7)
    mean_change = float(np.mean(np.abs(np.diff(np.sort(y_prob))))) if len(y_prob) > 1 else 0.0

    def _safe(callable_obj):
        try:
            return float(callable_obj())
        except Exception:
            return float("nan")

    return {
        "auprc": _safe(lambda: average_precision_score(y_true, y_prob)),
        "auroc": _safe(lambda: roc_auc_score(y_true, y_prob)),
        "brier": _safe(lambda: brier_score_loss(y_true, y_prob)),
        "nll": _safe(lambda: log_loss(y_true, y_prob, labels=[0, 1])),
        "ece": _safe(lambda: binary_ece(y_true, y_prob, n_bins=15)),
        "mean_day_to_day_change": mean_change,
        "normalized_consistency_score": normalized_consistency_score(mean_change),
    }


def _catalog_lookup(names: Sequence[str]) -> list[dict[str, Any]]:
    catalog = load_model_catalog("main")
    allowed = set(names)
    selected = [row for row in catalog if row["name"] in allowed]
    if len(selected) != len(allowed):
        found = {row["name"] for row in selected}
        missing = sorted(allowed - found)
        raise KeyError(f"Missing models in wildfire main catalog: {missing}")
    return selected


def _build_setting(
    *,
    contract: dict[str, Any],
    model_spec: dict[str, Any],
    seed: int,
    num_steps: int,
    best_step: int,
    converged_step: int,
    metrics: dict[str, float],
    notes: dict[str, Any],
    learning_weight: dict[str, Any],
) -> dict[str, Any]:
    setting = build_default_experiment_setting(
        model_name=str(model_spec["name"]),
        display_name=str(model_spec["display_name"]),
        group=str(model_spec["group"]),
        source_tier=str(model_spec["source_tier"]),
        train_unit=str(model_spec["train_unit"]),
        defaults=model_spec.get("defaults", {}),
        seed=int(seed),
        num_steps=int(num_steps),
        best_step=int(best_step),
        converged_step=int(converged_step),
        step_name=str(model_spec["train_unit"]),
        mode=str(contract["mode"]),
        task=str(contract["task"]),
        notes=notes,
        metrics=metrics,
    )
    setting.benchmark.contract_version = str(contract["contract_version"])
    setting.run.learning_weight = learning_weight
    return setting.to_dict()


def _run_logistic_regression(
    *,
    bundle,
    seed: int,
    model_spec: dict[str, Any],
) -> dict[str, Any]:
    defaults = dict(model_spec.get("defaults", {}))
    model = build_model("logistic_regression", task="classification", **defaults)
    model.fit_bundle(bundle, train_split="train", val_split="val")

    train_split = bundle.get_split("train")
    val_split = bundle.get_split("val")
    test_split = bundle.get_split("test")

    train_prob = np.clip(_positive_class_prob(model, train_split.inputs), 1e-7, 1.0 - 1e-7)
    val_prob = np.clip(_positive_class_prob(model, val_split.inputs), 1e-7, 1.0 - 1e-7)
    test_prob = np.clip(_positive_class_prob(model, test_split.inputs), 1e-7, 1.0 - 1e-7)

    train_true = _to_numpy(train_split.targets).reshape(-1).astype(np.float32)
    val_true = _to_numpy(val_split.targets).reshape(-1).astype(np.float32)
    test_true = _to_numpy(test_split.targets).reshape(-1).astype(np.float32)

    train_loss = float(log_loss(train_true, train_prob, labels=[0, 1]))
    val_loss = float(log_loss(val_true, val_prob, labels=[0, 1]))
    fitted_steps = int(np.max(getattr(model.estimator, "n_iter_", np.asarray([1]))))
    history = [{"iteration": fitted_steps, "train_loss": train_loss, "val_loss": val_loss}]

    return {
        "history": history,
        "val_metrics": _binary_metrics(val_true, val_prob),
        "test_metrics": _binary_metrics(test_true, test_prob),
        "best_step": fitted_steps,
        "converged_step": fitted_steps,
        "learning_weight": {"kind": "class_weight", "value": defaults.get("class_weight", "balanced")},
        "notes": {
            "model_source": "pyhazards.models.logistic_regression",
            "device": "cpu",
            "gpu_assignment": None,
        },
    }


def _run_random_forest(
    *,
    bundle,
    seed: int,
    model_spec: dict[str, Any],
) -> dict[str, Any]:
    defaults = dict(model_spec.get("defaults", {}))
    model = build_model("random_forest", task="classification", **defaults)
    model.fit_bundle(bundle, train_split="train", val_split="val")

    train_split = bundle.get_split("train")
    val_split = bundle.get_split("val")
    test_split = bundle.get_split("test")

    train_prob = np.clip(_positive_class_prob(model, train_split.inputs), 1e-7, 1.0 - 1e-7)
    val_prob = np.clip(_positive_class_prob(model, val_split.inputs), 1e-7, 1.0 - 1e-7)
    test_prob = np.clip(_positive_class_prob(model, test_split.inputs), 1e-7, 1.0 - 1e-7)

    train_true = _to_numpy(train_split.targets).reshape(-1).astype(np.float32)
    val_true = _to_numpy(val_split.targets).reshape(-1).astype(np.float32)
    test_true = _to_numpy(test_split.targets).reshape(-1).astype(np.float32)

    train_loss = float(log_loss(train_true, train_prob, labels=[0, 1]))
    val_loss = float(log_loss(val_true, val_prob, labels=[0, 1]))
    n_estimators = int(getattr(model.estimator, 'n_estimators', defaults.get('n_estimators', 500)))
    history = [{"tree": n_estimators, "train_loss": train_loss, "val_loss": val_loss}]

    return {
        "history": history,
        "val_metrics": _binary_metrics(val_true, val_prob),
        "test_metrics": _binary_metrics(test_true, test_prob),
        "best_step": n_estimators,
        "converged_step": n_estimators,
        "learning_weight": {"kind": "class_weight", "value": defaults.get("class_weight", "balanced_subsample")},
        "notes": {
            "model_source": "pyhazards.models.random_forest",
            "device": "cpu",
            "gpu_assignment": None,
        },
    }


def _run_xgboost(
    *,
    bundle,
    seed: int,
    model_spec: dict[str, Any],
    num_boost_round: int,
) -> dict[str, Any]:
    import xgboost as xgb

    defaults = dict(model_spec.get("defaults", {}))
    x_train = _to_numpy(bundle.get_split("train").inputs)
    y_train = _to_numpy(bundle.get_split("train").targets).reshape(-1)
    x_val = _to_numpy(bundle.get_split("val").inputs)
    y_val = _to_numpy(bundle.get_split("val").targets).reshape(-1)
    x_test = _to_numpy(bundle.get_split("test").inputs)
    y_test = _to_numpy(bundle.get_split("test").targets).reshape(-1)

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    dtest = xgb.DMatrix(x_test, label=y_test)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": int(defaults.get("max_depth", 8)),
        "eta": float(defaults.get("eta", 0.05)),
        "subsample": float(defaults.get("subsample", 0.8)),
        "colsample_bytree": float(defaults.get("colsample_bytree", 0.8)),
        "seed": int(seed),
    }
    evals_result: dict[str, Any] = {}
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(num_boost_round),
        evals=[(dtrain, "train"), (dval, "val")],
        evals_result=evals_result,
        verbose_eval=False,
    )

    train_curve = [float(v) for v in evals_result.get("train", {}).get("logloss", [])]
    val_curve = [float(v) for v in evals_result.get("val", {}).get("logloss", [])]
    history = [
        {"round": idx + 1, "train_loss": tr, "val_loss": va}
        for idx, (tr, va) in enumerate(zip(train_curve, val_curve, strict=True))
    ]
    best_step = int(np.argmin(val_curve) + 1) if val_curve else 1
    test_prob = np.clip(np.asarray(booster.predict(dtest), dtype=np.float32), 1e-7, 1.0 - 1e-7)
    val_prob = np.clip(np.asarray(booster.predict(dval), dtype=np.float32), 1e-7, 1.0 - 1e-7)

    return {
        "history": history,
        "val_metrics": _binary_metrics(y_val, val_prob),
        "test_metrics": _binary_metrics(y_test, test_prob),
        "best_step": best_step,
        "converged_step": len(history),
        "learning_weight": {"kind": "native_binary_objective", "value": "binary:logistic"},
        "notes": {
            "model_source": "pyhazards.models.xgboost",
            "device": "cpu",
            "gpu_assignment": None,
        },
    }


def _run_lightgbm(
    *,
    bundle,
    seed: int,
    model_spec: dict[str, Any],
    num_boost_round: int,
) -> dict[str, Any]:
    import lightgbm as lgb

    defaults = dict(model_spec.get("defaults", {}))
    x_train = _to_numpy(bundle.get_split("train").inputs)
    y_train = _to_numpy(bundle.get_split("train").targets).reshape(-1)
    x_val = _to_numpy(bundle.get_split("val").inputs)
    y_val = _to_numpy(bundle.get_split("val").targets).reshape(-1)
    x_test = _to_numpy(bundle.get_split("test").inputs)
    y_test = _to_numpy(bundle.get_split("test").targets).reshape(-1)

    dtrain = lgb.Dataset(x_train, label=y_train)
    dval = lgb.Dataset(x_val, label=y_val, reference=dtrain)
    evals_result: dict[str, Any] = {}
    train_pos = max(float(y_train.sum()), 1.0)
    train_neg = max(float(y_train.size - y_train.sum()), 1.0)
    scale_pos_weight = float(defaults.get("scale_pos_weight", min(train_neg / train_pos, 500.0)))
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": int(defaults.get("num_leaves", 15)),
        "learning_rate": float(defaults.get("learning_rate", 0.03)),
        "feature_fraction": float(defaults.get("feature_fraction", 0.8)),
        "bagging_fraction": float(defaults.get("bagging_fraction", 0.8)),
        "bagging_freq": int(defaults.get("bagging_freq", 1)),
        "min_data_in_leaf": int(defaults.get("min_data_in_leaf", 200)),
        "min_sum_hessian_in_leaf": float(defaults.get("min_sum_hessian_in_leaf", 1e-3)),
        "lambda_l2": float(defaults.get("lambda_l2", 1.0)),
        "max_depth": int(defaults.get("max_depth", -1)),
        "scale_pos_weight": scale_pos_weight,
        "seed": int(seed),
        "verbose": -1,
        "force_col_wise": True,
    }
    booster = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=int(num_boost_round),
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.log_evaluation(period=0),
            lgb.record_evaluation(evals_result),
        ],
    )

    train_curve = [float(v) for v in evals_result.get("train", {}).get("binary_logloss", [])]
    val_curve = [float(v) for v in evals_result.get("val", {}).get("binary_logloss", [])]
    history = [
        {"round": idx + 1, "train_loss": tr, "val_loss": va}
        for idx, (tr, va) in enumerate(zip(train_curve, val_curve, strict=True))
    ]
    best_step = int(np.argmin(val_curve) + 1) if val_curve else len(history)
    val_prob = np.clip(np.asarray(booster.predict(x_val), dtype=np.float32), 1e-7, 1.0 - 1e-7)
    test_prob = np.clip(np.asarray(booster.predict(x_test), dtype=np.float32), 1e-7, 1.0 - 1e-7)

    return {
        "history": history,
        "val_metrics": _binary_metrics(y_val, val_prob),
        "test_metrics": _binary_metrics(y_test, test_prob),
        "best_step": best_step,
        "converged_step": len(history),
        "learning_weight": {"kind": "scale_pos_weight", "value": float(scale_pos_weight), "derived_from": "train_neg_over_pos_clipped"},
        "notes": {
            "model_source": "pyhazards.models.lightgbm",
            "device": "cpu",
            "gpu_assignment": None,
        },
    }


def _run_unet(
    *,
    bundle,
    seed: int,
    device: str,
    max_epochs: int,
    patience: int,
) -> dict[str, Any]:
    train_split = bundle.get_split("train")
    val_split = bundle.get_split("val")
    test_split = bundle.get_split("test")

    cfg = UNetTrackOConfig(
        in_channels=int(bundle.feature_spec.channels or train_split.inputs.shape[1]),
        batch_size=4,
        max_epochs=int(max_epochs),
        early_stopping_rounds=int(patience),
        seed=int(seed),
        device=device,
    )
    model, history, val_metrics, best_epoch, pos_weight = train_unet_track_o(
        _to_numpy(train_split.inputs),
        _to_numpy(train_split.targets),
        _to_numpy(val_split.inputs),
        _to_numpy(val_split.targets),
        cfg,
    )

    with torch.no_grad():
        logits = model(test_split.inputs.to(torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")))
        test_prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    test_true = _to_numpy(test_split.targets).reshape(-1)

    return {
        "history": history,
        "val_metrics": val_metrics,
        "test_metrics": _binary_metrics(test_true, test_prob),
        "best_step": int(best_epoch),
        "converged_step": len(history),
        "learning_weight": {"kind": "pos_weight_neg_over_pos", "value": float(pos_weight), "clip_max": float(cfg.pos_weight_clip_max)},
        "notes": {
            "model_source": "pyhazards.models.unet",
            "device": device,
            "gpu_assignment": device if str(device).startswith("cuda") else None,
        },
    }


def _run_convlstm(
    *,
    bundle,
    seed: int,
    device: str,
    max_epochs: int,
    patience: int,
    history_len: int,
) -> dict[str, Any]:
    train_split = bundle.get_split("train")
    val_split = bundle.get_split("val")
    test_split = bundle.get_split("test")

    cfg = ConvLSTMTrackOConfig(
        seq_len=int(history_len),
        in_channels=int(bundle.feature_spec.channels or train_split.inputs.shape[2]),
        batch_size=2,
        max_epochs=int(max_epochs),
        early_stopping_rounds=int(patience),
        seed=int(seed),
        device=device,
    )
    model, history, val_metrics, best_epoch, pos_weight = train_convlstm_track_o(
        _to_numpy(train_split.inputs),
        _to_numpy(train_split.targets),
        _to_numpy(val_split.inputs),
        _to_numpy(val_split.targets),
        cfg,
    )

    eval_device = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        logits = model(test_split.inputs.to(eval_device))
        test_prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    test_true = _to_numpy(test_split.targets).reshape(-1)

    return {
        "history": history,
        "val_metrics": val_metrics,
        "test_metrics": _binary_metrics(test_true, test_prob),
        "best_step": int(best_epoch),
        "converged_step": len(history),
        "learning_weight": {"kind": "pos_weight_neg_over_pos", "value": float(pos_weight), "clip_max": float(cfg.pos_weight_clip_max)},
        "notes": {
            "model_source": "pyhazards.models.convlstm",
            "device": device,
            "gpu_assignment": device if str(device).startswith("cuda") else None,
        },
    }


def _write_per_seed_outputs(
    *,
    contract: dict[str, Any],
    model_spec: dict[str, Any],
    seed: int,
    run_name: str,
    result: dict[str, Any],
    notes_extra: dict[str, Any],
) -> dict[str, Any]:
    paths = prepare_run_paths(track="real", run_name=run_name, model_name=str(model_spec["name"]), seed=int(seed), create=True)
    write_history_csv(paths.history_csv_path, result["history"])
    plot_loss_curve(result["history"], str(model_spec["train_unit"]), paths.loss_curve_path, f"{model_spec['display_name']} ({model_spec['train_unit']})")
    metrics_payload = {
        "val": result["val_metrics"],
        "test": result["test_metrics"],
        "best_step": int(result["best_step"]),
        "converged_step": int(result["converged_step"]),
    }
    write_json(paths.metrics_path, metrics_payload)

    setting = _build_setting(
        contract=contract,
        model_spec=model_spec,
        seed=int(seed),
        num_steps=int(result["converged_step"]),
        best_step=int(result["best_step"]),
        converged_step=int(result["converged_step"]),
        metrics=result["test_metrics"],
        notes={**result["notes"], **notes_extra, "val_metrics": result["val_metrics"]},
        learning_weight=result["learning_weight"],
    )
    write_json(paths.experiment_setting_path, setting)
    return {"paths": paths, "metrics": metrics_payload, "setting": setting}


def run_real_baselines(
    *,
    cache_dir: str | Path = "/home/runyang/my-copy/data_cache/wildfire_2024_v1",
    run_name: str = "track_o_2024_real_v1_first4_dryrun",
    models: Sequence[str] | None = None,
    seed: int = 42,
    train_limit_days: int | None = None,
    val_limit_days: int | None = None,
    test_limit_days: int | None = None,
    tabular_downsample: int = 8,
    raster_downsample: int = 4,
    temporal_downsample: int = 8,
    temporal_history: int = 6,
    xgboost_rounds: int = 120,
    lightgbm_rounds: int = 120,
    unet_epochs: int = 12,
    convlstm_epochs: int = 12,
    deep_patience: int = 4,
    device: str | None = None,
) -> Path:
    cache_root = Path(cache_dir)
    contract = load_contract(Path(__file__).resolve().parents[2] / "configs" / "wildfire_benchmark" / "track_o_2024_real_v1.json")
    selected_names = tuple(models or REPRESENTATIVE_MODELS)
    model_specs = _catalog_lookup(selected_names)

    run_root = WILDFIRE_RUNS_ROOT / "real" / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "benchmark_contract_snapshot.json", contract)

    dataset_common = {
        "cache_dir": str(cache_root),
        "train_limit_days": train_limit_days,
        "val_limit_days": val_limit_days,
        "test_limit_days": test_limit_days,
    }
    bundles: dict[str, Any] = {}
    if any(name in selected_names for name in ("logistic_regression", "random_forest", "xgboost", "lightgbm")):
        bundles["tabular"] = WildfireTrackO2024TabularDataset(
            downsample_factor=tabular_downsample,
            **dataset_common,
        ).load()
    if "unet" in selected_names:
        bundles["raster"] = WildfireTrackO2024RasterDataset(
            downsample_factor=raster_downsample,
            **dataset_common,
        ).load()
    if "convlstm" in selected_names:
        bundles["temporal"] = WildfireTrackO2024TemporalDataset(
            history=temporal_history,
            downsample_factor=temporal_downsample,
            **dataset_common,
        ).load()

    device_text = str(device or auto_device())
    benchmark_rows: List[dict[str, Any]] = []
    templates_index: dict[str, Any] = {}

    for model_spec in model_specs:
        name = str(model_spec["name"])
        model_root = run_root / name
        model_root.mkdir(parents=True, exist_ok=True)
        template = build_model_template(contract, model_spec)
        templates_index[name] = template
        write_json(model_root / "model_template.json", template)

        if name == "logistic_regression":
            result = _run_logistic_regression(bundle=bundles["tabular"], seed=seed, model_spec=model_spec)
            dataset_meta = bundles["tabular"].metadata
        elif name == "random_forest":
            result = _run_random_forest(bundle=bundles["tabular"], seed=seed, model_spec=model_spec)
            dataset_meta = bundles["tabular"].metadata
        elif name == "xgboost":
            result = _run_xgboost(bundle=bundles["tabular"], seed=seed, model_spec=model_spec, num_boost_round=xgboost_rounds)
            dataset_meta = bundles["tabular"].metadata
        elif name == "lightgbm":
            result = _run_lightgbm(bundle=bundles["tabular"], seed=seed, model_spec=model_spec, num_boost_round=lightgbm_rounds)
            dataset_meta = bundles["tabular"].metadata
        elif name == "unet":
            result = _run_unet(bundle=bundles["raster"], seed=seed, device=device_text, max_epochs=unet_epochs, patience=deep_patience)
            dataset_meta = bundles["raster"].metadata
        elif name == "convlstm":
            result = _run_convlstm(
                bundle=bundles["temporal"],
                seed=seed,
                device=device_text,
                max_epochs=convlstm_epochs,
                patience=deep_patience,
                history_len=temporal_history,
            )
            dataset_meta = bundles["temporal"].metadata
        else:
            raise ValueError(f"Unsupported representative model: {name}")

        has_static_fuel = bool(dataset_meta.get("has_static_fuel", False))
        payload = _write_per_seed_outputs(
            contract=contract,
            model_spec=model_spec,
            seed=seed,
            run_name=run_name,
            result=result,
            notes_extra={
                "cache_root": str(cache_root),
                "dataset_metadata": dataset_meta,
                "split_version": "cache_2024_v1",
                "feature_set_version": "weather_plus_fuel_v1" if has_static_fuel else "weather_only_v1_static_fuel_pending",
                "static_fuel_status": "aligned" if has_static_fuel else "manifest_only",
            },
        )

        metric_stats = {k: mean_std([float(v)]) for k, v in result["test_metrics"].items()}
        write_json(
            model_root / "model_summary.json",
            {
                "model": {
                    "name": name,
                    "display_name": model_spec["display_name"],
                    "group": model_spec["group"],
                    "source_tier": model_spec["source_tier"],
                    "train_unit": model_spec["train_unit"],
                },
                "mode": contract["mode"],
                "n_seeds": 1,
                "seeds": [int(seed)],
                "metrics_mean_std": metric_stats,
                "per_seed": [
                    {
                        "seed": int(seed),
                        "best_step": int(result["best_step"]),
                        "converged_step": int(result["converged_step"]),
                        "train_unit": model_spec["train_unit"],
                        **result["test_metrics"],
                    }
                ],
            },
        )
        benchmark_rows.append(
            {
                "name": name,
                "display_name": model_spec["display_name"],
                "group": model_spec["group"],
                "source_tier": model_spec["source_tier"],
                "train_unit": model_spec["train_unit"],
                "auprc_mean": metric_stats.get("auprc", {}).get("mean"),
                "auprc_std": metric_stats.get("auprc", {}).get("std"),
                "auroc_mean": metric_stats.get("auroc", {}).get("mean"),
                "auroc_std": metric_stats.get("auroc", {}).get("std"),
                "brier_mean": metric_stats.get("brier", {}).get("mean"),
                "nll_mean": metric_stats.get("nll", {}).get("mean"),
                "ece_mean": metric_stats.get("ece", {}).get("mean"),
                "normalized_consistency_score_mean": metric_stats.get("normalized_consistency_score", {}).get("mean"),
            }
        )

    write_json(
        run_root / "benchmark_summary.json",
        {
            "benchmark": {
                "name": contract["benchmark_name"],
                "contract_version": contract["contract_version"],
                "mode": contract["mode"],
                "task": contract["task"],
                "generated_at": datetime.now().isoformat(),
                "note": "First real-data dry run on the 2024 wildfire cache.",
                "cache_root": str(cache_root),
            },
            "models_selected": list(selected_names),
            "n_models": len(selected_names),
            "seeds": [int(seed)],
            "rows": benchmark_rows,
        },
    )
    write_json(
        run_root / "experiment_templates.json",
        {
            "template_version": "track_o_model_template_v1",
            "generated_at": datetime.now().isoformat(),
            "models": templates_index,
        },
    )
    return run_root


__all__ = ["REPRESENTATIVE_MODELS", "run_real_baselines"]
