"""Train and score registry models on a Track-O wildfire occurrence cache.

Adapted from ``pyhazards/benchmarks/wildfire_benchmark/real_runner.py`` and
``scripts/run_wildfire_2024_real_baselines.py`` of PyHazards PR #33 (runyangxu). The PR trained its
own model copies; this script builds the models from the PyHazards registry instead:

- ``random_forest`` and ``xgboost`` (tabular layout, fitted with ``model.fit``),
- ``logistic_regression`` (raster layout: the WildfireSpreadTS per-pixel 3x3 linear model; PR #33
  used scikit-learn ``LogisticRegression`` on the tabular rows instead),
- ``unet`` (raster layout, the Ronneberger et al. U-Net with ``padding="same"`` and one output
  channel, so the grid is cropped to a multiple of 16),
- ``convlstm`` (temporal layout),
- any other registry model given with ``--layout name=raster|temporal|tabular``.

Gradient-trained models use PR #33's protocol: AdamW (lr 1e-3, weight decay 1e-4), BCE-with-logits
with ``pos_weight = negatives / positives`` of the training split clipped at 50, early stopping on
validation average precision (patience 20, min delta 1e-4) and the best epoch's weights. Test
scores come from the ``wildfire`` benchmark (``wildfire.danger``: accuracy, macro F1, ROC AUC,
PR AUC), plus Brier score, log loss, expected calibration error (15 bins) and the mean absolute
day-to-day change of the predicted probabilities between consecutive days (a smoothness
diagnostic, not a skill score). PR #33's "normalized consistency score" (1 minus that change) is
not reported, and LightGBM is not run because PyHazards has no LightGBM model.

Per model and seed, ``<output-dir>/<model>/seed_<seed>/`` gets ``history.csv``, ``metrics.json``,
``experiment_setting.json``, the benchmark report and, when matplotlib is installed
(``pip install 'pyhazards[plot]'``), ``loss_curve.png``. ``<output-dir>/summary.json`` has the mean
and standard deviation over seeds (PR #33 used seeds 42 for the dry run and 42,52,62,72,82 for the
final runs).

Example::

    python scripts/run_wildfire_track_o_baselines.py --cache-dir data/track_o_2024 \\
        --output-dir runs/track_o_2024 --models logistic_regression,random_forest,unet,convlstm --seeds 42
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import platform
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

import pyhazards  # noqa: E402
from pyhazards.benchmarks import run_benchmark  # noqa: E402
from pyhazards.configs import BenchmarkConfig, DatasetRef, ExperimentConfig, ModelRef, ReportConfig  # noqa: E402
from pyhazards.datasets import load_dataset  # noqa: E402
from pyhazards.models import build_model  # noqa: E402
from pyhazards.utils.hardware import auto_device  # noqa: E402

LAYOUT_DATASETS = {
    "raster": "wildfire_track_o_raster",
    "temporal": "wildfire_track_o_temporal",
    "tabular": "wildfire_track_o_tabular",
}
MODEL_LAYOUTS = {
    "logistic_regression": "raster",
    "unet": "raster",
    "convlstm": "temporal",
    "random_forest": "tabular",
    "xgboost": "tabular",
}
MODEL_KWARGS: Dict[str, Dict[str, Any]] = {
    # Zero-padded 3x3 convolutions (a PyHazards option of the paper U-Net) and one logit per cell.
    "unet": {"padding": "same", "out_channels": 1},
}
DEFAULT_MODELS = "logistic_regression,random_forest,unet,convlstm"


# ---------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------
def probability_diagnostics(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 15) -> Dict[str, float]:
    """Brier score, log loss and expected calibration error (equal-width bins) of binary probabilities."""
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(prob, dtype=np.float64).reshape(-1)
    if y.shape != p.shape:
        raise ValueError(f"y_true and prob differ in size: {y.shape} vs {p.shape}")
    clipped = np.clip(p, 1e-7, 1.0 - 1e-7)
    bins = np.minimum((p * n_bins).astype(np.int64), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(y[in_bin].mean() - p[in_bin].mean())
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "nll": float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))),
        "ece": float(ece),
    }


def mean_day_to_day_change(prob_by_day: np.ndarray, dates: Sequence[str]) -> float:
    """Mean |p(day) - p(day - 1)| over cells and over pairs of consecutive calendar days.

    ``prob_by_day`` is ``(n_days, ...)`` in the order of ``dates``. Returns NaN without such a pair.
    """
    flat = np.asarray(prob_by_day, dtype=np.float64).reshape(len(dates), -1)
    parsed = [date.fromisoformat(str(day)) for day in dates]
    changes = [
        np.mean(np.abs(flat[i] - flat[i - 1])) for i in range(1, len(parsed)) if (parsed[i] - parsed[i - 1]).days == 1
    ]
    return float(np.mean(changes)) if changes else float("nan")


# ---------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------
def _is_estimator(model: nn.Module) -> bool:
    return hasattr(model, "custom_fit_reason") and hasattr(model, "fit")


@torch.no_grad()
def _predict_logits(model: nn.Module, inputs: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    model.eval()
    return torch.cat([model(chunk.to(device)).float().cpu() for chunk in torch.split(inputs, batch_size)])


def _positive_probability(model: nn.Module, inputs: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    if _is_estimator(model):
        return np.asarray(model.predict_proba(inputs))[:, 1]
    return torch.sigmoid(_predict_logits(model, inputs, batch_size, device)).numpy()


def train_gradient_model(
    model: nn.Module,
    bundle,
    *,
    seed: int,
    device: torch.device,
    max_epochs: int,
    patience: int,
    min_delta: float = 1e-4,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 4,
    pos_weight_clip: float = 50.0,
) -> Dict[str, Any]:
    train, val = bundle.get_split("train"), bundle.get_split("val")
    positives = float(train.targets.sum())
    negatives = float(train.targets.numel()) - positives
    pos_weight = min(negatives / max(positives, 1.0), pos_weight_clip)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train.inputs, train.targets),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    history: List[Dict[str, float]] = []
    best_score, best_epoch, best_state, waited = -math.inf, 0, None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        total, count = 0.0, 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            if logits.shape != targets.shape:
                raise ValueError(
                    f"model output {tuple(logits.shape)} does not match the targets {tuple(targets.shape)}; "
                    "Track-O needs one logit per cell (out_channels=1, output size = input size)."
                )
            loss = loss_fn(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * inputs.shape[0]
            count += inputs.shape[0]
        val_logits = _predict_logits(model, val.inputs, batch_size, device)
        val_loss = float(loss_fn(val_logits.to(device), val.targets.to(device)))
        val_targets = val.targets.numpy().reshape(-1) > 0.5
        val_ap = (
            float(average_precision_score(val_targets, torch.sigmoid(val_logits).numpy().reshape(-1)))
            if val_targets.any()
            else 0.0
        )
        history.append({"step": epoch, "train_loss": total / max(count, 1), "val_loss": val_loss, "val_average_precision": val_ap})
        if val_ap > best_score + min_delta:
            best_score, best_epoch, waited = val_ap, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_step": best_epoch, "steps_run": len(history), "pos_weight": pos_weight, "train_unit": "epoch"}


def fit_estimator(model: nn.Module, bundle) -> Dict[str, Any]:
    train, val = bundle.get_split("train"), bundle.get_split("val")
    model.fit(train.inputs, train.targets)
    train_nll = probability_diagnostics(train.targets.numpy(), model.predict_proba(train.inputs)[:, 1])["nll"]
    val_nll = probability_diagnostics(val.targets.numpy(), model.predict_proba(val.inputs)[:, 1])["nll"]
    return {
        "history": [{"step": 1, "train_loss": train_nll, "val_loss": val_nll}],
        "best_step": 1,
        "steps_run": 1,
        "pos_weight": None,
        "train_unit": "fit",
    }


# ---------------------------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------------------------
def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_history(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_history(path: Path, rows: List[Dict[str, float]], title: str) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        return "skipped: matplotlib is not installed (pip install 'pyhazards[plot]')"
    steps = [row["step"] for row in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, [row["train_loss"] for row in rows], marker="o", label="train loss")
    ax.plot(steps, [row["val_loss"] for row in rows], marker="s", label="validation loss")
    ax.set_xlabel("epoch")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _probability_maps(prob: np.ndarray, split) -> np.ndarray:
    days = len(split.metadata["dates"])
    return prob.reshape(days, -1)


# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------
def dataset_params(layout: str, args: argparse.Namespace) -> Dict[str, Any]:
    params: Dict[str, Any] = {"micro": True} if args.micro else {"cache_dir": args.cache_dir}
    params.update(
        {
            "train_limit_days": args.train_limit_days or None,
            "val_limit_days": args.val_limit_days or None,
            "test_limit_days": args.test_limit_days or None,
        }
    )
    if layout == "raster":
        params.update({"downsample_factor": args.raster_downsample, "spatial_multiple": 16})
    elif layout == "temporal":
        params.update({"downsample_factor": args.temporal_downsample, "history": args.temporal_history})
    else:
        params.update({"downsample_factor": args.tabular_downsample})
    return params


def run_one(model_name: str, layout: str, bundle, params: Dict[str, Any], seed: int, args, device, out: Path) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    kwargs = dict(MODEL_KWARGS.get(model_name, {}))
    if layout == "tabular":
        kwargs.setdefault("random_state", seed)
        kwargs.setdefault("n_jobs", args.n_jobs)
    else:
        kwargs["in_channels"] = int(bundle.feature_spec.channels)
    kwargs.update(args.model_kwargs.get(model_name, {}))
    task = "classification" if layout == "tabular" else "segmentation"
    model = build_model(name=model_name, task=task, **kwargs)

    started = time.time()
    if _is_estimator(model):
        fit = fit_estimator(model, bundle)
    else:
        fit = train_gradient_model(
            model,
            bundle,
            seed=seed,
            device=device,
            max_epochs=args.max_epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
        )
    seconds = time.time() - started

    seed_dir = out / model_name / f"seed_{seed}"
    config = ExperimentConfig(
        benchmark=BenchmarkConfig(
            name="wildfire",
            hazard_task="wildfire.danger",
            metrics=["accuracy", "macro_f1", "auc", "pr_auc"],
            eval_split="test",
            params={"batch_size": args.batch_size},
        ),
        dataset=DatasetRef(name=LAYOUT_DATASETS[layout], params=params),
        model=ModelRef(name=model_name, task=task, params=kwargs),
        report=ReportConfig(output_dir=str(seed_dir / "report"), formats=["json"]),
        seed=seed,
    )
    benchmark = run_benchmark("wildfire", model, bundle, config)

    scores: Dict[str, Dict[str, float]] = {}
    for split_name in ("val", "test"):
        split = bundle.get_split(split_name)
        prob = _positive_probability(model, split.inputs, args.batch_size, device)
        y_true = split.targets.numpy().reshape(-1)
        split_scores = probability_diagnostics(y_true, prob)
        split_scores["average_precision"] = float(average_precision_score(y_true > 0.5, prob.reshape(-1))) if (y_true > 0.5).any() else 0.0
        split_scores["mean_day_to_day_change"] = mean_day_to_day_change(_probability_maps(prob, split), split.metadata["dates"])
        scores[split_name] = split_scores
    test_metrics = {**benchmark.metrics, **{k: v for k, v in scores["test"].items() if k != "average_precision"}}

    _write_history(seed_dir / "history.csv", fit["history"])
    curve = (
        _plot_history(seed_dir / "loss_curve.png", fit["history"], f"{model_name} (seed {seed})")
        if fit["train_unit"] == "epoch"
        else "skipped: fitted in one step"
    )
    metrics = {
        "test": test_metrics,
        "val": scores["val"],
        "best_step": fit["best_step"],
        "steps_run": fit["steps_run"],
        "train_unit": fit["train_unit"],
        "loss_curve": curve,
    }
    _write_json(seed_dir / "metrics.json", metrics)
    _write_json(
        seed_dir / "experiment_setting.json",
        {
            "model": {"name": model_name, "task": task, "build_kwargs": kwargs, "layout": layout},
            "dataset": {"name": LAYOUT_DATASETS[layout], "params": params, "metadata": {k: v for k, v in bundle.metadata.items() if k not in ("lat", "lon")}},
            "training": {
                "seed": seed,
                "device": str(device),
                "max_epochs": args.max_epochs,
                "patience": args.patience,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "pos_weight": fit["pos_weight"],
                "seconds": seconds,
            },
            "software": {
                "pyhazards": pyhazards.__version__,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "python": platform.python_version(),
            },
        },
    )
    return test_metrics


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and score registry models on a Track-O cache.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cache-dir", help="Cache written by scripts/build_wildfire_track_o_cache.py.")
    source.add_argument("--micro", action="store_true", help="Use the synthetic micro cache (smoke runs).")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", default=DEFAULT_MODELS, help="Comma-separated registry model names.")
    parser.add_argument("--layout", action="append", default=[], help="name=raster|temporal|tabular for other models.")
    parser.add_argument("--model-kwargs", default="{}", help='JSON, e.g. {"random_forest": {"n_estimators": 500}}.')
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=8, help="Threads for tree models.")
    parser.add_argument("--raster-downsample", type=int, default=4)
    parser.add_argument("--temporal-downsample", type=int, default=8)
    parser.add_argument("--temporal-history", type=int, default=6)
    parser.add_argument("--tabular-downsample", type=int, default=8)
    parser.add_argument("--train-limit-days", type=int, default=0)
    parser.add_argument("--val-limit-days", type=int, default=0)
    parser.add_argument("--test-limit-days", type=int, default=0)
    args = parser.parse_args(argv)
    args.model_kwargs = json.loads(args.model_kwargs)
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    layouts = dict(MODEL_LAYOUTS)
    for item in args.layout:
        name, _, layout = item.partition("=")
        if layout not in LAYOUT_DATASETS:
            raise SystemExit(f"--layout expects name=raster|temporal|tabular, got {item!r}")
        layouts[name] = layout
    models = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = [name for name in models if name not in layouts]
    if unknown:
        raise SystemExit(f"no layout known for {unknown}; pass --layout name=raster|temporal|tabular")
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    device = torch.device(args.device) if args.device else auto_device()
    out = Path(args.output_dir)

    bundles: Dict[str, Any] = {}
    params_by_layout: Dict[str, Dict[str, Any]] = {}
    for layout in sorted({layouts[name] for name in models}):
        params_by_layout[layout] = dataset_params(layout, args)
        bundles[layout] = load_dataset(LAYOUT_DATASETS[layout], **params_by_layout[layout]).load()

    summary: Dict[str, Any] = {"source": "micro_synthetic" if args.micro else args.cache_dir, "seeds": seeds, "models": {}}
    for name in models:
        layout = layouts[name]
        per_seed = [
            run_one(name, layout, bundles[layout], params_by_layout[layout], seed, args, device, out) for seed in seeds
        ]
        keys = sorted(per_seed[0])
        summary["models"][name] = {
            "layout": layout,
            "per_seed": per_seed,
            "mean": {key: float(np.mean([row[key] for row in per_seed])) for key in keys},
            "std": {key: float(np.std([row[key] for row in per_seed])) for key in keys},
        }
        print(f"[done] {name}: " + ", ".join(f"{k}={summary['models'][name]['mean'][k]:.4f}" for k in keys))
    _write_json(out / "summary.json", summary)
    print(f"[done] results in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
