from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from pyhazards.models.unet import (
        binary_ece,
        make_synthetic_fire_maps,
        normalized_consistency_score,
        split_train_val_test,
    )
else:
    from .unet import (
        binary_ece,
        make_synthetic_fire_maps,
        normalized_consistency_score,
        split_train_val_test,
    )


@dataclass
class DeepEnsembleTrackOConfig:
    in_channels: int = 1
    base_channels: int = 8
    ensemble_size: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 8
    max_epochs: int = 120
    early_stopping_rounds: int = 16
    min_delta: float = 1e-4
    seed: int = 42
    pos_weight_clip_max: float = 50.0
    device: str = "cpu"


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyEnsembleMember(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 8):
        super().__init__()
        c1, c2 = base_channels, base_channels * 2
        self.enc1 = ConvBlock(in_channels, c1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.enc2 = ConvBlock(c1, c2)
        self.up = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec = ConvBlock(c1 + c1, c1)
        self.head = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        y = self.up(x2)
        y = torch.cat([y, x1], dim=1)
        y = self.dec(y)
        return self.head(y)


def _choose_device(device_text: str) -> torch.device:
    if device_text == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _predict_probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    probs: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            p = torch.sigmoid(logits).detach().cpu().numpy()
            probs.append(p)
    if not probs:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(probs, axis=0).reshape(-1)


def _predict_ensemble_probabilities(models: List[nn.Module], loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, float]:
    member_probs: List[np.ndarray] = []
    for model in models:
        member_probs.append(_predict_probabilities(model, loader, device))
    if not member_probs:
        return np.zeros((0,), dtype=np.float32), 0.0
    stacked = np.stack(member_probs, axis=0)  # [M,N]
    return np.mean(stacked, axis=0), float(np.mean(np.std(stacked, axis=0)))


def _train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    model.train()
    losses: List[float] = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def _eval_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def train_deep_ensemble_track_o(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: DeepEnsembleTrackOConfig,
):
    if x_train.ndim != 4 or x_val.ndim != 4:
        raise ValueError("x_train and x_val must be 4D arrays [N,C,H,W]")
    if y_train.ndim != 4 or y_val.ndim != 4:
        raise ValueError("y_train and y_val must be 4D arrays [N,1,H,W]")
    if cfg.ensemble_size < 1:
        raise ValueError("ensemble_size must be >= 1")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = _choose_device(cfg.device)

    total_px = float(y_train.size)
    pos_px = float(np.sum(y_train))
    neg_px = max(1.0, total_px - pos_px)
    raw_pos_weight = neg_px / max(pos_px, 1.0)
    pos_weight = float(np.clip(raw_pos_weight, 1.0, cfg.pos_weight_clip_max))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    train_loader = _build_loader(x_train, y_train, batch_size=cfg.batch_size, shuffle=True)
    val_loader = _build_loader(x_val, y_val, batch_size=cfg.batch_size, shuffle=False)

    models: List[nn.Module] = []
    optimizers: List[torch.optim.Optimizer] = []
    for m in range(cfg.ensemble_size):
        member_seed = cfg.seed + 1000 + m
        torch.manual_seed(member_seed)
        np.random.seed(member_seed)
        member = TinyEnsembleMember(in_channels=cfg.in_channels, base_channels=cfg.base_channels).to(device)
        opt = torch.optim.AdamW(member.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        models.append(member)
        optimizers.append(opt)

    history: List[Dict[str, float]] = []
    best_epoch = 1
    best_val_loss = float("inf")
    best_states: List[Dict[str, torch.Tensor]] | None = None
    wait = 0

    for epoch in range(1, cfg.max_epochs + 1):
        train_member_losses: List[float] = []
        val_member_losses: List[float] = []

        for model, opt in zip(models, optimizers):
            tr = _train_one_epoch(model, train_loader, opt, criterion, device)
            va = _eval_loss(model, val_loader, criterion, device)
            train_member_losses.append(tr)
            val_member_losses.append(va)

        tr_loss = float(np.mean(train_member_losses)) if train_member_losses else float("nan")
        va_loss = float(np.mean(val_member_losses)) if val_member_losses else float("nan")

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": tr_loss,
                "val_loss": va_loss,
                "learning_rate": float(np.mean([opt.param_groups[0]["lr"] for opt in optimizers])),
            }
        )

        if va_loss < best_val_loss - cfg.min_delta:
            best_val_loss = va_loss
            best_epoch = epoch
            best_states = [deepcopy(model.state_dict()) for model in models]
            wait = 0
        else:
            wait += 1

        if wait >= cfg.early_stopping_rounds:
            break

    if best_states is not None:
        for model, state in zip(models, best_states):
            model.load_state_dict(state)

    val_prob, mean_ensemble_std = _predict_ensemble_probabilities(models, val_loader, device=device)
    val_prob = np.clip(val_prob, 1e-7, 1.0 - 1e-7)
    val_true = y_val.reshape(-1).astype(np.float32)

    mean_change = float(np.mean(np.abs(np.diff(np.sort(val_prob))))) if len(val_prob) > 1 else 0.0
    metrics = {
        "auprc": float(average_precision_score(val_true, val_prob)),
        "auroc": float(roc_auc_score(val_true, val_prob)),
        "brier": float(brier_score_loss(val_true, val_prob)),
        "nll": float(log_loss(val_true, val_prob)),
        "ece": float(binary_ece(val_true, val_prob, n_bins=15)),
        "mean_day_to_day_change": mean_change,
        "normalized_consistency_score": normalized_consistency_score(mean_change),
        "mean_ensemble_std": float(mean_ensemble_std),
    }

    return models, history, metrics, best_epoch, pos_weight


def save_history_and_plot(history: List[Dict[str, float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    history_csv = output_dir / "history.csv"
    with history_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "learning_rate"])
        writer.writeheader()
        writer.writerows(history)

    x = [int(r["epoch"]) for r in history]
    y_tr = [float(r["train_loss"]) for r in history]
    y_va = [float(r["val_loss"]) for r in history]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y_tr, label="train_bce", marker="o", linewidth=1.4)
    plt.plot(x, y_va, label="val_bce", marker="s", linewidth=1.2)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Deep Ensemble Track-O: train loss vs epoch")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()


def build_experiment_setting(
    cfg: DeepEnsembleTrackOConfig,
    best_epoch: int,
    pos_weight: float,
    metrics: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "benchmark": {
            "task": "Track-O",
            "model_name": "deep_ensemble",
            "run_time": datetime.now().isoformat(),
        },
        "evaluation_protocol": {
            "discrimination": {"primary": "auprc", "secondary": "auroc"},
            "reliability": ["brier", "nll", "ece"],
            "temporal_consistency": ["mean_day_to_day_change", "normalized_consistency_score"],
            "uncertainty": ["mean_ensemble_std"],
        },
        "training": {
            "train_unit": "epoch",
            "max_epochs": cfg.max_epochs,
            "early_stopping_rounds": cfg.early_stopping_rounds,
            "best_epoch": best_epoch,
            "seed": cfg.seed,
            "batch_size": cfg.batch_size,
            "ensemble_size": cfg.ensemble_size,
        },
        "optimizer": {
            "name": "AdamW",
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
        },
        "learning_weight": {
            "type": "pixel_pos_weight",
            "value": pos_weight,
            "clip_max": cfg.pos_weight_clip_max,
        },
        "params": asdict(cfg),
        "val_metrics": metrics,
        "note": "This module supports both real data and synthetic smoke demonstration.",
    }


def run_synthetic_demo(
    output_dir: Path,
    seed: int = 42,
    n_samples: int = 192,
    image_size: int = 24,
    max_epochs: int = 60,
    early_stopping_rounds: int = 12,
) -> None:
    x, y = make_synthetic_fire_maps(n_samples=n_samples, image_size=image_size, seed=seed)
    x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(x, y, seed=seed)

    cfg = DeepEnsembleTrackOConfig(
        seed=seed,
        max_epochs=max_epochs,
        early_stopping_rounds=early_stopping_rounds,
        device="cpu",
    )

    models, history, val_metrics, best_epoch, pos_weight = train_deep_ensemble_track_o(
        x_train,
        y_train,
        x_val,
        y_val,
        cfg,
    )

    test_loader = _build_loader(x_test, y_test, batch_size=cfg.batch_size, shuffle=False)
    test_prob, test_std = _predict_ensemble_probabilities(models, test_loader, _choose_device(cfg.device))
    test_prob = np.clip(test_prob, 1e-7, 1.0 - 1e-7)
    test_true = y_test.reshape(-1).astype(np.float32)

    test_mean_change = float(np.mean(np.abs(np.diff(np.sort(test_prob))))) if len(test_prob) > 1 else 0.0
    test_metrics = {
        "auprc": float(average_precision_score(test_true, test_prob)),
        "auroc": float(roc_auc_score(test_true, test_prob)),
        "brier": float(brier_score_loss(test_true, test_prob)),
        "nll": float(log_loss(test_true, test_prob)),
        "ece": float(binary_ece(test_true, test_prob, n_bins=15)),
        "mean_day_to_day_change": test_mean_change,
        "normalized_consistency_score": normalized_consistency_score(test_mean_change),
        "mean_ensemble_std": float(test_std),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    save_history_and_plot(history, output_dir)

    torch.save(
        {
            "members": [model.state_dict() for model in models],
            "config": asdict(cfg),
            "best_epoch": best_epoch,
        },
        output_dir / "deep_ensemble_model.pt",
    )

    setting = build_experiment_setting(cfg, best_epoch=best_epoch, pos_weight=pos_weight, metrics=val_metrics)
    setting["data"] = {
        "n_samples": n_samples,
        "image_size": image_size,
        "split": {"train": int(x_train.shape[0]), "val": int(x_val.shape[0]), "test": int(x_test.shape[0])},
    }
    setting["test_metrics"] = test_metrics

    (output_dir / "experiment_setting.json").write_text(json.dumps(setting, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(
        json.dumps({"val": val_metrics, "test": test_metrics, "best_epoch": best_epoch}, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Deep Ensemble Track-O synthetic smoke demo")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=192)
    parser.add_argument("--image_size", type=int, default=24)
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--early_stopping_rounds", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(__file__).resolve().parents[1] / "runs_scaffold"
    out = (
        Path(args.output_dir)
        if args.output_dir
        else base / f"deep_ensemble_synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    run_synthetic_demo(
        output_dir=out,
        seed=args.seed,
        n_samples=args.n_samples,
        image_size=args.image_size,
        max_epochs=args.max_epochs,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    print(f"[done] deep_ensemble synthetic demo saved to: {out}")


if __name__ == "__main__":
    main()

from ._wildfire_benchmark_utils import SegmentationPort, require_task


class DeepEnsemble(nn.Module):
    """Benchmark-facing deep ensemble that averages member logits."""

    def __init__(self, in_channels: int = 1, base_channels: int = 8, ensemble_size: int = 5):
        super().__init__()
        if ensemble_size < 1:
            raise ValueError('ensemble_size must be >= 1')
        self.members = nn.ModuleList(
            [TinyEnsembleMember(in_channels=in_channels, base_channels=base_channels) for _ in range(int(ensemble_size))]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.stack([member(x) for member in self.members], dim=0).mean(dim=0)
        return logits


def deep_ensemble_builder(
    task: str,
    in_channels: int = 1,
    out_dim: int = 1,
    base_channels: int = 8,
    ensemble_size: int = 5,
    **kwargs: Any,
) -> nn.Module:
    _ = kwargs
    require_task(task, {"segmentation"}, "deep_ensemble")
    model = DeepEnsemble(in_channels=int(in_channels), base_channels=int(base_channels), ensemble_size=int(ensemble_size))
    return SegmentationPort(model=model, out_channels=int(out_dim))


__all__ = ["DeepEnsemble", "TinyEnsembleMember", "deep_ensemble_builder"]
