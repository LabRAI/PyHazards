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
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from pyhazards.models.mau import (
        binary_ece,
        make_synthetic_fire_sequences,
        normalized_consistency_score,
        split_train_val_test,
    )
else:
    from .mau import (
        binary_ece,
        make_synthetic_fire_sequences,
        normalized_consistency_score,
        split_train_val_test,
    )


@dataclass
class ConvGRTrajGRUTrackOConfig:
    seq_len: int = 6
    in_channels: int = 1
    enc_channels: int = 16
    hidden_channels: int = 16
    kernel_size: int = 3
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 8
    max_epochs: int = 120
    early_stopping_rounds: int = 16
    min_delta: float = 1e-4
    seed: int = 42
    pos_weight_clip_max: float = 50.0
    device: str = "cpu"


def _normalized_base_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1)  # [H,W,2]


def _warp_hidden(hidden: torch.Tensor, flow_xy: torch.Tensor) -> torch.Tensor:
    # hidden: [B,C,H,W], flow_xy: [B,2,H,W] in pixel space
    b, _, h, w = hidden.shape
    base = _normalized_base_grid(h, w, hidden.device, hidden.dtype).unsqueeze(0).repeat(b, 1, 1, 1)

    fx = flow_xy[:, 0] / max((w - 1) / 2.0, 1.0)
    fy = flow_xy[:, 1] / max((h - 1) / 2.0, 1.0)
    flow = torch.stack((fx, fy), dim=-1)  # [B,H,W,2]

    grid = base + flow
    return F.grid_sample(hidden, grid, mode="bilinear", padding_mode="border", align_corners=True)


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv_zr = nn.Conv2d(input_channels + hidden_channels, hidden_channels * 2, kernel_size, padding=padding)
        self.conv_n = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([x_t, h_prev], dim=1)
        z, r = torch.chunk(self.conv_zr(fused), 2, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)

        n = torch.tanh(self.conv_n(torch.cat([x_t, r * h_prev], dim=1)))
        return (1.0 - z) * h_prev + z * n


class TrajGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.flow_net = nn.Conv2d(input_channels + hidden_channels, 2, kernel_size=3, padding=1)
        self.conv_zr = nn.Conv2d(input_channels + hidden_channels, hidden_channels * 2, kernel_size, padding=padding)
        self.conv_n = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        flow = self.flow_net(torch.cat([x_t, h_prev], dim=1))
        h_warp = _warp_hidden(h_prev, flow)

        fused = torch.cat([x_t, h_warp], dim=1)
        z, r = torch.chunk(self.conv_zr(fused), 2, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)

        n = torch.tanh(self.conv_n(torch.cat([x_t, r * h_warp], dim=1)))
        return (1.0 - z) * h_warp + z * n


class TinyConvGRTrajGRU(nn.Module):
    def __init__(self, in_channels: int = 1, enc_channels: int = 16, hidden_channels: int = 16, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, enc_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(enc_channels, enc_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.convgru = ConvGRUCell(input_channels=enc_channels, hidden_channels=hidden_channels, kernel_size=kernel_size)
        self.trajgru = TrajGRUCell(input_channels=hidden_channels, hidden_channels=hidden_channels, kernel_size=kernel_size)

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: [B,T,C,H,W]
        b, _, _, h, w = x_seq.shape
        device = x_seq.device
        dtype = x_seq.dtype

        h_conv = torch.zeros((b, self.hidden_channels, h, w), device=device, dtype=dtype)
        h_traj = torch.zeros((b, self.hidden_channels, h, w), device=device, dtype=dtype)

        for t in range(x_seq.shape[1]):
            x_t = self.encoder(x_seq[:, t])
            h_conv = self.convgru(x_t, h_conv)
            h_traj = self.trajgru(h_conv, h_traj)

        return self.decoder(h_traj)


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


def train_convgru_trajgru_track_o(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: ConvGRTrajGRUTrackOConfig,
):
    if x_train.ndim != 5 or x_val.ndim != 5:
        raise ValueError("x_train and x_val must be 5D arrays [N,T,C,H,W]")
    if y_train.ndim != 4 or y_val.ndim != 4:
        raise ValueError("y_train and y_val must be 4D arrays [N,1,H,W]")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = _choose_device(cfg.device)

    model = TinyConvGRTrajGRU(
        in_channels=cfg.in_channels,
        enc_channels=cfg.enc_channels,
        hidden_channels=cfg.hidden_channels,
        kernel_size=cfg.kernel_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_px = float(y_train.size)
    pos_px = float(np.sum(y_train))
    neg_px = max(1.0, total_px - pos_px)
    raw_pos_weight = neg_px / max(pos_px, 1.0)
    pos_weight = float(np.clip(raw_pos_weight, 1.0, cfg.pos_weight_clip_max))

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    train_loader = _build_loader(x_train, y_train, batch_size=cfg.batch_size, shuffle=True)
    val_loader = _build_loader(x_val, y_val, batch_size=cfg.batch_size, shuffle=False)

    history: List[Dict[str, float]] = []
    best_epoch = 1
    best_val_loss = float("inf")
    best_state: Dict[str, torch.Tensor] | None = None
    wait = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        train_losses: List[float] = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(float(loss.item()))

        tr_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        va_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": tr_loss,
                "val_loss": va_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if va_loss < best_val_loss - cfg.min_delta:
            best_val_loss = va_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= cfg.early_stopping_rounds:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_prob = np.clip(_predict_probabilities(model, val_loader, device=device), 1e-7, 1.0 - 1e-7)
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
    }

    return model, history, metrics, best_epoch, pos_weight


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
    plt.title("ConvGRU/TrajGRU Track-O: train loss vs epoch")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()


def build_experiment_setting(
    cfg: ConvGRTrajGRUTrackOConfig,
    best_epoch: int,
    pos_weight: float,
    metrics: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "benchmark": {
            "task": "Track-O",
            "model_name": "convgru_trajgru",
            "run_time": datetime.now().isoformat(),
        },
        "evaluation_protocol": {
            "discrimination": {"primary": "auprc", "secondary": "auroc"},
            "reliability": ["brier", "nll", "ece"],
            "temporal_consistency": ["mean_day_to_day_change", "normalized_consistency_score"],
        },
        "training": {
            "train_unit": "epoch",
            "max_epochs": cfg.max_epochs,
            "early_stopping_rounds": cfg.early_stopping_rounds,
            "best_epoch": best_epoch,
            "seed": cfg.seed,
            "batch_size": cfg.batch_size,
            "seq_len": cfg.seq_len,
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
    n_samples: int = 160,
    seq_len: int = 6,
    image_size: int = 24,
    max_epochs: int = 60,
    early_stopping_rounds: int = 12,
) -> None:
    x, y = make_synthetic_fire_sequences(
        n_samples=n_samples,
        seq_len=seq_len,
        image_size=image_size,
        seed=seed,
    )
    x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(x, y, seed=seed)

    cfg = ConvGRTrajGRUTrackOConfig(
        seed=seed,
        seq_len=seq_len,
        max_epochs=max_epochs,
        early_stopping_rounds=early_stopping_rounds,
        device="cpu",
    )

    model, history, val_metrics, best_epoch, pos_weight = train_convgru_trajgru_track_o(
        x_train,
        y_train,
        x_val,
        y_val,
        cfg,
    )

    test_loader = _build_loader(x_test, y_test, batch_size=cfg.batch_size, shuffle=False)
    test_prob = np.clip(_predict_probabilities(model, test_loader, _choose_device(cfg.device)), 1e-7, 1.0 - 1e-7)
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
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    save_history_and_plot(history, output_dir)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(cfg),
            "best_epoch": best_epoch,
        },
        output_dir / "convgru_trajgru_model.pt",
    )

    setting = build_experiment_setting(cfg, best_epoch=best_epoch, pos_weight=pos_weight, metrics=val_metrics)
    setting["data"] = {
        "n_samples": n_samples,
        "image_size": image_size,
        "seq_len": seq_len,
        "split": {"train": int(x_train.shape[0]), "val": int(x_val.shape[0]), "test": int(x_test.shape[0])},
    }
    setting["test_metrics"] = test_metrics

    (output_dir / "experiment_setting.json").write_text(json.dumps(setting, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(
        json.dumps({"val": val_metrics, "test": test_metrics, "best_epoch": best_epoch}, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ConvGRU/TrajGRU Track-O synthetic smoke demo")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=160)
    parser.add_argument("--seq_len", type=int, default=6)
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
        else base / f"convgru_trajgru_synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    run_synthetic_demo(
        output_dir=out,
        seed=args.seed,
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        image_size=args.image_size,
        max_epochs=args.max_epochs,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    print(f"[done] convgru_trajgru synthetic demo saved to: {out}")


if __name__ == "__main__":
    main()

from ._wildfire_benchmark_utils import SegmentationPort, filter_init_kwargs, require_task


def convgru_trajgru_builder(task: str, in_channels: int = 1, out_dim: int = 1, **kwargs: Any) -> nn.Module:
    require_task(task, {"segmentation"}, "convgru_trajgru")
    init_kwargs = filter_init_kwargs(TinyConvGRTrajGRU, {"in_channels": int(in_channels), **kwargs})
    model = TinyConvGRTrajGRU(**init_kwargs)
    return SegmentationPort(model=model, out_channels=int(out_dim))


__all__ = ["TinyConvGRTrajGRU", "convgru_trajgru_builder"]
