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
class SwinLSTMTrackOConfig:
    seq_len: int = 6
    in_channels: int = 1
    embed_dim: int = 16
    hidden_channels: int = 16
    num_heads: int = 4
    window_size: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 8
    max_epochs: int = 120
    early_stopping_rounds: int = 16
    min_delta: float = 1e-4
    seed: int = 42
    pos_weight_clip_max: float = 50.0
    device: str = "cpu"


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    # x: [B, H, W, C]
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = x.view(-1, window_size * window_size, c)
    return windows


def _window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int, b: int) -> torch.Tensor:
    # windows: [B*num_windows, window_size*window_size, C]
    c = windows.shape[-1]
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(b, h, w, c)
    return x


class WindowAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int, shift_size: int = 0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        ws = self.window_size

        x = x.permute(0, 2, 3, 1).contiguous()  # [B,H,W,C]
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows = _window_partition(x, ws)  # [B*nw, ws*ws, C]
        w_norm = self.norm1(windows)
        attn_out, _ = self.attn(w_norm, w_norm, w_norm, need_weights=False)
        windows = windows + attn_out
        windows = windows + self.mlp(self.norm2(windows))

        x = _window_reverse(windows, ws, h, w, b)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        return x.permute(0, 3, 1, 2).contiguous()  # [B,C,H,W]


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(input_channels + hidden_channels, hidden_channels * 4, kernel_size=3, padding=1)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fused = torch.cat([x_t, h_prev], dim=1)
        gates = self.conv(fused)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class TinySwinLSTM(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 16,
        hidden_channels: int = 16,
        num_heads: int = 4,
        window_size: int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_channels = hidden_channels

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=2, padding=1)
        self.block1 = WindowAttentionBlock(dim=embed_dim, num_heads=num_heads, window_size=window_size, shift_size=0)
        self.block2 = WindowAttentionBlock(
            dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=max(1, window_size // 2),
        )

        self.rnn = ConvLSTMCell(input_channels=embed_dim, hidden_channels=hidden_channels)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: [B,T,C,H,W]
        b, _, _, h, w = x_seq.shape
        h2, w2 = (h + 1) // 2, (w + 1) // 2
        device = x_seq.device

        h_state = torch.zeros((b, self.hidden_channels, h2, w2), device=device)
        c_state = torch.zeros((b, self.hidden_channels, h2, w2), device=device)

        for t in range(x_seq.shape[1]):
            x_t = self.patch_embed(x_seq[:, t])
            x_t = self.block1(x_t)
            x_t = self.block2(x_t)
            h_state, c_state = self.rnn(x_t, h_state, c_state)

        return self.decoder(h_state)


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


def train_swinlstm_track_o(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: SwinLSTMTrackOConfig,
):
    if x_train.ndim != 5 or x_val.ndim != 5:
        raise ValueError("x_train and x_val must be 5D arrays [N,T,C,H,W]")
    if y_train.ndim != 4 or y_val.ndim != 4:
        raise ValueError("y_train and y_val must be 4D arrays [N,1,H,W]")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = _choose_device(cfg.device)

    model = TinySwinLSTM(
        in_channels=cfg.in_channels,
        embed_dim=cfg.embed_dim,
        hidden_channels=cfg.hidden_channels,
        num_heads=cfg.num_heads,
        window_size=cfg.window_size,
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
    plt.title("SwinLSTM Track-O: train loss vs epoch")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()


def build_experiment_setting(
    cfg: SwinLSTMTrackOConfig,
    best_epoch: int,
    pos_weight: float,
    metrics: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "benchmark": {
            "task": "Track-O",
            "model_name": "swinlstm",
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

    cfg = SwinLSTMTrackOConfig(
        seed=seed,
        seq_len=seq_len,
        max_epochs=max_epochs,
        early_stopping_rounds=early_stopping_rounds,
        device="cpu",
    )

    model, history, val_metrics, best_epoch, pos_weight = train_swinlstm_track_o(
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
        output_dir / "swinlstm_model.pt",
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
    parser = argparse.ArgumentParser(description="Run SwinLSTM Track-O synthetic smoke demo")
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
    out = Path(args.output_dir) if args.output_dir else base / f"swinlstm_synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_synthetic_demo(
        output_dir=out,
        seed=args.seed,
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        image_size=args.image_size,
        max_epochs=args.max_epochs,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    print(f"[done] swinlstm synthetic demo saved to: {out}")


if __name__ == "__main__":
    main()

from ._wildfire_benchmark_utils import SegmentationPort, filter_init_kwargs, require_task


def swinlstm_builder(task: str, in_channels: int = 1, out_dim: int = 1, **kwargs: Any) -> nn.Module:
    require_task(task, {"segmentation"}, "swinlstm")
    init_kwargs = filter_init_kwargs(TinySwinLSTM, {"in_channels": int(in_channels), **kwargs})
    model = TinySwinLSTM(**init_kwargs)
    return SegmentationPort(model=model, out_channels=int(out_dim))


__all__ = ["TinySwinLSTM", "swinlstm_builder"]
