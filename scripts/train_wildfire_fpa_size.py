#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

def _to_float_tensor(x):
    """Convert numpy array OR torch tensor -> torch.float32 tensor."""
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float()
    return torch.from_numpy(x).float()

from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
except Exception:
    import pytorch_lightning as pl  # type: ignore
    from pytorch_lightning.callbacks import ModelCheckpoint  # type: ignore
    from pytorch_lightning.loggers import CSVLogger  # type: ignore


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _write_text(path: str, s: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)


def _safe_accelerator(requested: str) -> str:
    req = (requested or "auto").lower()
    if req == "cpu":
        return "cpu"
    if req == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req == "mps":
        ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return "mps" if ok else "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _safe_devices(requested: str) -> Any:
    req = (requested or "auto").lower()
    if req == "auto":
        return 1
    try:
        return int(req)
    except Exception:
        return req


def _confusion_update(cm: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1


def _metrics_from_cm(cm: np.ndarray) -> Dict[str, Any]:
    eps = 1e-12
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    pred_support = cm.sum(axis=0).astype(np.float64)

    precision = tp / (pred_support + eps)
    recall = tp / (support + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        "accuracy": float(tp.sum() / (cm.sum() + eps)),
        "macro_precision": float(np.nanmean(precision)),
        "macro_recall": float(np.nanmean(recall)),
        "macro_f1": float(np.nanmean(f1)),
        "per_class": [
            {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(cm.shape[0])
        ],
    }


def _plot_cm(cm: np.ndarray, classes: List[str], out_png: str, title: str) -> None:
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    thresh = cm.max() * 0.5 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(int(cm[i, j])),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=7
            )
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _compute_balanced_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = (counts.sum() / (num_classes * counts)).astype(np.float32)
    return torch.from_numpy(w)


def _make_loaders(
    x_train: np.ndarray, y_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray,
    batch_size: int, num_workers: int,
    imbalance: str,
    num_classes: int,
    seed: int,
):
    tx_train = _to_float_tensor(x_train)
    ty_train = (_to_float_tensor(y_train).long())
    tx_val = _to_float_tensor(x_val)
    ty_val = (_to_float_tensor(y_val).long())
    tx_test = _to_float_tensor(x_test)
    ty_test = (_to_float_tensor(y_test).long())

    train_ds = TensorDataset(tx_train, ty_train)
    val_ds = TensorDataset(tx_val, ty_val)
    test_ds = TensorDataset(tx_test, ty_test)

    persistent_workers = bool(num_workers and num_workers > 0)

    class_weights = None
    sampler = None

    if imbalance == "weighted_sampler":
        counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
        counts[counts == 0] = 1.0
        inv = 1.0 / counts
        w_per_sample = inv[y_train]
        gen = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(w_per_sample).double(),
            num_samples=len(w_per_sample),
            replacement=True,
            generator=gen,
        )
        class_weights = torch.from_numpy((inv / inv.mean()).astype(np.float32))
    elif imbalance == "class_weight_balanced":
        class_weights = _compute_balanced_class_weights(y_train, num_classes)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=persistent_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=persistent_workers)
    return train_loader, val_loader, test_loader, class_weights


def _majority_baseline_metrics(y_train: np.ndarray, y_test: np.ndarray, num_classes: int) -> Dict[str, Any]:
    counts = np.bincount(y_train, minlength=num_classes)
    maj = int(np.argmax(counts))
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    y_pred = np.full_like(y_test, fill_value=maj)
    _confusion_update(cm, y_test, y_pred)
    return {"majority_class": maj, "train_counts": counts.tolist(), "metrics": _metrics_from_cm(cm)}


# -------------------------
# strict builder (double bulletproof)
# -------------------------
def _get_builder(model_name: str):
    from pyhazards.models.registry import get_model_config  # type: ignore
    cfg = get_model_config(model_name)
    if not cfg:
        raise ValueError(f"Model '{model_name}' not found in registry.")
    return cfg["builder"]


def _parse_unexpected_kwargs(msg: str) -> List[str]:
    m = re.search(r"Unexpected kwargs.*?:\s*(\[[^\]]*\])", msg)
    if m:
        try:
            xs = ast.literal_eval(m.group(1))
            return [str(x) for x in xs]
        except Exception:
            return []
    m2 = re.search(r"unexpected keyword argument\s+'([^']+)'", msg)
    if m2:
        return [m2.group(1)]
    return []


def _build_with_stripping(builder, kwargs: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
    cur = dict(kwargs)
    for _ in range(20):
        try:
            model = builder(**cur)
            return model, cur
        except TypeError as e:
            msg = str(e)
            bad = _parse_unexpected_kwargs(msg)
            if bad:
                for k in bad:
                    cur.pop(k, None)
                continue
            raise
    raise RuntimeError(f"Builder could not be satisfied after stripping retries. Last kwargs={cur}")


def build_wildfire_fpa_mlp_strict(
    model_name: str,
    task: str,
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    dropout: float,
    depth: int,
) -> Tuple[nn.Module, Dict[str, Any]]:
    builder = _get_builder(model_name)
    base = {
        "task": task,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "depth": depth,
        "num_layers": depth,
        "layers": depth,
    }
    model, used = _build_with_stripping(builder, base)
    return model, dict(used)


# -------------------------
# losses
# -------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = torch.log_softmax(logits, dim=1)
        p = torch.exp(logp)
        tgt = target.long()
        logp_t = logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
        p_t = p.gather(1, tgt.unsqueeze(1)).squeeze(1)
        loss = -(1.0 - p_t) ** self.gamma * logp_t
        if self.weight is not None:
            w = self.weight.to(loss.device).gather(0, tgt)
            loss = loss * w
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


# -------------------------
# lightning module
# -------------------------
class LitClassifier(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        lr: float,
        weight_decay: float,
        loss_name: str,
        focal_gamma: float,
        class_weights: Optional[torch.Tensor],
        label_smoothing: float,
    ):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay

        self._val_cm: Optional[np.ndarray] = None
        self._test_cm: Optional[np.ndarray] = None

        if loss_name == "focal":
            self.criterion = FocalLoss(gamma=focal_gamma, weight=class_weights)
        else:
            kwargs = {}
            if class_weights is not None:
                kwargs["weight"] = class_weights
            try:
                kwargs["label_smoothing"] = float(label_smoothing)
                self.criterion = nn.CrossEntropyLoss(**kwargs)
            except TypeError:
                kwargs.pop("label_smoothing", None)
                self.criterion = nn.CrossEntropyLoss(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def training_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self._val_cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)

        assert self._val_cm is not None
        _confusion_update(self._val_cm, y.detach().cpu().numpy(), preds.detach().cpu().numpy())

    def on_validation_epoch_end(self):
        if self._val_cm is None:
            return
        m = _metrics_from_cm(self._val_cm)
        self.log("val/macro_f1", float(m["macro_f1"]), prog_bar=True)

    def on_test_epoch_start(self):
        self._test_cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def test_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        assert self._test_cm is not None
        _confusion_update(self._test_cm, y.detach().cpu().numpy(), preds.detach().cpu().numpy())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--model_name", default="wildfire_fpa_mlp")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--region", choices=["US", "CA"], default="US")
    ap.add_argument("--normalize", action="store_true")

    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--num_layers", type=int, default=None, help="Alias for --depth")
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--imbalance", choices=["none", "class_weight_balanced", "weighted_sampler"], default="class_weight_balanced")

    ap.add_argument("--loss", choices=["ce", "focal"], default="ce")
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--label_smoothing", type=float, default=0.0)

    ap.add_argument("--limit_train_batches", type=float, default=1.0)
    ap.add_argument("--limit_val_batches", type=float, default=1.0)
    ap.add_argument("--limit_test_batches", type=float, default=1.0)
    ap.add_argument("--log_every_n_steps", type=int, default=50)

    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--devices", default="auto")
    ap.add_argument("--precision", default="32")

    ap.add_argument("--out_root", default="outputs/train_wildfire_fpa_size")
    ap.add_argument("--resume_from", default=None)

    args = ap.parse_args()
    if args.num_layers is not None:
        args.depth = int(args.num_layers)

    run_dir = os.path.join(args.out_root, _now_tag())
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    _mkdir(ckpt_dir)
    _write_json(os.path.join(run_dir, "config.json"), vars(args))
    _write_text(os.path.join(run_dir, "command.txt"), " ".join(sys.argv) + "\n")

    try:
        pl.seed_everything(args.seed, workers=True)

        from pyhazards.datasets import FPAFODWildfireTabular  # type: ignore

        b = FPAFODWildfireTabular(
            task="size",
            region=args.region,
            data_path=args.data_path,
            micro=False,
            normalize=args.normalize,
            seed=args.seed,
        ).load()

        x_train, y_train = b.splits["train"].inputs, b.splits["train"].targets
        x_val, y_val = b.splits["val"].inputs, b.splits["val"].targets
        x_test, y_test = b.splits["test"].inputs, b.splits["test"].targets

        cls = b.label_spec.extra.get("classes")
        if cls is None:
            # fallback: numeric class ids if names not provided
            n = b.label_spec.num_targets
            if n is None: raise AttributeError("LabelSpec missing extra['classes'] and num_targets")
            cls = list(range(int(n)))
        classes = list(cls)
        num_classes = int(b.label_spec.num_targets)
        in_dim = int(x_train.shape[1])

        train_loader, val_loader, test_loader, class_weights = _make_loaders(
            x_train, y_train, x_val, y_val, x_test, y_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            imbalance=args.imbalance,
            num_classes=num_classes,
            seed=args.seed,
        )

        model, used_kwargs = build_wildfire_fpa_mlp_strict(
            model_name=args.model_name,
            task="classification",
            in_dim=in_dim,
            out_dim=num_classes,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            depth=args.depth,
        )
        _write_json(os.path.join(run_dir, "model_kwargs_used.json"), used_kwargs)

        lit = LitClassifier(
            model=model,
            num_classes=num_classes,
            lr=args.lr,
            weight_decay=args.weight_decay,
            loss_name=args.loss,
            focal_gamma=args.focal_gamma,
            class_weights=class_weights,
            label_smoothing=args.label_smoothing,
        )

        ckpt = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best",
            monitor="val/macro_f1",
            mode="max",
            save_top_k=1,
            save_last=True,
        )
        logger = CSVLogger(save_dir=run_dir, name="csv")

        accel = _safe_accelerator(args.accelerator)
        devs = _safe_devices(args.devices)

        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator=_safe_accelerator(args.accelerator),
            devices=devs,
            precision=args.precision,
            logger=logger,
            callbacks=[ckpt],
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
            limit_test_batches=args.limit_test_batches,
            log_every_n_steps=args.log_every_n_steps,
        )

        trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.resume_from)
        trainer.test(lit, dataloaders=test_loader, ckpt_path=ckpt.best_model_path)

        cm = lit._test_cm if lit._test_cm is not None else np.zeros((num_classes, num_classes), dtype=np.int64)
        m = _metrics_from_cm(cm)

        _write_json(os.path.join(run_dir, "confusion_matrix.json"), {"cm": cm.tolist(), "classes": classes})
        _plot_cm(cm, classes, os.path.join(run_dir, "confusion_matrix.png"), "Size confusion matrix")

        torch.save(lit.model.state_dict(), os.path.join(run_dir, "model_final.pt"))

        baseline = _majority_baseline_metrics(y_train, y_test, num_classes)

        metrics = {
            "task": "fpa_fod_size",
            "config": vars(args),
            "git_sha": _git_sha(),
            "system": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__},
            "best_ckpt": ckpt.best_model_path,
            "last_ckpt": ckpt.last_model_path,
            "test": {"metrics": m},
            "baseline_majority_test": baseline,
        }
        _write_json(os.path.join(run_dir, "metrics.json"), metrics)

        summary = [
            "# Paper reproduction summary — FPA-FOD Size",
            "",
            f"## Model kwargs actually used (strict builder)\n- {used_kwargs}",
            "",
            "## Your test metrics",
            f"- accuracy: {m['accuracy']:.4f}",
            f"- macro_f1: {m['macro_f1']:.4f}",
            f"- macro_precision: {m['macro_precision']:.4f}",
            f"- macro_recall: {m['macro_recall']:.4f}",
            "",
            "## Majority baseline (test)",
            f"- accuracy: {baseline['metrics']['accuracy']:.4f}",
            f"- macro_f1: {baseline['metrics']['macro_f1']:.4f}",
        ]
        _write_text(os.path.join(run_dir, "paper_summary.md"), "\n".join(summary) + "\n")

        print(f"[OK] Outputs saved to: {run_dir}")

    except BaseException as e:
        _write_text(os.path.join(run_dir, "FAILURE.txt"), f"{type(e).__name__}: {e}\n")
        raise


if __name__ == "__main__":
    main()
