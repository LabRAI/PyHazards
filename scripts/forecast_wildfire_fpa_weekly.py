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
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float()
    # assume numpy-like
    return torch.from_numpy(x).float()

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
except Exception:
    import pytorch_lightning as pl  # type: ignore
    from pytorch_lightning.callbacks import ModelCheckpoint  # type: ignore
    from pytorch_lightning.loggers import CSVLogger  # type: ignore


SIZE_GROUPS = ["A", "B", "C", "D", "EFG"]


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


# -------------------------
# builder (double bulletproof)
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
    for _ in range(25):
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


def build_wildfire_fpa_lstm_strict(
    model_name: str,
    task: str,
    input_dim: int,
    output_dim: int,
    lookback: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
) -> Tuple[nn.Module, Dict[str, Any]]:
    builder = _get_builder(model_name)
    base = {
        "task": task,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "lookback": lookback,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        # synonyms (auto-stripped if rejected)
        "in_dim": input_dim,
        "out_dim": output_dim,
        "layers": num_layers,
        "seq_len": lookback,
    }
    model, used = _build_with_stripping(builder, base)
    return model, dict(used)


# -------------------------
# metrics / plotting
# -------------------------
def _to_numpy(x):
    import numpy as np
    import torch

    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _mae_rmse(y_true, y_pred):
    yt = _to_numpy(y_true).astype(np.float32)
    yp = _to_numpy(y_pred).astype(np.float32)
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {"mae": mae, "rmse": rmse}



def _plot_series(y_true: np.ndarray, y_pred: np.ndarray, out_png: str, title: str, groups: List[str]) -> None:
    t = np.arange(len(y_true))
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(t, y_true.sum(axis=1), label="true_total")
    ax.plot(t, y_pred.sum(axis=1), label="pred_total")
    ax.set_title(title)
    ax.set_xlabel("Time index (test samples)")
    ax.set_ylabel("Total next-week fires")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    for i, g in enumerate(groups):
        ax.plot(t, y_true[:, i], label=f"true_{g}")
        ax.plot(t, y_pred[:, i], label=f"pred_{g}", linestyle="--")
    ax.set_title(title + " (per group)")
    ax.set_xlabel("Time index (test samples)")
    ax.set_ylabel("Next-week count")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png.replace(".png", "_per_group.png"), dpi=200)
    plt.close(fig)


def _resolve_lookback(args) -> int:
    if args.lookback is not None and args.lookback_weeks is not None:
        raise SystemExit("Provide only one: --lookback or --lookback_weeks")
    if args.lookback is None and args.lookback_weeks is None:
        return 50
    return int(args.lookback if args.lookback is not None else args.lookback_weeks)


def _counts_slice_for_features(x: np.ndarray, out_dim: int) -> np.ndarray:
    # For counts or counts+time, assume counts are the first out_dim features.
    # This matches your dataset behavior in practice.
    return x[..., :out_dim]


# -------------------------
# Lightning module with invertible transforms
# -------------------------
class LitWeekly(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        weight_decay: float,
        loss_name: str,
        nonneg: bool,
        log1p: bool,
        standardize: bool,
        predict_delta: bool,
        y_mean: Optional[np.ndarray],
        y_std: Optional[np.ndarray],
        out_dim: int,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.nonneg = nonneg

        self.log1p = log1p
        self.standardize = standardize
        self.predict_delta = predict_delta

        self.out_dim = out_dim
        self.y_mean = _to_float_tensor(y_mean)
        self.y_std = _to_float_tensor(y_std)

        if loss_name == "huber":
            self.loss = nn.HuberLoss()
        else:
            self.loss = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _inv_standardize(self, z: torch.Tensor) -> torch.Tensor:
        if not self.standardize:
            return z
        assert self.y_mean is not None and self.y_std is not None
        mu = self.y_mean.to(z.device)
        sd = self.y_std.to(z.device)
        return z * sd + mu

    def _inv_log1p(self, z: torch.Tensor) -> torch.Tensor:
        if not self.log1p:
            return z
        return torch.expm1(z)

    def _to_original_counts(self, x: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        """
        Convert transformed target/pred back to next-week counts.
        y_t is in the model/training space.
        """
        y = y_t
        y = self._inv_standardize(y)
        y = self._inv_log1p(y)

        if self.predict_delta:
            # add back last-week counts (first out_dim features of last timestep)
            last_week = x[:, -1, : self.out_dim]
            y = y + last_week

        if self.nonneg:
            y = torch.clamp(y, min=0.0)
        return y

    def training_step(self, batch, batch_idx: int):
        x, y_t = batch
        pred_t = self(x)
        loss = self.loss(pred_t, y_t)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y_t = batch
        pred_t = self(x)
        loss = self.loss(pred_t, y_t)

        # MAE in ORIGINAL count space
        y_true = self._to_original_counts(x, y_t)
        y_pred = self._to_original_counts(x, pred_t)
        mae = torch.mean(torch.abs(y_pred - y_true))

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/mae", mae, prog_bar=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x, y_t = batch
        pred_t = self(x)
        y_true = self._to_original_counts(x, y_t).detach().cpu()
        y_pred = self._to_original_counts(x, pred_t).detach().cpu()
        return {"pred": y_pred, "y": y_true}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--model_name", default="wildfire_fpa_lstm")
    ap.add_argument("--region", choices=["US", "CA"], default="US")

    ap.add_argument("--lookback", type=int, default=None)
    ap.add_argument("--lookback_weeks", type=int, default=None)
    ap.add_argument("--features", choices=["counts", "counts+time"], default="counts")

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--loss", choices=["mse", "huber"], default="mse")
    ap.add_argument("--nonneg", action="store_true")
    ap.add_argument("--log1p", action="store_true")
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--predict_delta", action="store_true")

    ap.add_argument("--limit_train_batches", type=float, default=1.0)
    ap.add_argument("--limit_val_batches", type=float, default=1.0)
    ap.add_argument("--limit_test_batches", type=float, default=1.0)
    ap.add_argument("--log_every_n_steps", type=int, default=1)

    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--devices", default="auto")
    ap.add_argument("--precision", default="32")

    ap.add_argument("--out_root", default="outputs/forecast_wildfire_fpa_weekly")
    ap.add_argument("--resume_from", default=None)

    args = ap.parse_args()
    args.lookback = _resolve_lookback(args)

    # Hard safety rule (the one you hit)
    if args.predict_delta and args.log1p:
        raise SystemExit(
            "Refusing: --predict_delta with --log1p (delta can be negative; log1p would be invalid). "
            "Use --standardize only, or disable delta."
        )

    run_dir = os.path.join(args.out_root, _now_tag())
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    _mkdir(ckpt_dir)
    _write_json(os.path.join(run_dir, "config.json"), vars(args))
    _write_text(os.path.join(run_dir, "command.txt"), " ".join(sys.argv) + "\n")

    try:
        pl.seed_everything(args.seed, workers=True)

        from pyhazards.datasets import FPAFODWildfireWeekly  # type: ignore

        b = FPAFODWildfireWeekly(
            region=args.region,
            data_path=args.data_path,
            micro=False,
            lookback_weeks=args.lookback,
            features=args.features,
            seed=args.seed,
        ).load()

        x_train, y_train = b.splits["train"].inputs, b.splits["train"].targets
        x_val, y_val = b.splits["val"].inputs, b.splits["val"].targets
        x_test, y_test = b.splits["test"].inputs, b.splits["test"].targets

        out_dim = int(y_train.shape[-1])  # usually 5
        input_dim = int(x_train.shape[-1])

        # --------
        # Build transformed targets for training
        # --------
        def transform_targets(x: np.ndarray, y: np.ndarray, y_mean: Optional[np.ndarray], y_std: Optional[np.ndarray]) -> np.ndarray:
            if isinstance(y, torch.Tensor):
                yt = y.detach().cpu().float()
            else:
                import numpy as np
                yt = np.asarray(y, dtype=np.float32)

            if args.predict_delta:
                last = _counts_slice_for_features(x, out_dim)[:, -1, :]
                yt = yt - last

            if args.log1p:
                yt = np.log1p(np.maximum(yt, 0.0))

            if args.standardize:
                assert y_mean is not None and y_std is not None
                yt = (yt - y_mean) / y_std

            return yt

        # compute standardization stats on TRAIN transformed space (before standardize step itself)
        y_base = y_train.detach().cpu().numpy().astype(np.float32)
        if args.predict_delta:
            last_tr = _counts_slice_for_features(x_train, out_dim)[:, -1, :]
            y_base = y_base - last_tr
        if args.log1p:
            y_base = np.log1p(np.maximum(y_base, 0.0))

        y_mean = None
        y_std = None
        if args.standardize:
            y_mean = y_base.mean(axis=0)
            y_std = y_base.std(axis=0)
            y_std[y_std < 1e-6] = 1.0

        y_train_t = transform_targets(x_train, y_train, y_mean, y_std)
        y_val_t = transform_targets(x_val, y_val, y_mean, y_std)
        y_test_t = transform_targets(x_test, y_test, y_mean, y_std)

        tx_train = (x_train.float() if hasattr(x_train, "float") else torch.from_numpy(x_train).float())
        ty_train = _to_float_tensor(y_train_t)
        tx_val = (x_val.float() if hasattr(x_val, "float") else torch.from_numpy(x_val).float())
        ty_val = _to_float_tensor(y_val_t)
        tx_test = (x_test.float() if hasattr(x_test, "float") else torch.from_numpy(x_test).float())
        ty_test = _to_float_tensor(y_test_t)

        persistent_workers = bool(args.num_workers and args.num_workers > 0)
        train_loader = DataLoader(TensorDataset(tx_train, ty_train), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, persistent_workers=persistent_workers)
        val_loader = DataLoader(TensorDataset(tx_val, ty_val), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, persistent_workers=persistent_workers)
        test_loader = DataLoader(TensorDataset(tx_test, ty_test), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, persistent_workers=persistent_workers)

        model, used_kwargs = build_wildfire_fpa_lstm_strict(
            model_name=args.model_name,
            task="regression",
            input_dim=input_dim,
            output_dim=out_dim,
            lookback=args.lookback,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
        _write_json(os.path.join(run_dir, "model_kwargs_used.json"), used_kwargs)

        lit = LitWeekly(
            model=model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            loss_name=args.loss,
            nonneg=args.nonneg,
            log1p=args.log1p,
            standardize=args.standardize,
            predict_delta=args.predict_delta,
            y_mean=y_mean,
            y_std=y_std,
            out_dim=out_dim,
        )

        ckpt = ModelCheckpoint(dirpath=ckpt_dir, filename="best", monitor="val/mae", mode="min", save_top_k=1, save_last=True)
        logger = CSVLogger(save_dir=run_dir, name="csv")

        accel = _safe_accelerator(args.accelerator)
        devs = _safe_devices(args.devices)

        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator=accel,
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

        outs = trainer.predict(lit, dataloaders=test_loader, ckpt_path=ckpt.best_model_path)
        preds = torch.cat([o["pred"] for o in outs], dim=0).numpy()
        ys = torch.cat([o["y"] for o in outs], dim=0).numpy()

        test_metrics = _mae_rmse(ys, preds)

        # naive baseline: last-week counts (level) always works in original count space
        last_week = _counts_slice_for_features(x_test, out_dim)[:, -1, :]
        naive_metrics = _mae_rmse(y_test, last_week)

        _plot_series(ys, preds, os.path.join(run_dir, "pred_vs_true.png"), "Weekly forecast", SIZE_GROUPS[:out_dim])
        torch.save(lit.model.state_dict(), os.path.join(run_dir, "model_final.pt"))

        metrics = {
            "task": "fpa_fod_weekly",
            "config": vars(args),
            "git_sha": _git_sha(),
            "system": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__},
            "best_ckpt": ckpt.best_model_path,
            "last_ckpt": ckpt.last_model_path,
            "test": test_metrics,
            "naive_last_week_baseline": naive_metrics,
        }
        _write_json(os.path.join(run_dir, "metrics.json"), metrics)

        summary = [
            "# Paper reproduction summary — FPA-FOD Weekly Forecast",
            "",
            f"## Model kwargs actually used (strict builder)\n- {used_kwargs}",
            "",
            "## Your test metrics",
            f"- MAE:  {test_metrics['mae']:.3f}",
            f"- RMSE: {test_metrics['rmse']:.3f}",
            f"- MAE per group:  {test_metrics.get('mae_per_group')}",
            f"- RMSE per group: {test_metrics.get('rmse_per_group')}",
            "",
            "## Naive baseline (last week)",
            f"- MAE:  {naive_metrics['mae']:.3f}",
            f"- RMSE: {naive_metrics['rmse']:.3f}",
        ]
        _write_text(os.path.join(run_dir, "paper_summary.md"), "\n".join(summary) + "\n")

        print(f"[OK] Outputs saved to: {run_dir}")

    except BaseException as e:
        _write_text(os.path.join(run_dir, "FAILURE.txt"), f"{type(e).__name__}: {e}\n")
        raise


if __name__ == "__main__":
    main()
