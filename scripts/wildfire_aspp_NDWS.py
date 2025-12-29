#!/usr/bin/env python3
"""
train_wildfire_aspp_micro.py

End-to-end trainer for wildfire next-day spread segmentation on NPZ tiles.
- PyHazards-registered CNN + ASPP model (wildfire_cnn_aspp via build_model)
- Lightning training loop
- Fast epoch metrics during training (TP/FP/FN/TN -> Precision/Recall/F1/IoU/PixelAcc)
- Threshold sweep after training (optionally with small-component removal)

Expected NPZ keys:
- inputs: (C,H,W) float32
- targets: (1,H,W) or (H,W) {0,1}
Also supports x/y, image/mask, label/target/targets.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from pyhazards.models.wildfire_aspp import WildfireASPP, TverskyLoss

# ✅ Use the PyHazards model registry
from pyhazards.models import build_model
# keep your loss
from pyhazards.models.wildfire_aspp import TverskyLoss


# -----------------------------
# Utilities
# -----------------------------

def seed_everything(seed: int) -> None:
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def list_npz_files(root: str) -> List[str]:
    root_p = Path(root)
    if root_p.is_file() and root_p.suffix == ".npz":
        return [str(root_p)]
    if not root_p.exists():
        raise FileNotFoundError(f"root not found: {root}")
    files = sorted([str(p) for p in root_p.rglob("*.npz")])
    if not files:
        raise FileNotFoundError(f"No .npz files found under: {root}")
    return files


def load_npz_xy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    x_keys = ["x", "X", "inputs", "image", "inp"]
    y_keys = ["y", "Y", "label", "mask", "target", "targets"]

    x = None
    y = None
    for k in x_keys:
        if k in d:
            x = d[k]
            break
    for k in y_keys:
        if k in d:
            y = d[k]
            break

    if x is None or y is None:
        raise KeyError(
            f"{path} missing required keys. Found keys={list(d.keys())}. "
            f"Expected one of x={x_keys} and y={y_keys}."
        )

    x = x.astype(np.float32)
    y = y.astype(np.float32)

    if x.ndim == 2:
        x = x[None, ...]
    if x.ndim != 3:
        raise ValueError(f"{path}: x must be (C,H,W) got {x.shape}")

    if y.ndim == 2:
        y = y[None, ...]
    if y.ndim == 3 and y.shape[0] != 1 and y.shape[-1] == 1:
        y = np.transpose(y, (2, 0, 1))
    if y.ndim != 3 or y.shape[0] != 1:
        raise ValueError(f"{path}: y must be (1,H,W) or (H,W) got {y.shape}")

    y = (y > 0.5).astype(np.float32)
    return x, y


def save_json(obj: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def fbeta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision <= 0 and recall <= 0:
        return 0.0
    b2 = beta * beta
    denom = (b2 * precision + recall)
    return 0.0 if denom == 0 else (1 + b2) * precision * recall / denom


# -----------------------------
# Post-processing: remove small FP blobs
# -----------------------------

def remove_small_components(mask: np.ndarray, min_component_size: int = 0, **kwargs) -> np.ndarray:
    """
    Removes connected components smaller than min_component_size.
    Accepts **kwargs for backward-compat with older call-sites that used min_size=...
    """
    if "min_size" in kwargs and min_component_size == 0:
        try:
            min_component_size = int(kwargs["min_size"])
        except Exception:
            pass

    if min_component_size <= 0:
        return mask

    try:
        from scipy.ndimage import label  # type: ignore
        lab, n = label(mask.astype(np.uint8))
        if n == 0:
            return mask
        counts = np.bincount(lab.ravel())
        keep = counts >= min_component_size
        keep[0] = False
        return keep[lab]
    except Exception:
        H, W = mask.shape
        visited = np.zeros((H, W), dtype=np.uint8)
        out = mask.copy()

        def neighbors(r, c):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    yield rr, cc

        for r in range(H):
            for c in range(W):
                if out[r, c] and not visited[r, c]:
                    stack = [(r, c)]
                    coords = []
                    visited[r, c] = 1
                    while stack:
                        rr, cc = stack.pop()
                        coords.append((rr, cc))
                        for r2, c2 in neighbors(rr, cc):
                            if out[r2, c2] and not visited[r2, c2]:
                                visited[r2, c2] = 1
                                stack.append((r2, c2))
                    if len(coords) < min_component_size:
                        for rr, cc in coords:
                            out[rr, cc] = False
        return out


# -----------------------------
# Dataset / Splits
# -----------------------------

class NPZTileDataset(Dataset):
    def __init__(self, files: List[str]):
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = load_npz_xy(self.files[idx])
        return torch.from_numpy(x), torch.from_numpy(y)


def make_split(files: List[str], seed: int, val_ratio: float, test_ratio: float) -> Dict[str, List[str]]:
    rng = np.random.default_rng(seed)
    files = list(files)
    rng.shuffle(files)
    n = len(files)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_train = n - n_val - n_test
    train = files[:n_train]
    val = files[n_train:n_train + n_val]
    test = files[n_train + n_val:]
    return {"train": train, "val": val, "test": test}


def tile_has_positive(npz_path: str) -> bool:
    _, y = load_npz_xy(npz_path)
    return bool((y > 0.5).any())


# -----------------------------
# Metrics dataclass (offline eval / sweep)
# -----------------------------

@dataclass
class Metrics:
    PixelAcc: float
    Precision: float
    Recall: float
    F1: float
    IoU: float
    Threshold: float


@torch.no_grad()
def compute_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    min_component_size: int = 0,
) -> Metrics:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y = targets.detach().cpu().numpy().astype(np.uint8)

    tp = fp = fn = tn = 0
    for i in range(probs.shape[0]):
        pm = probs[i, 0]
        yt = y[i, 0].astype(bool)

        pred = (pm >= threshold)
        pred = remove_small_components(pred, min_component_size=min_component_size)

        tp += int(np.logical_and(pred, yt).sum())
        fp += int(np.logical_and(pred, np.logical_not(yt)).sum())
        fn += int(np.logical_and(np.logical_not(pred), yt).sum())
        tn += int(np.logical_and(np.logical_not(pred), np.logical_not(yt)).sum())

    pixel_acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    iou = tp / max(tp + fp + fn, 1)

    return Metrics(
        PixelAcc=float(pixel_acc),
        Precision=float(precision),
        Recall=float(recall),
        F1=float(f1),
        IoU=float(iou),
        Threshold=float(threshold),
    )


# -----------------------------
# Lightning Module (FAST val/test metrics)
# -----------------------------

class WildfireLitModule(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        lr: float,
        loss: str,
        pos_weight_value: float,
        use_pos_weight: bool,
        tversky_alpha: float,
        tversky_beta: float,
        bce_weight: float,
        tversky_weight: float,
        threshold: float,
        # NOTE: we keep min_component_size for offline eval only (not used in fast epoch metrics)
        min_component_size: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ✅ build from registry (your CNN+ASPP implemented in PyHazards)
        self.model = build_model(
            name="wildfire_cnn_aspp",
            task="segmentation",
            in_channels=int(in_channels),
            base_channels=32,
            aspp_channels=32,
            dilations=(1, 3, 6, 12),
            dropout=0.0,
            )

        capped = min(float(pos_weight_value), 25.0)
        self.register_buffer("pos_weight", torch.tensor([capped], dtype=torch.float32))

        self.loss_name = loss
        self.thr = float(threshold)

        self.tversky = TverskyLoss(alpha=float(tversky_alpha), beta=float(tversky_beta))
        if bool(use_pos_weight):
            self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            self.bce = nn.BCEWithLogitsLoss()

        self.bce_weight = float(bce_weight)
        self.tversky_weight = float(tversky_weight)

        # running confusion for val/test (epoch)
        self._reset_running("val")
        self._reset_running("test")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _reset_running(self, stage: str) -> None:
        setattr(self, f"{stage}_tp", 0.0)
        setattr(self, f"{stage}_fp", 0.0)
        setattr(self, f"{stage}_fn", 0.0)
        setattr(self, f"{stage}_tn", 0.0)

    @torch.no_grad()
    def _update_running(self, stage: str, logits: torch.Tensor, y: torch.Tensor) -> None:
        # IMPORTANT: no connected-components here (too slow). Just threshold.
        p = (torch.sigmoid(logits) >= self.thr).to(torch.int32)
        t = (y >= 0.5).to(torch.int32)

        tp = (p & t).sum().item()
        fp = (p & (1 - t)).sum().item()
        fn = ((1 - p) & t).sum().item()
        tn = ((1 - p) & (1 - t)).sum().item()

        setattr(self, f"{stage}_tp", getattr(self, f"{stage}_tp") + tp)
        setattr(self, f"{stage}_fp", getattr(self, f"{stage}_fp") + fp)
        setattr(self, f"{stage}_fn", getattr(self, f"{stage}_fn") + fn)
        setattr(self, f"{stage}_tn", getattr(self, f"{stage}_tn") + tn)

    def _log_running(self, stage: str) -> None:
        tp = getattr(self, f"{stage}_tp")
        fp = getattr(self, f"{stage}_fp")
        fn = getattr(self, f"{stage}_fn")
        tn = getattr(self, f"{stage}_tn")

        pixel_acc = (tp + tn) / max(tp + tn + fp + fn, 1.0)
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        iou = tp / max(tp + fp + fn, 1.0)

        self.log(f"{stage}/PixelAcc", pixel_acc, prog_bar=False, on_epoch=True)
        self.log(f"{stage}/Precision", precision, prog_bar=False, on_epoch=True)
        self.log(f"{stage}/Recall", recall, prog_bar=False, on_epoch=True)
        self.log(f"{stage}/F1", f1, prog_bar=True, on_epoch=True)
        self.log(f"{stage}/IoU", iou, prog_bar=True, on_epoch=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        if self.loss_name == "bce":
            loss = self.bce(logits, y)
        elif self.loss_name == "tversky":
            loss = self.tversky(logits, y)
        elif self.loss_name == "bce_tversky":
            loss = self.bce_weight * self.bce(logits, y) + self.tversky_weight * self.tversky(logits, y)
        else:
            raise ValueError(f"Unknown loss: {self.loss_name}")

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # ✅ fixes the "no validation_step" issue
    def on_validation_epoch_start(self):
        self._reset_running("val")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self._update_running("val", logits, y)

    def on_validation_epoch_end(self):
        self._log_running("val")

    def on_test_epoch_start(self):
        self._reset_running("test")

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self._update_running("test", logits, y)

    def on_test_epoch_end(self):
        self._log_running("test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=float(self.hparams.lr))


# -----------------------------
# Eval helpers (offline collect + sweep)
# -----------------------------

@torch.no_grad()
def gather_logits_targets(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_logits = []
    all_targets = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)


@torch.no_grad()
def sweep_thresholds(
    logits: torch.Tensor,
    targets: torch.Tensor,
    thresholds: np.ndarray,
    pick: str,
    beta: float,
    min_recall: float,
    min_component_size: int,
) -> Metrics:
    best: Metrics | None = None
    best_score = -1e9

    for t in thresholds:
        m = compute_metrics_from_logits(
            logits,
            targets,
            threshold=float(t),
            min_component_size=int(min_component_size),
        )

        if pick == "f05":
            score = fbeta(m.Precision, m.Recall, beta=float(beta))

        elif pick == "precision_at_recall":
            # HARD constraint: only consider thresholds that meet recall requirement
            if m.Recall < float(min_recall):
                continue
            score = m.Precision

        else:
            raise ValueError(f"Unknown pick mode: {pick}")

        if score > best_score:
            best_score = score
            best = m

    if best is None:
        raise RuntimeError(
            f"No threshold achieved recall >= {float(min_recall):.4f}. "
            f"Try lowering --min_recall, adjusting thresholds range, or retraining."
        )

    return best


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", type=str, required=True, help="Path to NPZ files or directory containing NPZs")
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=4e-4)

    ap.add_argument("--loss", type=str, default="bce_tversky", choices=["bce", "tversky", "bce_tversky"])
    ap.add_argument("--tversky_alpha", type=float, default=0.8)
    ap.add_argument("--tversky_beta", type=float, default=0.2)

    ap.add_argument("--use_pos_weight", action="store_true", help="Use BCE pos_weight (often hurts precision)")
    ap.add_argument("--bce_weight", type=float, default=0.5)
    ap.add_argument("--tversky_weight", type=float, default=0.5)

    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min_component_size", type=int, default=0)

    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)

    ap.add_argument("--device", type=str, default="cpu", help="cpu|cuda|mps")
    ap.add_argument("--log_every_steps", type=int, default=50)

    # oversampling: biggest lever for learning positives
    ap.add_argument("--pos_oversample", type=float, default=5.0, help=">1 oversamples tiles that contain positives")
    ap.add_argument("--no_oversample", action="store_true", help="Disable positive-tile oversampling")

    # eval/sweep
    ap.add_argument("--sweep_after", action="store_true")
    ap.add_argument("--sweep_pick", type=str, default="f05", choices=["f05", "precision_at_recall"])
    ap.add_argument("--sweep_beta", type=float, default=0.5)
    ap.add_argument("--min_recall", type=float, default=0.30)
    ap.add_argument("--sweep_n", type=int, default=40)

    ap.add_argument("--ckpt_out", type=str, default="aspp_micro.ckpt")
    ap.add_argument("--eval_only", action="store_true", help="Skip training; just load ckpt and eval/sweep")
    ap.add_argument("--ckpt_in", type=str, default="", help="Path to ckpt for --eval_only")

    args = ap.parse_args()
    seed_everything(args.seed)

    # split files
    files = list_npz_files(args.root)
    split = make_split(files, seed=args.seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    print(f"[files] train={len(split['train'])}  val={len(split['val'])}  test={len(split['test'])}")
    save_json(split, "split_manifest.json")
    print("[manifest] saved: split_manifest.json")

    # datasets
    train_ds = NPZTileDataset(split["train"])
    val_ds = NPZTileDataset(split["val"])
    test_ds = NPZTileDataset(split["test"])

    # sanity
    x0, y0 = train_ds[0]
    print(f"[sanity] x0: {tuple(x0.shape)} {x0.dtype}")
    print(f"[sanity] y0: {tuple(y0.shape)} {y0.dtype}")

    # estimate pos_weight (pixel-level, sample subset)
    sample_k = min(len(train_ds), 256)
    pos = 0
    neg = 0
    for i in range(sample_k):
        _, y = train_ds[i]
        yy = y.numpy().astype(np.uint8)
        pos += int(yy.sum())
        neg += int(yy.size - yy.sum())
    pos_weight = (neg / max(pos, 1))
    print(f"[pos_weight] approx {pos_weight:.3f} (enable with --use_pos_weight; often hurts precision)")

    # device selection
    if args.device == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
        accelerator = "cuda"
        devices = 1
    elif args.device == "mps" and torch.backends.mps.is_available():
        dev = torch.device("mps")
        accelerator = "mps"
        devices = 1
    else:
        dev = torch.device("cpu")
        accelerator = "cpu"
        devices = 1
    print(f"[trainer] accelerator={accelerator} devices={devices}")

    # loaders
    sampler = None
    if not args.no_oversample and len(split["train"]) > 0 and args.pos_oversample and args.pos_oversample > 1.0:
        print("[oversample] scanning train tiles for positives (one-time)...")
        has_pos = np.array([tile_has_positive(p) for p in split["train"]], dtype=np.bool_)
        npos = int(has_pos.sum())
        nneg = int((~has_pos).sum())
        print(f"[oversample] train tiles: pos={npos} neg={nneg} pos%={npos/max(npos+nneg,1):.4f}")

        w = np.ones(len(has_pos), dtype=np.float32)
        w[has_pos] = float(args.pos_oversample)
        sampler = WeightedRandomSampler(weights=w.tolist(), num_samples=len(w), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(accelerator == "cuda"),
        persistent_workers=bool(args.num_workers and args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(accelerator == "cuda"),
        persistent_workers=bool(args.num_workers and args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(accelerator == "cuda"),
        persistent_workers=bool(args.num_workers and args.num_workers > 0),
    )

    # module
    lit = WildfireLitModule(
        in_channels=int(x0.shape[0]),
        lr=args.lr,
        loss=args.loss,
        pos_weight_value=float(pos_weight),
        use_pos_weight=bool(args.use_pos_weight),
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        bce_weight=args.bce_weight,
        tversky_weight=args.tversky_weight,
        threshold=args.threshold,
        min_component_size=args.min_component_size,
    )

    # load ckpt (eval only)
    if args.eval_only:
        if not args.ckpt_in:
            raise ValueError("--eval_only requires --ckpt_in path")
        ck = torch.load(args.ckpt_in, map_location="cpu")
        lit.load_state_dict(ck["state_dict"])
        lit = lit.to(dev)
        print(f"[eval_only] loaded ckpt: {args.ckpt_in}")
    else:
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator=accelerator,
            devices=devices,
            log_every_n_steps=args.log_every_steps,
            enable_checkpointing=False,
            enable_progress_bar=True,
        )
        trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)

        torch.save({"state_dict": lit.state_dict(), "hparams": dict(lit.hparams)}, args.ckpt_out)
        print(f"[ckpt] saved: {args.ckpt_out}")

    # offline eval + sweep (with optional component removal)
    lit = lit.to(dev)
    logits_val, targets_val = gather_logits_targets(lit.model.to(dev), val_loader, dev)
    logits_test, targets_test = gather_logits_targets(lit.model.to(dev), test_loader, dev)

    val_m = compute_metrics_from_logits(
        logits_val, targets_val,
        threshold=args.threshold,
        min_component_size=args.min_component_size,
    )
    test_m = compute_metrics_from_logits(
        logits_test, targets_test,
        threshold=args.threshold,
        min_component_size=args.min_component_size,
    )

    print("val:", val_m.__dict__)
    print("test:", test_m.__dict__)

    if args.sweep_after:
        # ✅ include tiny thresholds (critical for recall>=0.30 constraints)
        thresholds = np.linspace(5e-4, 2e-2, args.sweep_n, dtype=np.float32)
        best = sweep_thresholds(
            logits=logits_val,
            targets=targets_val,
            thresholds=thresholds,
            pick=args.sweep_pick,
            beta=args.sweep_beta,
            min_recall=args.min_recall,
            min_component_size=args.min_component_size,
        )
        print(f"\n[SWEEP] best on val (by {args.sweep_pick}): {best.__dict__}")


if __name__ == "__main__":
    main()
