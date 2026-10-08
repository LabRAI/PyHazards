#!/usr/bin/env python3
"""Train HydroGraphNet as the PhysicsNeMo example does, then score autoregressive rollouts.

Recipe of examples/weather/flood_modeling/hydrographnet (train.py, conf/config.yaml): batch size 1, Adam
with learning rate 1e-4 (the configured weight decay is not passed to the optimizer), the learning rate
multiplied by 0.9999979 after every batch, 100 epochs, loss = MSE + 1.0 * physics (volume continuity,
delta_t 1200 s), noise_type "none"; inference rolls out 30 steps per test hydrograph.

usage:
  python scripts/train_hydrographnet.py --data-dir /data/HydroGraphNet --test-ids test.txt
  python scripts/train_hydrographnet.py --synthetic --epochs 1     # smoke run on flood_mesh_synthetic
"""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from pyhazards.benchmarks.flood import evaluate_inundation
from pyhazards.datasets import load_dataset
from pyhazards.datasets.flood.hydrograph import hydrograph_collate
from pyhazards.models import build_model
from pyhazards.models.hydrographnet import HydroGraphNetLoss


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", help="unzipped HydroGraphNet.zip (Zenodo 14969507)")
    parser.add_argument("--test-ids", help="file or comma-separated list of test hydrograph ids")
    parser.add_argument("--synthetic", action="store_true", help="use flood_mesh_synthetic instead of real data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-decay-rate", type=float, default=0.9999979)
    parser.add_argument("--physics-loss-weight", type=float, default=1.0)
    parser.add_argument("--rollout-length", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    if args.synthetic:
        data = load_dataset("flood_mesh_synthetic", micro=True).load()
    else:
        if not args.data_dir or not args.test_ids:
            parser.error("--data-dir and --test-ids are required (or pass --synthetic)")
        test_ids = args.test_ids if args.test_ids.endswith(".txt") else args.test_ids.split(",")
        data = load_dataset(
            "hydrographnet_white_river", data_dir=args.data_dir, test_ids=test_ids, rollout_length=args.rollout_length
        ).load()

    model = build_model("hydrographnet", task="regression").to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: args.lr_decay_rate**step)
    criterion = HydroGraphNetLoss(physics_loss_weight=args.physics_loss_weight, delta_t=1200.0)
    loader = DataLoader(data.get_split("train").inputs, batch_size=1, shuffle=True, collate_fn=hydrograph_collate)

    for epoch in range(args.epochs):
        model.train()
        total, batches = 0.0, 0
        for batch, target in loader:
            physics = {key: value.to(args.device) for key, value in batch.pop("physics").items()}
            batch = {key: value.to(args.device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss, _ = criterion(model(batch), target.to(args.device), physics, batch["batch"])
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += float(loss.detach())
            batches += 1
        print(f"epoch {epoch}: mean loss {total / max(batches, 1):.4e}")

    model.eval()
    metrics, extra = evaluate_inundation(model, data, "test")
    print(json.dumps({"metrics": metrics, "rollout_rmse_per_step_m": extra.get("rollout_rmse_per_step_m")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
