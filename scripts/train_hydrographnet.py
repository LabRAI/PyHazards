#!/usr/bin/env python3

import torch
import torch.nn.functional as F

from pyhazards.engine import Trainer
from pyhazards.models import build_model
from pyhazards.datasets import graph_collate
from pyhazards.metrics import RegressionMetrics

from pyhazards.data.load_hydrograph_data import load_hydrograph_data


def main():
    torch.manual_seed(0)

    bundle = load_hydrograph_data(
        era5_path="pyhazards/data/era5_subset",
    )

    train_split = "train"

    # Infer dimensions from dataset
    sample_x, sample_y = bundle.splits[train_split].inputs[0]

    x_tensor = sample_x["x"]

    node_feats = x_tensor.shape[2]
    out_dim = 1

    print("Dataset shapes:")
    print("  x:", x_tensor.shape)
    print("  y:", sample_y.shape)

    # Model
    model = build_model(
        name="hydrographnet",
        task="regression",
        node_in_dim=node_feats,
        edge_in_dim=3,
        out_dim=out_dim,
    )
    # Optimizer + loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def loss_fn(pred, target):
        return F.mse_loss(pred, target)

    # Trainer
    trainer = Trainer(model=model, metrics=[RegressionMetrics()])

    trainer.fit(
        bundle,
        train_split=train_split,
        val_split=None,
        optimizer=optimizer,
        loss_fn=loss_fn,
        batch_size=1,
        max_epochs=5,
        collate_fn=graph_collate,
    )
    metrics = trainer.evaluate(
        bundle,
        split=train_split,
        batch_size=1,
        collate_fn=graph_collate,
    )
    print("Evaluation metrics:", metrics)


if __name__ == "__main__":
    main()
