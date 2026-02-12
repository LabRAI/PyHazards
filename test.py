import os
import math
os.environ["PYHAZARDS_DEVICE"] = "cuda:0"

import torch
from torch.utils.data import DataLoader

from pyhazards.data.load_hydrograph_data import load_hydrograph_data
from pyhazards.datasets import graph_collate
from pyhazards.engine import Trainer
from pyhazards.metrics import RegressionMetrics
from pyhazards.models import build_model


def _to_device(obj, device):
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please install a CUDA-enabled PyTorch build "
            "compatible with your GPU and ensure NVIDIA driver/CUDA runtime are working."
        )

    device = torch.device("cuda:0")
    print("== PyHazards GPU smoke test (ERA5 + HydroGraphNet) ==")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA runtime in torch: {torch.version.cuda}")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

    data = load_hydrograph_data("pyhazards/data/era5_subset", max_nodes=50)
    assert "train" in data.splits, "Expected 'train' split in loaded data."
    assert data.feature_spec.input_dim == 2, f"Unexpected input_dim: {data.feature_spec.input_dim}"
    assert data.label_spec.task_type == "regression", f"Unexpected task type: {data.label_spec.task_type}"

    train_inputs = data.get_split("train").inputs
    print(f"Loaded split keys: {list(data.splits.keys())}")
    print(f"Dataset samples: {len(train_inputs)}")
    print(f"Feature spec: {data.feature_spec}")
    print(f"Label spec: {data.label_spec}")

    model = build_model(
        name="hydrographnet",
        task="regression",
        node_in_dim=2,
        edge_in_dim=3,
        out_dim=1,
    )
    model = model.to(device)
    print(f"Model: {type(model).__name__}")

    # Forward-pass sanity check on one real batch on GPU.
    sample_loader = DataLoader(train_inputs, batch_size=1, shuffle=False, collate_fn=graph_collate)
    batch_x, batch_y = next(iter(sample_loader))
    batch_x = _to_device(batch_x, device)
    batch_y = _to_device(batch_y, device)
    with torch.no_grad():
        pred = model(batch_x)
    assert pred.shape == batch_y.shape, f"Prediction shape {pred.shape} != target shape {batch_y.shape}"
    print(f"Forward pass OK: pred shape {tuple(pred.shape)}")

    trainer = Trainer(model=model, device="cuda:0", metrics=[RegressionMetrics()], mixed_precision=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    trainer.fit(
        data,
        optimizer=optimizer,
        loss_fn=loss_fn,
        max_epochs=1,
        batch_size=1,
        collate_fn=graph_collate,
    )

    metrics = trainer.evaluate(
        data,
        split="train",
        batch_size=1,
        collate_fn=graph_collate,
    )

    assert "MAE" in metrics and "RMSE" in metrics, f"Missing expected metrics: {metrics}"
    assert math.isfinite(metrics["MAE"]) and math.isfinite(metrics["RMSE"]), f"Non-finite metrics: {metrics}"

    print(f"Evaluation metrics: {metrics}")
    print("Note: MAE/RMSE are error metrics (lower is better). This smoke test is for pipeline validity, not benchmark quality.")
    print("PASS: end-to-end implementation is working.")


if __name__ == "__main__":
    main()
