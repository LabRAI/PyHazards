import torch

from pyhazards.datasets import load_dataset
from pyhazards.datasets.base import DataSplit
from pyhazards.engine import Trainer
from pyhazards.models import build_model


def test_trainer_fit_smoke():
    # Load dataset via registry; micro=True must not require any real files
    ds = load_dataset("wildfire_fpa_fod_tabular", micro=True, seed=1337)

    # Support both patterns: registry may return Dataset instance or DataBundle directly
    bundle = ds.load() if hasattr(ds, "load") else ds

    train = bundle.get_split("train")
    x_train = train.inputs
    y_train = train.targets

    # Enforce torch tensors for Trainer determinism
    if not torch.is_tensor(x_train):
        x_train = torch.as_tensor(x_train, dtype=torch.float32)
    else:
        x_train = x_train.to(dtype=torch.float32)

    if not torch.is_tensor(y_train):
        y_train = torch.as_tensor(y_train, dtype=torch.long)
    else:
        y_train = y_train.to(dtype=torch.long)

    # Put back into bundle as a proper DataSplit (NOT a dict)
    bundle.splits = dict(bundle.splits)
    bundle.splits["train"] = DataSplit(inputs=x_train, targets=y_train, metadata=getattr(train, "metadata", {}))

    in_dim = int(x_train.shape[1])
    out_dim = int(y_train.max().item() + 1)

    model = build_model(
        name="wildfire_fpa_mlp",
        task="classification",
        in_dim=in_dim,
        out_dim=out_dim,
    )

    trainer = Trainer(model=model, mixed_precision=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Smoke: 1 epoch, no crash
    trainer.fit(
        bundle,
        optimizer=optimizer,
        loss_fn=loss_fn,
        max_epochs=1,
        batch_size=32,
        num_workers=0,
    )
