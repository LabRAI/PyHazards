import torch

from pyhazards.datasets import load_dataset
from pyhazards.engine import Trainer
from pyhazards.models import build_model


def test_fpa_fod_trainer_smoke():
    bundle = load_dataset("fpa_fod_tabular", micro=True, task="cause").load()
    train = bundle.get_split("train")

    model = build_model(
        name="wildfire_fpa",
        task="classification",
        in_dim=train.inputs.shape[1],
        out_dim=int(train.targets.max().item() + 1),
    )
    trainer = Trainer(model=model, mixed_precision=False, device="cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    trainer.fit(
        bundle,
        optimizer=optimizer,
        loss_fn=loss_fn,
        max_epochs=1,
        batch_size=32,
        num_workers=0,
    )
