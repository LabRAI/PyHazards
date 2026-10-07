import math

import torch

from pyhazards.datasets import DataBundle, DataSplit, FeatureSpec, LabelSpec
from pyhazards.engine import Trainer


def test_trainer_evaluate_uses_default_regression_metrics():
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    targets = torch.tensor([[1.0], [2.0]])
    bundle = DataBundle(
        splits={"test": DataSplit(inputs=inputs, targets=targets)},
        feature_spec=FeatureSpec(input_dim=2),
        label_spec=LabelSpec(num_targets=1, task_type="regression"),
    )

    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()

    metrics = Trainer(model=model, device="cpu", mixed_precision=False).evaluate(
        bundle,
        batch_size=2,
    )

    assert set(metrics) == {"MAE", "RMSE"}
    assert math.isclose(metrics["MAE"], 1.5, rel_tol=1e-6)
    assert math.isclose(metrics["RMSE"], math.sqrt(2.5), rel_tol=1e-6)
