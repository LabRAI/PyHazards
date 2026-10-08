"""UrbanFloodCast DNO, its one-shot inputs and the inundation metrics: counts, shapes and validation."""

from __future__ import annotations

import pytest
import torch

from pyhazards.configs import load_experiment_config
from pyhazards.datasets import load_dataset
from pyhazards.datasets.flood.urbanfloodcast import prepare_urbanfloodcast_event, synthetic_urbanfloodcast_event
from pyhazards.engine.runner import BenchmarkRunner
from pyhazards.metrics.inundation import (
    critical_success_index,
    inundation_metrics,
    nash_sutcliffe,
    pearson_correlation,
    relative_l2,
)
from pyhazards.models import build_model
from pyhazards.models.urbanfloodcast import UrbanFloodCast


def test_parameter_count_and_names_at_official_configuration():
    model = build_model("urbanfloodcast", task="regression")
    assert sum(p.numel() for p in model.parameters()) == 4_470_437
    assert sum(p.numel() * (1 + p.is_complex()) for p in model.parameters()) == 8_937_637
    counts = {name: sum(p.numel() for p in getattr(model, name).parameters()) for name in ("fc", "fc0", "conv0", "conv7", "conv8", "fc1", "fc2")}
    assert counts == {"fc": 144, "fc0": 170, "conv0": 1_280_560, "conv7": 627_760, "conv8": 2_560_840, "fc1": 840, "fc2": 123}
    keys = set(model.state_dict())
    for key in ("conv0.conv.weights1", "conv0.mlp.mlp1.weight", "conv0.w.conv.weight", "conv0.normalize_layer.weight", "conv8.mlp.mlp2.bias"):
        assert key in keys
    assert "conv8.normalize_layer.weight" not in keys
    assert model.conv0.conv.weights1.dtype == torch.complex64


def test_forward_predicts_all_steps_at_once():
    torch.manual_seed(0)
    model = UrbanFloodCast().eval()
    x = torch.randn(2, 28, 30, 14, 1, 5)
    with torch.no_grad():
        out = model(x)
        assert out.shape == (2, 28, 30, 14, 3)
        torch.testing.assert_close(model(x.reshape(2, 28, 30, 14, 5)), out)


@pytest.mark.parametrize(
    "shape",
    [(1, 28, 28, 14, 1, 4), (1, 26, 28, 14, 1, 5), (1, 28, 28, 13, 1, 5), (28, 28, 14, 5)],
)
def test_bad_shapes_raise(shape):
    with pytest.raises(ValueError, match="shape|ndim"):
        UrbanFloodCast()(torch.randn(*shape))


def test_builder_validation():
    with pytest.raises(ValueError, match="regression"):
        build_model("urbanfloodcast", task="segmentation")
    with pytest.raises(ValueError, match="Unknown"):
        build_model("urbanfloodcast", task="regression", history=4)
    with pytest.raises(ValueError, match="pad"):
        UrbanFloodCast(pad=1)


def test_one_shot_event_preparation():
    event = synthetic_urbanfloodcast_event(30, 31, 25, torch.Generator().manual_seed(0))
    event[0, 0, :, 0] = 0.0
    event[0, 0, :, 1:3] = 5.0  # discharge where the depth is zero is dropped
    before = event.clone()
    x, y, mask = prepare_urbanfloodcast_event(event)
    assert torch.equal(event.nan_to_num(-1), before.nan_to_num(-1))
    assert x.shape == (30, 31, 24, 1, 5) and y.shape == mask.shape == (30, 31, 24, 3)
    assert torch.all(x[0, 0, :, 0, 1:3] == 0) and torch.all(y[0, 0, :, 1:3] == 0)
    assert torch.equal(x[:, :, 3, 0, :3], x[:, :, 0, 0, :3])  # the first step is repeated over the outputs
    torch.testing.assert_close(x[..., 0, 3], torch.log(1 + before[:, :, 1:, 3] / 0.01) / 10)
    assert torch.equal(mask, ~torch.isnan(before[:, :, 1:, :3]))
    assert torch.isfinite(y).all()
    with pytest.raises(ValueError, match="shape"):
        prepare_urbanfloodcast_event(torch.zeros(4, 4, 25, 3))
    with pytest.raises(ValueError, match="steps"):
        prepare_urbanfloodcast_event(torch.zeros(4, 4, 10, 5))


def test_synthetic_dataset_layout():
    bundle = load_dataset("urbanfloodcast_synthetic", micro=True).load()
    train = bundle.get_split("train")
    assert train.inputs.shape[1:] == (32, 32, 24, 1, 5) and train.targets.shape[1:] == (32, 32, 24, 3)
    assert train.metadata["mask"].dtype == torch.bool
    assert bundle.metadata["synthetic"] is True and bundle.metadata["depth_channel"] == 0


def test_metric_definitions():
    target = torch.tensor([[[0.0], [0.2], [0.6], [1.0]]])
    assert float(nash_sutcliffe(target, target)) == 1.0
    assert float(pearson_correlation(target, target)) == pytest.approx(0.75)  # population cov / sample std, as the reference
    assert float(relative_l2(target, target).sum()) == 0.0
    assert float(relative_l2(2 * target, target).sum()) == pytest.approx(1.0)
    pred = torch.tensor([[[0.05], [0.0], [0.7], [0.3]]])
    # threshold 0.1: hits (0.6, 1.0), misses (0.2), false alarms none -> 2 / 3
    assert float(critical_success_index(pred, target, 0.1)) == pytest.approx(2 / 3)
    metrics = inundation_metrics(pred.reshape(1, 4), target.reshape(1, 4))
    assert metrics["pixel_mae"] == pytest.approx((0.05 + 0.2 + 0.1 + 0.7) / 4)
    assert metrics["iou"] == pytest.approx(1 / 3)  # predicted wet (>= 0.5): 0.7; observed wet (> 0): 3 cells
    with pytest.raises(ValueError, match="match"):
        inundation_metrics(torch.zeros(2, 3), torch.zeros(2, 4))


def test_smoke_config_scores_all_variables(tmp_path):
    summary = BenchmarkRunner().run(load_experiment_config("pyhazards/configs/flood/urbanfloodcast_smoke.yaml"), output_dir=str(tmp_path))
    for metric in ("relative_l2", "nse", "pearson_r", "csi_1cm", "csi_10cm", "csi_50cm", "pixel_mae", "rmse", "iou", "f1"):
        assert metric in summary.metrics
    assert "rollout_rmse" not in summary.metrics
