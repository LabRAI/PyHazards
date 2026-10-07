from pyhazards.configs import load_experiment_config
from pyhazards.engine.runner import BenchmarkRunner


def test_wildfire_danger_vertical_slice(tmp_path):
    config = load_experiment_config("pyhazards/configs/wildfire/wildfire_danger_smoke.yaml")
    summary = BenchmarkRunner().run(config, output_dir=str(tmp_path))

    assert summary.benchmark_name == "wildfire"
    assert summary.hazard_task == "wildfire.danger"
    assert "accuracy" in summary.metrics
    assert "auc" in summary.metrics


def test_wildfire_spread_vertical_slice(tmp_path):
    config = load_experiment_config("pyhazards/configs/wildfire/wildfire_spread_smoke.yaml")
    summary = BenchmarkRunner().run(config, output_dir=str(tmp_path))

    assert summary.hazard_task == "wildfire.spread"
    assert "iou" in summary.metrics
    assert "burned_area_mae" in summary.metrics
    assert 0.0 <= summary.metrics["average_precision"] <= 1.0


def test_spread_average_precision_matches_sklearn():
    import torch
    from sklearn.metrics import average_precision_score

    from pyhazards.benchmarks.wildfire import _spread_metrics

    torch.manual_seed(0)
    logits = torch.randn(3, 1, 8, 8)
    targets = (torch.rand(3, 1, 8, 8) > 0.8).float()
    expected = average_precision_score(targets.flatten().numpy(), torch.sigmoid(logits).flatten().numpy())
    assert abs(_spread_metrics(logits, targets)["average_precision"] - expected) < 1e-9


def test_added_wildfire_breadth_configs(tmp_path):
    expectations = {
        "pyhazards/configs/wildfire/wildfire_forecasting_smoke.yaml": "macro_f1",
        "pyhazards/configs/wildfire/asufm_smoke.yaml": "burned_area_mae",
        "pyhazards/configs/wildfire/wildfirespreadts_smoke.yaml": "burned_area_mae",
        "pyhazards/configs/wildfire/firecastnet_smoke.yaml": "burned_area_mae",
        "pyhazards/configs/wildfire/track_o_convlstm_smoke.yaml": "pr_auc",
    }
    for path, metric_name in expectations.items():
        summary = BenchmarkRunner().run(load_experiment_config(path), output_dir=str(tmp_path))
        assert metric_name in summary.metrics
