import torch

from pyhazards.configs import load_experiment_config
from pyhazards.engine.runner import BenchmarkRunner
from pyhazards.models import build_model


def test_storm_breadth_models_forward():
    x = torch.randn(2, 6, 8)
    model_names = [
        "tropicalcyclone_mlp",
        "tropicyclonenet",
        "saf_net",
        "tcif_fusion",
        "graphcast_tc",
        "pangu_tc",
        "fourcastnet_tc",
    ]
    for name in model_names:
        kwargs = {
            "input_dim": 8,
            "horizon": 5,
            "output_dim": 3,
        }
        if name in {"tropicalcyclone_mlp", "fourcastnet_tc"}:
            kwargs["history"] = 6
        model = build_model(name=name, task="regression", **kwargs)
        preds = model(x)
        assert preds.shape == (2, 5, 3)


def test_tc_breadth_configs(tmp_path):
    for config_name in [
        "pyhazards/configs/tc/tropicalcyclone_mlp_smoke.yaml",
        "pyhazards/configs/tc/tropicyclonenet_smoke.yaml",
        "pyhazards/configs/tc/saf_net_smoke.yaml",
        "pyhazards/configs/tc/tcif_fusion_smoke.yaml",
        "pyhazards/configs/tc/graphcast_tc_smoke.yaml",
        "pyhazards/configs/tc/pangu_tc_smoke.yaml",
        "pyhazards/configs/tc/fourcastnet_tc_smoke.yaml",
    ]:
        summary = BenchmarkRunner().run(
            load_experiment_config(config_name),
            output_dir=str(tmp_path),
        )
        assert summary.hazard_task == "tc.track_intensity"
        assert "track_error" in summary.metrics
        assert "intensity_mae" in summary.metrics
