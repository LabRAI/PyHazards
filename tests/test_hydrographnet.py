"""HydroGraphNet (PhysicsNeMo MeshGraphKAN) and its mesh data: shapes, counts, validation, reader and rollout."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from pyhazards.configs import load_experiment_config
from pyhazards.datasets import load_dataset
from pyhazards.datasets.flood.hydrograph import (
    hydrograph_collate,
    read_hydrograph,
    read_static,
    synthetic_hydrograph_mesh,
    write_hydrograph_files,
)
from pyhazards.engine.runner import BenchmarkRunner
from pyhazards.models import build_model
from pyhazards.models.hydrographnet import HydroGraphNet, HydroGraphNetLoss, hydrographnet_physics_loss


def _n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _ring(num_nodes: int) -> torch.Tensor:
    src = torch.arange(num_nodes)
    return torch.stack([torch.cat([src, src]), torch.cat([(src + 1) % num_nodes, (src - 1) % num_nodes])])


def test_parameter_count_and_layout_at_hydrographnet_configuration():
    model = build_model("hydrographnet", task="regression")
    assert _n_params(model) == 2_318_722
    assert _n_params(model.edge_encoder) == 33_792
    assert _n_params(model.node_encoder) == 20_608
    assert _n_params(model.processor) == 2_231_040
    assert _n_params(model.node_decoder) == 33_282
    keys = list(model.state_dict())
    assert keys[:10] == [
        "edge_encoder.model.0.weight",
        "edge_encoder.model.0.bias",
        "edge_encoder.model.2.weight",
        "edge_encoder.model.2.bias",
        "edge_encoder.model.4.weight",
        "edge_encoder.model.4.bias",
        "edge_encoder.model.5.weight",  # LayerNorm
        "edge_encoder.model.5.bias",
        "node_encoder.fourier_coeffs",
        "node_encoder.bias",
    ]
    assert "processor.processor_layers.29.node_mlp.model.5.weight" in keys
    assert not any(key.startswith("node_decoder.model.5") for key in keys)  # no LayerNorm on the decoder


def test_forward_accepts_tensors_graph_objects_and_mappings():
    torch.manual_seed(0)
    model = HydroGraphNet(processor_size=2, hidden_dim_processor=16, hidden_dim_node_encoder=16,
                          hidden_dim_edge_encoder=16, hidden_dim_node_decoder=16).eval()
    nodes, edge_index = torch.randn(10, 16), _ring(10)
    edges = torch.randn(edge_index.shape[1], 3)

    class Graph:
        pass

    graph = Graph()
    graph.edge_index = edge_index
    with torch.no_grad():
        out = model(nodes, edges, edge_index)
        assert out.shape == (10, 2)
        torch.testing.assert_close(model(nodes, edges, graph), out)
        torch.testing.assert_close(model({"node_features": nodes, "edge_features": edges, "edge_index": edge_index}), out)


def test_sum_aggregation_is_not_mean():
    torch.manual_seed(0)
    summed = HydroGraphNet(processor_size=1, hidden_dim_processor=8, hidden_dim_node_encoder=8, hidden_dim_edge_encoder=8, hidden_dim_node_decoder=8)
    averaged = HydroGraphNet(processor_size=1, hidden_dim_processor=8, hidden_dim_node_encoder=8, hidden_dim_edge_encoder=8, hidden_dim_node_decoder=8, aggregation="mean")
    averaged.load_state_dict(summed.state_dict())
    nodes, edge_index = torch.randn(6, 16), _ring(6)
    edges = torch.randn(12, 3)
    with torch.no_grad():
        assert not torch.allclose(summed(nodes, edges, edge_index), averaged(nodes, edges, edge_index))


@pytest.mark.parametrize(
    "nodes, edges, edge_index",
    [
        (torch.randn(10, 15), torch.randn(20, 3), _ring(10)),
        (torch.randn(10, 16), torch.randn(20, 2), _ring(10)),
        (torch.randn(10, 16), torch.randn(19, 3), _ring(10)),
        (torch.randn(10, 16), torch.randn(20, 3), torch.zeros(3, 20, dtype=torch.long)),
    ],
)
def test_bad_inputs_raise(nodes, edges, edge_index):
    model = HydroGraphNet(processor_size=1, hidden_dim_processor=8, hidden_dim_node_encoder=8, hidden_dim_edge_encoder=8, hidden_dim_node_decoder=8)
    with pytest.raises(ValueError, match="shape|edge"):
        model(nodes, edges, edge_index)


def test_builder_validation():
    with pytest.raises(ValueError, match="regression"):
        build_model("hydrographnet", task="segmentation")
    with pytest.raises(ValueError, match="Unknown"):
        build_model("hydrographnet", task="regression", hidden_dim=64)
    with pytest.raises(ValueError, match="aggregation"):
        HydroGraphNet(aggregation="max")
    with pytest.raises(ValueError, match="mlp_activation_fn"):
        HydroGraphNet(mlp_activation_fn="swishy")


def test_rollout_updates_windows_and_forcings():
    torch.manual_seed(0)
    model = HydroGraphNet(processor_size=1, hidden_dim_processor=8, hidden_dim_node_encoder=8, hidden_dim_edge_encoder=8, hidden_dim_node_decoder=8).eval()
    nodes, edge_index = torch.randn(7, 16), _ring(7)
    edges = torch.randn(14, 3)
    inflow, precipitation = torch.randn(5), torch.randn(5)
    out = model.rollout(nodes, edges, edge_index, inflow, precipitation)
    assert out["water_depth"].shape == out["volume"].shape == (5, 7)
    with torch.no_grad():
        first = model(nodes, edges, edge_index)
    torch.testing.assert_close(out["water_depth"][0], nodes[:, 13] + first[:, 0])
    torch.testing.assert_close(out["volume"][0], nodes[:, 15] + first[:, 1])
    with pytest.raises(ValueError, match="shape"):
        model.rollout(nodes[:, :15], edges, edge_index, inflow, precipitation)


def test_physics_loss_is_zero_for_volume_inside_the_continuity_bounds():
    physics = {
        "past_volume": torch.tensor([0.0]),
        "future_volume": torch.tensor([0.0]),
        "avg_inflow": torch.tensor([10.0]),
        "avg_precipitation": torch.tensor([0.0]),
        "next_inflow": torch.tensor([0.0]),
        "next_precip": torch.tensor([0.0]),
        "volume_mean": torch.tensor([0.0]),
        "volume_std": torch.tensor([1.0]),
        "num_nodes": torch.tensor([4.0]),
        "area_sum": torch.tensor([100.0]),
        "infiltration_area_sum": torch.tensor([1.0]),
    }
    pred = torch.zeros(4, 2)
    assert float(hydrographnet_physics_loss(pred, physics, delta_t=1.0)) == 0.0
    pred[:, 1] = 10.0  # 40 m3 gained but only 10 m3 entered: term1 = ((40 - 10) / 100)^2
    assert float(hydrographnet_physics_loss(pred, physics, delta_t=1.0)) == pytest.approx(0.3**2)
    loss, parts = HydroGraphNetLoss(physics_loss_weight=2.0, delta_t=1.0)(pred, torch.zeros(4, 2), physics)
    assert float(parts["physics_loss"]) == pytest.approx(0.3**2)
    assert float(loss) == pytest.approx(float(parts["mse_loss"]) + 2.0 * 0.3**2)
    loss, parts = HydroGraphNetLoss(use_physics_loss=False)(pred, torch.zeros(4, 2), physics)
    assert set(parts) == {"total_loss", "mse_loss"}
    with pytest.raises(ValueError, match="missing"):
        hydrographnet_physics_loss(pred, {"past_volume": torch.tensor([0.0])})


def _write_fixture(tmp_path, test_ids=("H4",)):
    static, series = synthetic_hydrograph_mesh(nodes=20, hydrographs=4, steps=50, seed=1)
    write_hydrograph_files(tmp_path, static, series, extra_cells=3, train_ids=["H1", "H2", "H3", "H4"])
    return static, series


def test_reader_follows_release_layout(tmp_path):
    static, series = _write_fixture(tmp_path)
    raw = read_static(tmp_path)
    assert raw["xy_coords"].shape == (20, 2) and raw["area"].shape == (20, 1)
    np.testing.assert_allclose(raw["elevation"][:, 0], static["elevation"][:, 0], atol=1e-8)
    hydrograph = read_hydrograph(tmp_path, "H1", num_points=20)
    peak = int(np.argmax(series["H1"]["inflow_hydrograph"]))
    assert hydrograph["water_depth"].shape == (min(50, peak + 25), 20)
    np.testing.assert_allclose(hydrograph["precipitation"], series["H1"]["precipitation"][: peak + 25], rtol=1e-6, atol=1e-12)

    with pytest.raises(ValueError, match="test_ids"):
        load_dataset("hydrographnet_white_river", data_dir=str(tmp_path)).load()  # train.txt lists every hydrograph
    bundle = load_dataset("hydrographnet_white_river", data_dir=str(tmp_path), test_ids=["H4"], rollout_length=10).load()
    assert bundle.metadata["train_ids"] == ["H1", "H2", "H3"] and bundle.metadata["test_ids"] == ["H4"]
    windows = bundle.get_split("train").inputs
    inputs, target = windows[0]
    assert inputs["node_features"].shape == (20, 16)
    assert inputs["edge_index"].shape == (2, 80) and inputs["edge_features"].shape == (80, 3)
    assert target.shape == (20, 2)
    assert {"past_volume", "future_volume", "area_sum", "infiltration_area_sum"} <= set(inputs["physics"])
    rollouts = bundle.get_split("test").inputs
    inputs, targets = rollouts[0]
    assert inputs["inflow"].shape == (10,) and targets["water_depth"].shape == (10, 20)

    (tmp_path / "test.txt").write_text("H4\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_dataset("hydrographnet_white_river", data_dir=str(tmp_path), test_ids="test.txt", norm_stats="files").load()
    for name, stats in (("static_norm_stats.json", bundle.metadata["static_stats"]), ("dynamic_norm_stats.json", bundle.metadata["dynamic_stats"])):
        (tmp_path / name).write_text(json.dumps(stats), encoding="utf-8")
    files = load_dataset("hydrographnet_white_river", data_dir=str(tmp_path), test_ids="test.txt", norm_stats="files", rollout_length=10).load()
    assert files.metadata["dynamic_stats"] == bundle.metadata["dynamic_stats"]
    with pytest.raises(ValueError, match="data_dir"):
        load_dataset("hydrographnet_white_river")


def test_collate_offsets_edges_and_stacks_physics():
    bundle = load_dataset("flood_mesh_synthetic", micro=True).load()
    windows = bundle.get_split("train").inputs
    batch, target = hydrograph_collate([windows[0], windows[3]])
    n = windows[0][0]["node_features"].shape[0]
    assert batch["node_features"].shape == (2 * n, 16) and target.shape == (2 * n, 2)
    assert int(batch["edge_index"].max()) == 2 * n - 1
    assert batch["physics"]["past_volume"].shape == (2,) and batch["physics"]["past_volume"].dtype == torch.float32
    assert torch.equal(batch["batch"], torch.repeat_interleave(torch.arange(2), n))
    model = HydroGraphNet(processor_size=1, hidden_dim_processor=8, hidden_dim_node_encoder=8, hidden_dim_edge_encoder=8, hidden_dim_node_decoder=8)
    pred = model(batch)
    loss, parts = HydroGraphNetLoss()(pred, target, batch["physics"], batch["batch"])
    loss.backward()
    assert torch.isfinite(loss) and set(parts) == {"total_loss", "mse_loss", "physics_loss"}


def test_synthetic_mesh_bundle_is_marked_synthetic():
    bundle = load_dataset("flood_mesh_synthetic", micro=True).load()
    assert bundle.metadata["synthetic"] is True
    assert bundle.metadata["inundation_layout"] == "mesh_rollout"
    assert set(bundle.splits) == {"train", "test"}


def test_smoke_config_scores_rollouts_in_metres(tmp_path):
    summary = BenchmarkRunner().run(load_experiment_config("pyhazards/configs/flood/hydrographnet_smoke.yaml"), output_dir=str(tmp_path))
    assert summary.hazard_task == "flood.inundation"
    for metric in ("rollout_rmse", "pixel_mae", "rmse", "csi_1cm", "csi_10cm", "csi_50cm", "relative_l2", "nse", "pearson_r"):
        assert metric in summary.metrics
    assert len(summary.metadata["rollout_rmse_per_step_m"]) == 6
    assert summary.metrics["rollout_rmse"] == pytest.approx(np.mean(summary.metadata["rollout_rmse_per_step_m"]))
