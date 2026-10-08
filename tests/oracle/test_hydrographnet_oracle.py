"""HydroGraphNet checked against NVIDIA PhysicsNeMo (pinned in repos.yaml, Apache-2.0).

The oracle is PhysicsNeMo's own code, imported from the pinned checkout: ``MeshGraphKAN`` (with PyTorch
Geometric graphs and ``torch_scatter`` aggregation), the HydroGraphNet example's
``compute_physics_loss`` (``examples/weather/flood_modeling/hydrographnet/utils.py``) and rollout loop
(``inference.py``; the loop is executed from the file's own source because ``main`` is a Hydra entry
point that also loads a checkpoint and renders animations), and ``HydroGraphDataset``. Data fixtures are
written in the White River release layout (Zenodo 14969507). With ``PYHAZARDS_HYDROGRAPHNET_DATA``
pointing at a local copy of the release, the reader is also compared on real hydrographs.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from oracle_utils import import_from, missing, oracle_repo
from pyhazards.datasets import load_dataset
from pyhazards.datasets.flood.hydrograph import (
    hydrograph_collate,
    knn_edge_index,
    synthetic_hydrograph_mesh,
    write_hydrograph_files,
)
from pyhazards.models import build_model
from pyhazards.models.hydrographnet import HydroGraphNet, HydroGraphNetLoss, hydrographnet_physics_loss

EXAMPLE = Path("examples") / "weather" / "flood_modeling" / "hydrographnet"
SMALL = dict(
    input_dim_nodes=5,
    input_dim_edges=2,
    output_dim=3,
    processor_size=3,
    mlp_activation_fn="silu",
    num_layers_node_processor=3,
    num_layers_edge_processor=1,
    hidden_dim_processor=32,
    hidden_dim_node_encoder=48,
    num_layers_node_encoder=1,
    hidden_dim_edge_encoder=24,
    num_layers_edge_encoder=1,
    hidden_dim_node_decoder=40,
    num_layers_node_decoder=3,
    aggregation="mean",
    num_harmonics=7,
)


def _pyg():
    try:
        import torch_geometric  # noqa: F401
        import torch_scatter  # noqa: F401
    except ImportError:
        missing("PhysicsNeMo's MeshGraphKAN needs torch_geometric and torch_scatter; pip install -r tests/oracle/requirements-physicsnemo.txt")
    import torch_geometric as pyg

    return pyg


def _meshgraphkan():
    _pyg()
    os.environ.setdefault("PHYSICSNEMO_FORCE_TE", "False")
    return import_from(oracle_repo("physicsnemo"), "physicsnemo.models.meshgraphnet.meshgraphkan").MeshGraphKAN


def _hydrograph_dataset_class():
    _pyg()
    return import_from(oracle_repo("physicsnemo"), "physicsnemo.datapipes.gnn.hydrographnet_dataset").HydroGraphDataset


def _graph(num_nodes: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(knn_edge_index(rng.uniform(0, 1, (num_nodes, 2)), k=4)).long()


def _assert_same_state(reference, port) -> None:
    ref_state, port_state = reference.state_dict(), port.state_dict()
    assert list(ref_state) == list(port_state)
    for key, value in ref_state.items():
        assert value.shape == port_state[key].shape and value.dtype == port_state[key].dtype, key
        assert torch.equal(value, port_state[key]), key


@pytest.mark.parametrize("config", [dict(input_dim_nodes=16, input_dim_edges=3, output_dim=2), SMALL], ids=["hydrographnet", "small_mean_silu"])
def test_parameters_names_and_seeded_initialisation_match_physicsnemo(config):
    MeshGraphKAN = _meshgraphkan()
    torch.manual_seed(0)
    reference = MeshGraphKAN(**config)
    torch.manual_seed(0)
    port = HydroGraphNet(**config)
    n_ref = sum(p.numel() for p in reference.parameters())
    assert n_ref == sum(p.numel() for p in port.parameters())
    if config["input_dim_nodes"] == 16:
        assert n_ref == 2_318_722
        torch.manual_seed(0)
        _assert_same_state(reference, build_model("hydrographnet", task="regression"))
    _assert_same_state(reference, port)


@pytest.mark.parametrize("config", [dict(input_dim_nodes=16, input_dim_edges=3, output_dim=2), SMALL], ids=["hydrographnet", "small_mean_silu"])
def test_outputs_and_gradients_match_physicsnemo(config):
    pyg = _pyg()
    MeshGraphKAN = _meshgraphkan()
    torch.manual_seed(0)
    reference = MeshGraphKAN(**config)
    port = HydroGraphNet(**config)
    port.load_state_dict(reference.state_dict(), strict=True)
    edge_index = _graph()
    graph = pyg.data.Data(edge_index=edge_index, num_nodes=200)
    generator = torch.Generator().manual_seed(1)
    nodes = torch.randn(200, config["input_dim_nodes"], generator=generator)
    edges = torch.randn(edge_index.shape[1], config["input_dim_edges"], generator=generator)
    for train in (False, True):
        reference.train(train)
        port.train(train)
        expected = reference(nodes, edges, graph)
        actual = port(nodes, edges, edge_index)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(port(nodes, edges, graph), expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            port({"node_features": nodes, "edge_features": edges, "edge_index": edge_index}), expected, rtol=1e-5, atol=1e-6
        )
    reference.zero_grad()
    port.zero_grad()
    reference(nodes, edges, graph).square().sum().backward()
    port(nodes, edges, edge_index).square().sum().backward()
    # Multi-threaded CPU sums make PhysicsNeMo's own gradients differ between two identical calls by up to
    # ~5e-7 of the largest entry, so near-zero entries need an absolute tolerance scaled to each tensor.
    for (name, ref_param), port_param in zip(reference.named_parameters(), port.parameters()):
        scale = float(ref_param.grad.abs().max())
        torch.testing.assert_close(port_param.grad, ref_param.grad, rtol=1e-4, atol=1e-5 * scale, msg=name)


def test_batched_graphs_match_pyg_batching():
    pyg = _pyg()
    MeshGraphKAN = _meshgraphkan()
    torch.manual_seed(0)
    reference = MeshGraphKAN(16, 3, 2).eval()
    port = HydroGraphNet().eval()
    port.load_state_dict(reference.state_dict(), strict=True)
    samples, graphs = [], []
    for seed, n in ((0, 40), (1, 25)):
        edge_index = _graph(n, seed)
        generator = torch.Generator().manual_seed(seed)
        nodes, edges = torch.randn(n, 16, generator=generator), torch.randn(edge_index.shape[1], 3, generator=generator)
        samples.append(({"node_features": nodes, "edge_features": edges, "edge_index": edge_index}, torch.zeros(n, 2)))
        graphs.append(pyg.data.Data(x=nodes, edge_attr=edges, edge_index=edge_index))
    batch = pyg.data.Batch.from_data_list(graphs)
    collated, _ = hydrograph_collate(samples)
    assert torch.equal(collated["edge_index"], batch.edge_index)
    assert torch.equal(collated["batch"], batch.batch)
    with torch.no_grad():
        torch.testing.assert_close(port(collated), reference(batch.x, batch.edge_attr, batch), rtol=1e-5, atol=1e-6)


def test_physicsnemo_stored_regression_output_is_reproduced():
    """PhysicsNeMo's test_meshgraphkan_forward (MeshGraphKAN(4, 3, 2), seed 0) against its stored output."""
    pyg = _pyg()
    MeshGraphKAN = _meshgraphkan()
    repo = oracle_repo("physicsnemo")
    stored = list(torch.load(repo / "test" / "models" / "meshgraphnet" / "data" / "meshgraphkan_output.pth").values())[0]

    def run(cls):
        torch.manual_seed(0)
        np.random.seed(0)
        model = cls(4, 3, 2)
        graphs = []
        for _ in range(2):
            src = torch.tensor([np.random.randint(20) for _ in range(12)])
            dst = torch.tensor([np.random.randint(20) for _ in range(12)])
            graphs.append(pyg.data.Data(edge_index=torch.stack([src, dst], dim=0), num_nodes=20))
        graph = pyg.data.Batch.from_data_list(graphs)
        with torch.no_grad():
            return model(torch.randn(graph.num_nodes, 4), torch.randn(graph.num_edges, 3), graph)

    port_output = run(HydroGraphNet)
    torch.testing.assert_close(port_output, run(MeshGraphKAN), rtol=1e-5, atol=1e-6)
    # PhysicsNeMo itself checks this file with rtol = atol = 0.1 (outputs differ across devices).
    torch.testing.assert_close(port_output, stored, rtol=1e-4, atol=1e-5)


def _fixture(tmp_path: Path, nodes: int = 30, hydrographs: int = 4, steps: int = 60) -> Path:
    static, series = synthetic_hydrograph_mesh(nodes=nodes, hydrographs=hydrographs, steps=steps, seed=3)
    ids = list(series)
    write_hydrograph_files(tmp_path, static, series, extra_cells=5, train_ids=ids[:-1])
    (tmp_path / "test.txt").write_text(f"{ids[-1]}\n", encoding="utf-8")
    return tmp_path


def _compare_reader(official_cls, data_dir: Path, rollout_length: int, every: int = 1) -> None:
    official = official_cls(
        data_dir=str(data_dir), prefix="M80", n_time_steps=2, k=4, hydrograph_ids_file="train.txt", split="train", return_physics=True
    )
    bundle = load_dataset("hydrographnet_white_river", data_dir=str(data_dir), test_ids="test.txt", rollout_length=rollout_length).load()
    windows = bundle.get_split("train").inputs
    assert len(windows) == len(official)
    for idx in list(range(0, len(official), every)) + [len(official) - 1]:
        graph, physics = official[idx]
        inputs, target = windows[idx]
        assert torch.equal(inputs["node_features"], graph.x)
        assert torch.equal(inputs["edge_index"], graph.edge_index)
        assert torch.equal(inputs["edge_features"], graph.edge_attr)
        assert torch.equal(target, graph.y)
        assert set(inputs["physics"]) == set(physics)
        for key, value in physics.items():
            assert float(inputs["physics"][key]) == float(value), key
    # The test split of the example reads the normalisation statistics its training run wrote to data_dir.
    official_test = official_cls(
        data_dir=str(data_dir), prefix="M80", n_time_steps=2, hydrograph_ids_file="test.txt", split="test", rollout_length=rollout_length
    )
    rollouts = bundle.get_split("test").inputs
    assert len(rollouts) == len(official_test)
    for idx in range(len(official_test)):
        graph, rollout_data = official_test[idx]
        inputs, targets = rollouts[idx]
        assert torch.equal(inputs["node_features"], graph.x)
        assert torch.equal(inputs["edge_features"], graph.edge_attr)
        assert torch.equal(inputs["inflow"], rollout_data["inflow"])
        assert torch.equal(inputs["precipitation"], rollout_data["precipitation"])
        assert torch.equal(targets["water_depth"], rollout_data["water_depth_gt"])
        assert torch.equal(targets["volume"], rollout_data["volume_gt"])
    files = load_dataset(
        "hydrographnet_white_river", data_dir=str(data_dir), test_ids="test.txt", rollout_length=rollout_length, norm_stats="files"
    ).load()
    assert files.metadata["dynamic_stats"] == bundle.metadata["dynamic_stats"]


def test_reader_matches_hydrographdataset_on_release_layout(tmp_path):
    _compare_reader(_hydrograph_dataset_class(), _fixture(tmp_path), rollout_length=20)


def test_reader_matches_hydrographdataset_on_real_white_river(tmp_path):
    """Local check on the real release: set PYHAZARDS_HYDROGRAPHNET_DATA to the unzipped HydroGraphNet.zip."""
    root = os.environ.get("PYHAZARDS_HYDROGRAPHNET_DATA")
    if not root:
        pytest.skip("PYHAZARDS_HYDROGRAPHNET_DATA is not set (real White River data, Zenodo 14969507, 8.3 GB)")
    official_cls = _hydrograph_dataset_class()
    source = Path(root)
    # The example writes normalisation statistics into data_dir; work on links to the real files.
    for code in ("XY", "CA", "CE", "CS", "A", "CU", "N", "FA", "IP"):
        os.symlink(source / f"M80_{code}.txt", tmp_path / f"M80_{code}.txt")
    for hid in ("H1", "H2"):
        for code in ("WD", "V", "US_InF", "Pr"):
            os.symlink(source / f"M80_{code}_{hid}.txt", tmp_path / f"M80_{code}_{hid}.txt")
    (tmp_path / "train.txt").write_text("H1\n", encoding="utf-8")
    (tmp_path / "test.txt").write_text("H2\n", encoding="utf-8")
    _compare_reader(official_cls, tmp_path, rollout_length=30, every=7)


def _inference_loop():
    """The ``for t in range(rollout_length)`` loop of the example's inference.py, from its own source."""
    path = oracle_repo("physicsnemo") / EXAMPLE / "inference.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    loops = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "t"
    ]
    assert len(loops) == 1, "inference.py rollout loop not found"
    return compile(ast.Module(body=loops, type_ignores=[]), str(path), "exec")


def test_rollout_matches_inference_script(tmp_path):
    MeshGraphKAN = _meshgraphkan()
    official_cls = _hydrograph_dataset_class()
    data_dir = _fixture(tmp_path)
    official_cls(data_dir=str(data_dir), prefix="M80", n_time_steps=2, hydrograph_ids_file="train.txt", split="train")
    test_set = official_cls(
        data_dir=str(data_dir), prefix="M80", n_time_steps=2, hydrograph_ids_file="test.txt", split="test", rollout_length=20
    )
    torch.manual_seed(0)
    reference = MeshGraphKAN(16, 3, 2).eval()
    port = HydroGraphNet().eval()
    port.load_state_dict(reference.state_dict(), strict=True)
    g, rollout_data = test_set[0]
    scope = {
        "torch": torch,
        "model": reference,
        "g": g,
        "edge_features": g.edge_attr,
        "X_iter": g.x.clone(),
        "num_nodes": g.x.size(0),
        "n_time_steps": 2,
        "rollout_length": 20,
        "inflow_seq": rollout_data["inflow"],
        "precip_seq": rollout_data["precipitation"],
        "wd_gt_seq": rollout_data["water_depth_gt"],
        "rollout_preds": [],
        "ground_truth_list": [],
        "rmse_list": [],
    }
    with torch.no_grad():
        exec(_inference_loop(), scope)
    out = port.rollout(g.x, g.edge_attr, g.edge_index, rollout_data["inflow"], rollout_data["precipitation"], n_time_steps=2)
    torch.testing.assert_close(out["water_depth"], torch.stack(scope["rollout_preds"]), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out["volume"][-1].unsqueeze(1), scope["X_iter"][:, -1:], rtol=1e-5, atol=1e-6)
    from pyhazards.metrics.inundation import rollout_rmse

    np.testing.assert_allclose(rollout_rmse(out["water_depth"], rollout_data["water_depth_gt"]).numpy(), scope["rmse_list"], rtol=1e-5, atol=1e-7)


def test_physics_loss_matches_example(tmp_path):
    pyg = _pyg()
    compute_physics_loss = import_from(oracle_repo("physicsnemo") / EXAMPLE, "utils").compute_physics_loss
    bundle = load_dataset("hydrographnet_white_river", data_dir=str(_fixture(tmp_path)), test_ids="test.txt", rollout_length=20).load()
    windows = bundle.get_split("train").inputs
    samples = [windows[i] for i in (0, 7, 30, 61)]
    collated, target = hydrograph_collate(samples)
    batch = pyg.data.Batch.from_data_list(
        [pyg.data.Data(x=s[0]["node_features"], edge_index=s[0]["edge_index"], num_nodes=s[0]["node_features"].shape[0]) for s in samples]
    )
    generator = torch.Generator().manual_seed(0)
    for scale in (0.01, 1.0, 50.0):
        pred = scale * torch.randn(target.shape, generator=generator)
        expected = compute_physics_loss(pred, collated["physics"], batch, delta_t=1200.0)
        torch.testing.assert_close(hydrographnet_physics_loss(pred, collated["physics"], collated["batch"]), expected, rtol=1e-5, atol=0)
        loss, parts = HydroGraphNetLoss()(pred, target, collated["physics"], collated["batch"])
        torch.testing.assert_close(loss, torch.nn.functional.mse_loss(pred, target) + expected, rtol=1e-5, atol=0)
        assert set(parts) == {"total_loss", "mse_loss", "physics_loss"}
