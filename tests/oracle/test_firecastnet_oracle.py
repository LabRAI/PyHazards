"""FireCastNet checked against the official SeasFire/firecastnet code (PyTorch + Lightning + DGL).

The reference is pinned in repos.yaml and needs the packages of requirements-firecastnet.txt
(DGL 2.4 with CPU torch 2.4); the Oracle workflow runs this file in its own job. Checks:

- the generated icospheres equal the official ``icospheres/*.json.gz`` files bit for bit;
- the multi-mesh, grid->mesh and mesh->grid graphs equal those of the official ``GraphBuilder``
  (same edge order, bitwise-equal edge and node features), for the global icosphere mesh and a LAM mesh;
- a small global configuration: same seeded initial weights and parameter names, same outputs and
  gradients; and the other supported options (mean aggregation, deeper MLPs, no static channels, no
  cube LayerNorm, several output channels);
- the official global (ts24, h1) classification and regression checkpoints and the AUST local-area
  checkpoint (Hugging Face d-michail/firecastnet-artifacts, Apache-2.0) load with ``strict=True``
  and give the same outputs as the official model on the full 720 x 1440 grid.

The official LightningModule reads a SeasFire zarr cube at construction (GFED regions, land-sea
mask); a two-cell stand-in cube is written for it, with the land-sea filter and GFED loss weights
off. Neither affects the network output.
"""

from __future__ import annotations

import importlib.util
import os
import re

import numpy as np
import pytest
import torch

from oracle_utils import import_from, missing, oracle_asset, oracle_repo
from pyhazards.models.firecastnet import FireCastNet, load_official_checkpoint
from pyhazards.models.firecastnet_graph import build_graph, icospheres, latlon_grid, load_icospheres

GLOBAL_PARAMETERS = 9_087_568
SMALL_GLOBAL = dict(sp_res=1.25, max_lat=89.375, min_lat=-89.375, max_lon=179.375, min_lon=-179.375)
SMALL_GLOBAL_EMBED = dict(
    embed_cube_sp_res=5.0, embed_cube_max_lat=87.5, embed_cube_min_lat=-87.5, embed_cube_max_lon=177.5, embed_cube_min_lon=-177.5
)


def _require_reference_packages() -> None:
    for package in ("dgl", "lightning", "sklearn", "einops", "zarr", "torchmetrics"):
        if importlib.util.find_spec(package) is None:
            missing(f"{package} is not installed; pip install -r tests/oracle/requirements-firecastnet.txt")
    os.environ.setdefault("DGLBACKEND", "pytorch")  # otherwise DGL writes ~/.dgl/config.json


def _official(module: str):
    repo = oracle_repo("firecastnet")
    _require_reference_packages()
    return repo, import_from(repo, f"seasfire.{module}")


def _stand_in_cube(tmp_path) -> str:
    import xarray as xr

    path = tmp_path / "cube.zarr"
    xr.Dataset({"gfed_region": (("latitude", "longitude"), np.zeros((2, 2), dtype=np.float32))}).to_zarr(path, consolidated=False)
    return str(path)


def _official_model(tmp_path, **hparams):
    repo, lit = _official("firecastnet_lit")
    hparams["icospheres_graph_path"] = str(repo / hparams["icospheres_graph_path"])
    hparams.update(
        cube_path=_stand_in_cube(tmp_path),
        lsm_filter_enable=False,
        gfed_region_enable_loss_weighting=False,
        display_model_example=False,
    )
    return lit.FireCastNetLit(**hparams)


def _official_logits(reference, x: torch.Tensor) -> torch.Tensor:
    """Official forward for PyHazards input (B=1, T, C, H, W); the reference takes (1, C, T, H, W)."""
    prepared, _, _ = reference._prepare_data(x.transpose(1, 2).contiguous(), None, None)
    return reference(prepared)


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, **tol) -> None:
    torch.testing.assert_close(actual, expected, **{"rtol": 1e-5, "atol": 1e-6, **tol})


def test_icospheres_match_official_assets():
    repo = oracle_repo("firecastnet")
    files = sorted((repo / "icospheres").glob("icospheres_*.json.gz"))
    assert len(files) == 9
    for path in files:
        levels = [int(level) for level in re.findall(r"\d+", path.name)]
        official = load_icospheres(path)
        generated = icospheres(levels)
        assert sorted(official) == sorted(generated), path.name
        for key, value in official.items():
            assert np.array_equal(generated[key], value), (path.name, key)


def _assert_graph_equals_builder(builder, graph) -> None:
    for name, dgl_graph in (
        ("mesh", builder.create_mesh_graph()),
        ("g2m", builder.create_g2m_graph()),
        ("m2g", builder.create_m2g_graph()),
    ):
        src, dst = dgl_graph.edges()
        assert torch.equal(getattr(graph, f"{name}_src"), src.long()), name
        assert torch.equal(getattr(graph, f"{name}_dst"), dst.long()), name
        assert torch.equal(getattr(graph, f"{name}_edata"), dgl_graph.edata["x"]), name
        if name == "mesh":
            assert torch.equal(graph.mesh_ndata, dgl_graph.ndata["x"])
            assert graph.num_mesh_nodes == dgl_graph.num_nodes()


def test_global_graph_matches_official_graph_builder():
    repo, graph_builder = _official("backbones.graphcast.graph_builder")
    latitudes, longitudes = latlon_grid(1.0, 89.5, -89.5, 179.5, -179.5)
    grid = torch.stack(torch.meshgrid(latitudes, longitudes, indexing="ij"), dim=-1)
    builder = graph_builder.GraphBuilder(str(repo / "icospheres" / "icospheres_0_1_2_3_4_5_6.json.gz"), grid)
    graph = build_graph(latitudes, longitudes, icospheres(range(7)))
    assert (graph.num_grid_nodes, graph.num_mesh_nodes) == (64_800, 40_962)
    assert (graph.mesh_src.numel(), graph.g2m_src.numel(), graph.m2g_src.numel()) == (327_660, 100_160, 64_800)
    _assert_graph_equals_builder(builder, graph)


def test_lam_graph_matches_official_graph_builder():
    repo, graph_builder = _official("backbones.graphcast.graph_builder")
    mesh = repo / "icospheres-lam" / "icospheres" / "icosphere_s3_AUST_6u.json.gz"
    latitudes, longitudes = latlon_grid(1.0, 89.5, -89.5, 179.5, -179.5)
    grid = torch.stack(torch.meshgrid(latitudes, longitudes, indexing="ij"), dim=-1)
    graph = build_graph(latitudes, longitudes, load_icospheres(mesh))
    assert graph.num_mesh_nodes == 2002
    _assert_graph_equals_builder(graph_builder.GraphBuilder(str(mesh), grid), graph)


def test_small_global_configuration_matches_reference(tmp_path):
    torch.manual_seed(0)
    reference = _official_model(
        tmp_path,
        icospheres_graph_path="icospheres/icospheres_0_1_2_3.json.gz",
        embed_cube=True,
        embed_cube_time=6,
        embed_cube_dim=64,
        timeseries_len=6,
        input_dim_grid_nodes=11,
        output_dim_grid_nodes=16,
        processor_layers=12,
        hidden_dim=64,
        **SMALL_GLOBAL,
        **SMALL_GLOBAL_EMBED,
    )
    torch.manual_seed(0)
    port = FireCastNet(in_channels=11, timeseries_len=6, mesh_levels=(0, 1, 2, 3), **SMALL_GLOBAL)

    ref_state, port_state = reference._net.state_dict(), port.state_dict()
    assert list(ref_state) == list(port_state)
    for key, value in ref_state.items():
        assert torch.equal(value, port_state[key]), key
    assert [key[len("_net."):] for key in reference.state_dict()] == list(port_state)
    assert _n_params(port) == _n_params(reference._net) == 866_896

    torch.manual_seed(1)
    x = torch.randn(1, 6, 11, 144, 288)
    reference.eval()
    port.eval()
    with torch.no_grad():
        _assert_close(port(x), _official_logits(reference, x))

    # No dropout or batch norm: train mode gives the same outputs; gradients must agree too.
    reference.train()
    port.train()
    target = (torch.rand(1, 1, 144, 288) > 0.9).float()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    ref_out = _official_logits(reference, x)
    out = port(x)
    _assert_close(out, ref_out)
    loss_fn(ref_out, target).backward()
    loss_fn(out, target).backward()
    ref_params = dict(reference._net.named_parameters())
    for name, param in port.named_parameters():
        assert param.grad is not None and ref_params[name].grad is not None, name
        _assert_close(param.grad, ref_params[name].grad, rtol=1e-4, atol=1e-7)


def test_configuration_options_match_reference(tmp_path):
    """Mean aggregation, two hidden layers, no static channels, no cube LayerNorm, two output channels."""
    options = dict(aggregation="mean", hidden_layers=2, hidden_dim=32, processor_layers=3, output_dim_grid_nodes=32)
    torch.manual_seed(0)
    reference = _official_model(
        tmp_path,
        icospheres_graph_path="icospheres/icospheres_0_1_2_3.json.gz",
        lat_lon_static_data=False,
        embed_cube=True,
        embed_cube_time=3,
        embed_cube_dim=16,
        embed_cube_layer_norm=False,
        timeseries_len=3,
        input_dim_grid_nodes=5,
        **options,
        **SMALL_GLOBAL,
        **SMALL_GLOBAL_EMBED,
    )
    torch.manual_seed(0)
    port = FireCastNet(
        in_channels=5, timeseries_len=3, mesh_levels=(0, 1, 2, 3), lat_lon_static_data=False, embed_cube_dim=16,
        embed_cube_layer_norm=False, **options, **SMALL_GLOBAL,
    )
    ref_state, port_state = reference._net.state_dict(), port.state_dict()
    assert list(ref_state) == list(port_state)
    for key, value in ref_state.items():
        assert torch.equal(value, port_state[key]), key

    torch.manual_seed(1)
    x = torch.randn(1, 3, 5, 144, 288)
    reference.eval()
    port.eval()
    with torch.no_grad():
        out = port(x)
        assert out.shape == (1, 2, 144, 288)
        _assert_close(out, _official_logits(reference, x))


def _checkpoint_hparams(path) -> dict:
    return dict(torch.load(path, map_location="cpu", weights_only=True)["hyper_parameters"])


def _compare_checkpoint(tmp_path, checkpoint, icospheres_graph_path=None) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reference = _official_model(tmp_path, **dict(state["hyper_parameters"]))
    reference.load_state_dict({k: v for k, v in state["state_dict"].items() if k.startswith("_net.")}, strict=True)
    port = load_official_checkpoint(checkpoint, icospheres_graph_path=icospheres_graph_path)
    assert list(port.state_dict()) == list(reference._net.state_dict())
    assert _n_params(port) == GLOBAL_PARAMETERS
    del state

    torch.manual_seed(0)
    x = torch.randn(1, 24, 11, 720, 1440)
    reference.eval()
    port.eval()
    with torch.no_grad():
        logits = port(x)
        _assert_close(logits, _official_logits(reference, x))
        # predict_step: sigmoid for classification, raw values for regression
        predictions = reference.predict_step({"x": x.transpose(1, 2).contiguous()})
        expected = logits[:, 0] if reference._task == "regression" else torch.sigmoid(logits[:, 0])
        _assert_close(expected, predictions)


@pytest.mark.parametrize(
    "asset, filename, task",
    [
        ("firecastnet_global_cls_ts24_h1", "firecastnet-cls-ts24-h1.ckpt", "classification"),
        ("firecastnet_global_regr_ts24_h1", "firecastnet-regr-ts24-h1.ckpt", "regression"),
    ],
)
def test_official_global_checkpoint_matches_reference(tmp_path, asset, filename, task):
    _official("firecastnet_lit")
    checkpoint = oracle_asset(asset) / filename
    hparams = _checkpoint_hparams(checkpoint)
    assert hparams["icospheres_graph_path"] == "icospheres/icospheres_0_1_2_3_4_5_6.json.gz"
    assert hparams.get("task", "classification") == task
    assert (hparams["timeseries_len"], hparams["embed_cube_time"], hparams["processor_layers"], hparams["hidden_dim"]) == (24, 24, 12, 64)
    _compare_checkpoint(tmp_path, checkpoint)


def test_official_lam_checkpoint_matches_reference(tmp_path):
    repo, _ = _official("firecastnet_lit")
    checkpoint = oracle_asset("firecastnet_lam_aust_cls_ts24_h1") / "firecastnet-cls-AUST-ts24-h1.ckpt"
    mesh = _checkpoint_hparams(checkpoint)["icospheres_graph_path"]
    assert mesh == "icospheres-lam/icospheres/icosphere_s3_AUST_6u.json.gz"
    with pytest.raises(ValueError, match="icospheres_graph_path"):
        load_official_checkpoint(checkpoint)
    _compare_checkpoint(tmp_path, checkpoint, icospheres_graph_path=repo / mesh)
