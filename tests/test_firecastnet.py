"""FireCastNet without the reference code: sizes, shapes, validation and checkpoint loading.

The comparison with the official implementation is in tests/oracle/test_firecastnet_oracle.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pyhazards.models import build_model
from pyhazards.models.firecastnet import FireCastNet, lightning_state_dict_to_firecastnet, load_official_checkpoint
from pyhazards.models.firecastnet_graph import build_graph, icosphere, icospheres, latlon_grid

# A 10-degree graph grid (input 2.5 degrees, 72 x 144) on icosphere levels 0-2.
TINY_GLOBAL = dict(sp_res=2.5, max_lat=88.75, min_lat=-88.75, max_lon=178.75, min_lon=-178.75)
TINY = dict(in_channels=3, timeseries_len=2, mesh_levels=(0, 1, 2), hidden_dim=16, embed_cube_dim=16, processor_layers=3, **TINY_GLOBAL)


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_released_global_configuration():
    model = build_model(name="firecastnet", task="segmentation")
    assert _n_params(model) == 9_087_568
    assert _n_params(model._downsample.layer_norm) == 8_294_400  # LayerNorm over the whole 64x1x180x360 cube
    assert (model.lat_dim, model.lon_dim, model.graph_lat_dim, model.graph_lon_dim) == (720, 1440, 180, 360)
    net = model._net
    assert (net.num_grid_nodes, net.num_mesh_nodes) == (64_800, 40_962)
    assert (net.mesh_src.numel(), net.g2m_src.numel(), net.m2g_src.numel()) == (327_660, 100_160, 64_800)
    assert len(model.state_dict()) == 212
    assert all(not name.startswith(("static_data", "_net.mesh_", "_net.g2m_", "_net.m2g_")) for name in model.state_dict())
    assert len(net.processor.processor_layers) == 10


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_icosphere_sizes_and_nesting(level):
    vertices, faces = icosphere(level)
    assert vertices.shape == (10 * 4**level + 2, 3)
    assert faces.shape == (20 * 4**level, 3)
    np.testing.assert_allclose(np.linalg.norm(vertices, axis=1), 1.0, rtol=0, atol=1e-15)
    edges = {tuple(sorted(edge)) for face in faces.tolist() for edge in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))}
    assert len(vertices) - len(edges) + len(faces) == 2  # Euler characteristic of a sphere
    if level:
        coarse, _ = icosphere(level - 1)
        assert np.array_equal(vertices[: len(coarse)], coarse)
    meshes = icospheres(range(level + 1))
    assert np.array_equal(meshes[f"order_{level}_vertices"], vertices)


def test_graph_on_tiny_global_grid():
    latitudes, longitudes = latlon_grid(10.0, 85.0, -85.0, 175.0, -175.0)
    graph = build_graph(latitudes, longitudes, icospheres((0, 1, 2)))
    assert graph.num_grid_nodes == 18 * 36 and graph.num_mesh_nodes == 162
    # the multimesh of levels 0-2 has 30 + 120 + 480 undirected edges
    assert graph.mesh_src.numel() == 2 * (30 + 120 + 480)
    assert torch.all(graph.mesh_src != graph.mesh_dst)
    assert graph.g2m_edata.shape == (graph.g2m_src.numel(), 4) and graph.m2g_edata.shape == (648, 4)
    assert torch.equal(graph.m2g_dst, torch.arange(648))
    assert float(graph.mesh_edata[:, 3].max()) == pytest.approx(1.0)  # normalised by the longest edge


def test_forward_shape_and_batching():
    torch.manual_seed(0)
    model = build_model(name="firecastnet", task="segmentation", **TINY).eval()
    x = torch.randn(2, 2, 3, 72, 144)
    with torch.no_grad():
        out = model(x)
        assert out.shape == (2, 1, 72, 144)
        torch.testing.assert_close(out[1:], model(x[1:]), rtol=1e-5, atol=1e-6)


def test_two_output_channels_and_options():
    model = FireCastNet(output_dim_grid_nodes=32, lat_lon_static_data=False, aggregation="mean", hidden_layers=2, embed_cube_layer_norm=False, **TINY)
    assert model._downsample.layer_norm is None
    assert model(torch.randn(1, 2, 3, 72, 144)).shape == (1, 2, 72, 144)


def test_regional_window():
    # 16 x 16 cells of 0.25 degree; the reference code needs global grids, the port does not.
    model = FireCastNet(
        in_channels=6, timeseries_len=4, mesh_levels=(0, 1, 2, 3, 4), sp_res=0.25,
        max_lat=41.875, min_lat=38.125, max_lon=-120.125, min_lon=-123.875, hidden_dim=16, embed_cube_dim=16,
    )
    assert (model.graph_lat_dim, model.graph_lon_dim) == (4, 4)
    assert model(torch.randn(2, 4, 6, 16, 16)).shape == (2, 1, 16, 16)


def test_input_and_config_validation():
    model = FireCastNet(**TINY)
    for bad in (torch.randn(2, 3, 72, 144), torch.randn(1, 2, 4, 72, 144), torch.randn(1, 3, 3, 72, 144), torch.randn(1, 2, 3, 72, 140)):
        with pytest.raises(ValueError, match="shape"):
            model(bad)
    with pytest.raises(ValueError, match="embed_cube_time"):
        FireCastNet(embed_cube_time=1, **TINY)
    with pytest.raises(ValueError, match="at least 3 processor layers"):
        FireCastNet(**{**TINY, "processor_layers": 2})
    with pytest.raises(ValueError, match="norm_type"):
        FireCastNet(norm_type="BatchNorm", **TINY)
    with pytest.raises(ValueError, match="equal"):
        FireCastNet(embed_cube_width=4, embed_cube_height=2, **TINY)
    with pytest.raises(ValueError, match="graph grid"):
        FireCastNet(embed_cube_max_lat=80.0, **TINY)
    with pytest.raises(ValueError, match="task"):
        build_model(name="firecastnet", task="classification", **TINY)
    with pytest.raises(TypeError):
        build_model(name="firecastnet", task="segmentation", dropout=0.1, **TINY)


def test_official_checkpoint_layout_round_trip(tmp_path):
    torch.manual_seed(0)
    model = FireCastNet(**TINY).eval()
    state = {"_net." + key: value for key, value in model.state_dict().items()}
    state["_lsm_mask"] = torch.zeros(1, 72, 144, dtype=torch.bool)  # LightningModule buffer, not a weight
    hparams = dict(
        icospheres_graph_path="icospheres/icospheres_0_1_2.json.gz",
        input_dim_grid_nodes=3,
        timeseries_len=2,
        embed_cube=True,
        embed_cube_time=2,
        embed_cube_dim=16,
        embed_cube_sp_res=10.0,
        embed_cube_max_lat=85.0,
        embed_cube_min_lat=-85.0,
        embed_cube_max_lon=175.0,
        embed_cube_min_lon=-175.0,
        output_dim_grid_nodes=16,
        hidden_dim=16,
        processor_layers=3,
        **TINY_GLOBAL,
    )
    path = tmp_path / "firecastnet.ckpt"
    torch.save({"state_dict": state, "hyper_parameters": hparams}, path)
    assert list(lightning_state_dict_to_firecastnet(state)) == list(model.state_dict())

    loaded = build_model(name="firecastnet", task="segmentation", checkpoint=str(path)).eval()
    x = torch.randn(1, 2, 3, 72, 144)
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), model(x), rtol=0, atol=0)

    torch.save({"state_dict": state, "hyper_parameters": {**hparams, "icospheres_graph_path": "lam.json.gz"}}, path)
    with pytest.raises(ValueError, match="icospheres_graph_path"):
        load_official_checkpoint(path)
