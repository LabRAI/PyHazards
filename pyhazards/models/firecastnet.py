"""FireCastNet: seasonal wildfire forecasting with GraphCast on an icosahedral multi-mesh.

Port of Michail et al., "FireCastNet: Earth-as-a-Graph for Seasonal Fire Prediction", Scientific
Reports 16:1006 (2025), doi:10.1038/s41598-025-30645-7, arXiv:2502.01550. Official code:
https://github.com/SeasFire/firecastnet (PyTorch + Lightning + DGL), checked at commit
``dc0d131be7f0638e3a84c16817e7bc9be454e53e``.

The model has three stages (paper Sec. 4): a cube embedding (``Conv3d`` with kernel and stride
``T x 4 x 4`` plus LayerNorm) that turns ``T`` 8-day steps of a 0.25-degree grid into 64 features
on a 1-degree grid; a GraphCast encoder-processor-decoder between that grid and an icosahedral
multi-mesh (12 message-passing layers, hidden size 64); and a sub-pixel up-sampling
(``PixelShuffle(4)``) of 16 features per 1-degree cell back to the 0.25-degree grid.

Provenance and licenses
-----------------------
- ``MeshGraphMLP``, ``MeshGraphEdgeMLPConcat``, ``GraphCastEncoderEmbedder``,
  ``GraphCastDecoderEmbedder``, ``MeshGraphEncoder``, ``MeshGraphDecoder``, ``MeshProcessorBlock``
  and ``GraphCastMeshProcessor`` are ported from SeasFire/firecastnet
  ``seasfire/backbones/graphcast/gnn_layers/*.py`` and ``graph_cast_mesh_processor.py``, whose
  headers read ``SPDX-License-Identifier: Apache-2.0``, Copyright (c) 2023 - 2024 NVIDIA
  CORPORATION & AFFILIATES (derived from NVIDIA Modulus). Licensed under the Apache License 2.0
  (http://www.apache.org/licenses/LICENSE-2.0). Changes: DGL message passing is replaced by
  ``index_select`` / ``index_add_`` over edge-index buffers, and only the code paths without a
  time dimension are kept.
- ``GraphCastNet`` follows NVIDIA Modulus v0.5.0 ``modulus/models/graphcast/graph_cast_net.py``
  (Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES, Apache-2.0), with the module order used by
  FireCastNet. Graph construction lives in :mod:`pyhazards.models.firecastnet_graph`.
- The cube embedding, the up-sampling, the static latitude/longitude channels and the overall
  wiring (``FireCastNet``) are written from the paper. The rest of the official repository has no
  licence; it is used only as a test oracle (``tests/oracle/test_firecastnet_oracle.py``) and is not
  copied here.

Parameter names follow the official ``GraphCastCubeNet``, so official checkpoints load with
``strict=True`` after removing the LightningModule prefix (see :func:`load_official_checkpoint`).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from .firecastnet_graph import FireCastNetGraph, build_graph, deg2rad, icospheres, latlon_grid, load_icospheres

OFFICIAL_INPUT_VARIABLES = (
    "mslp",
    "tp",
    "vpd",
    "sst",
    "t2m_mean",
    "ssrd",
    "swvl1",
    "lst_day",
    "ndvi",
    "pop_dens",
    "lsm",
)

_Edges = Tuple[Tensor, Tensor, int]  # (src index, dst index, number of destination nodes)


def _concat_efeat(efeat: Tensor, nfeat: Union[Tensor, Tuple[Tensor, Tensor]], edges: _Edges) -> Tensor:
    """Concatenate edge, source-node and destination-node features per edge (in that order)."""
    src_feat, dst_feat = (nfeat, nfeat) if isinstance(nfeat, Tensor) else nfeat
    src, dst, _ = edges
    return torch.cat((efeat, src_feat.index_select(0, src), dst_feat.index_select(0, dst)), dim=1)


def _aggregate_and_concat(efeat: Tensor, dst_nfeat: Tensor, edges: _Edges, aggregation: str) -> Tensor:
    """Sum (or mean) edge features into their destination nodes, then append the node features."""
    _, dst, num_dst = edges
    agg = efeat.new_zeros((num_dst, efeat.size(1))).index_add_(0, dst, efeat)
    if aggregation == "mean":
        degree = torch.bincount(dst, minlength=num_dst).clamp(min=1).to(agg.dtype)
        agg = agg / degree.unsqueeze(1)
    elif aggregation != "sum":
        raise RuntimeError("Not a valid aggregation!")
    return torch.cat((agg, dst_nfeat), dim=-1)


class MeshGraphMLP(nn.Module):
    """Linear layers with SiLU in between and an optional LayerNorm after the last one."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: Optional[int] = 1,
        activation_fn: Optional[nn.Module] = None,
        norm_type: Optional[str] = "LayerNorm",
    ):
        super().__init__()
        activation_fn = nn.SiLU() if activation_fn is None else activation_fn
        if hidden_layers is not None:
            layers = [nn.Linear(input_dim, hidden_dim), activation_fn]
            self.hidden_layers = hidden_layers
            for _ in range(hidden_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), activation_fn]
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.norm_type = norm_type
            if norm_type is not None:
                if norm_type != "LayerNorm":
                    raise ValueError(f"norm_type must be 'LayerNorm' or None, got {norm_type!r}")
                layers.append(nn.LayerNorm(output_dim))
            self.model = nn.Sequential(*layers)
        else:
            self.model = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class MeshGraphEdgeMLPConcat(MeshGraphMLP):
    """Edge MLP on the concatenation of edge, source-node and destination-node features."""

    def __init__(
        self,
        efeat_dim: int = 512,
        src_dim: int = 512,
        dst_dim: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 2,
        activation_fn: Optional[nn.Module] = None,
        norm_type: Optional[str] = "LayerNorm",
    ):
        super().__init__(efeat_dim + src_dim + dst_dim, output_dim, hidden_dim, hidden_layers, activation_fn, norm_type)

    def forward(self, efeat: Tensor, nfeat: Union[Tensor, Tuple[Tensor, Tensor]], edges: _Edges) -> Tensor:
        return self.model(_concat_efeat(efeat, nfeat, edges))


class GraphCastEncoderEmbedder(nn.Module):
    """Embeds grid-node, mesh-node, mesh-edge and grid2mesh-edge input features."""

    def __init__(
        self,
        input_dim_grid_nodes: int = 10,
        input_dim_mesh_nodes: int = 3,
        input_dim_edges: int = 4,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: Optional[nn.Module] = None,
        norm_type: Optional[str] = "LayerNorm",
    ):
        super().__init__()
        mlp = dict(output_dim=output_dim, hidden_dim=hidden_dim, hidden_layers=hidden_layers, activation_fn=activation_fn, norm_type=norm_type)
        self.grid_node_mlp = MeshGraphMLP(input_dim=input_dim_grid_nodes, **mlp)
        self.mesh_node_mlp = MeshGraphMLP(input_dim=input_dim_mesh_nodes, **mlp)
        self.mesh_edge_mlp = MeshGraphMLP(input_dim=input_dim_edges, **mlp)
        self.grid2mesh_edge_mlp = MeshGraphMLP(input_dim=input_dim_edges, **mlp)

    def forward(self, grid_nfeat: Tensor, mesh_nfeat: Tensor, g2m_efeat: Tensor, mesh_efeat: Tensor):
        return (
            self.grid_node_mlp(grid_nfeat),
            self.mesh_node_mlp(mesh_nfeat),
            self.grid2mesh_edge_mlp(g2m_efeat),
            self.mesh_edge_mlp(mesh_efeat),
        )


class GraphCastDecoderEmbedder(nn.Module):
    """Embeds mesh2grid edge features."""

    def __init__(
        self,
        input_dim_edges: int = 4,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: Optional[nn.Module] = None,
        norm_type: Optional[str] = "LayerNorm",
    ):
        super().__init__()
        self.mesh2grid_edge_mlp = MeshGraphMLP(
            input_dim=input_dim_edges,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

    def forward(self, m2g_efeat: Tensor) -> Tensor:
        return self.mesh2grid_edge_mlp(m2g_efeat)


class MeshGraphEncoder(nn.Module):
    """Grid -> mesh message passing: updates the mesh nodes, and the grid nodes by a residual MLP."""

    def __init__(
        self,
        aggregation: str = "sum",
        input_dim_src_nodes: int = 512,
        input_dim_dst_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim_src_nodes: int = 512,
        output_dim_dst_nodes: int = 512,
        output_dim_edges: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: Optional[nn.Module] = None,
        norm_type: Optional[str] = "LayerNorm",
    ):
        super().__init__()
        self.aggregation = aggregation
        mlp = dict(hidden_dim=hidden_dim, hidden_layers=hidden_layers, activation_fn=activation_fn, norm_type=norm_type)
        self.edge_mlp = MeshGraphEdgeMLPConcat(
            efeat_dim=input_dim_edges, src_dim=input_dim_src_nodes, dst_dim=input_dim_dst_nodes, output_dim=output_dim_edges, **mlp
        )
        self.src_node_mlp = MeshGraphMLP(input_dim=input_dim_src_nodes, output_dim=output_dim_src_nodes, **mlp)
        self.dst_node_mlp = MeshGraphMLP(input_dim=input_dim_dst_nodes + output_dim_edges, output_dim=output_dim_dst_nodes, **mlp)

    def forward(self, g2m_efeat: Tensor, grid_nfeat: Tensor, mesh_nfeat: Tensor, edges: _Edges) -> Tuple[Tensor, Tensor]:
        efeat = self.edge_mlp(g2m_efeat, (grid_nfeat, mesh_nfeat), edges)
        cat_feat = _aggregate_and_concat(efeat, mesh_nfeat, edges, self.aggregation)
        mesh_nfeat = mesh_nfeat + self.dst_node_mlp(cat_feat)
        grid_nfeat = grid_nfeat + self.src_node_mlp(grid_nfeat)
        return grid_nfeat, mesh_nfeat


class MeshGraphDecoder(nn.Module):
    """Mesh -> grid message passing with a residual update of the grid nodes."""

    def __init__(
        self,
        aggregation: str = "sum",
        input_dim_src_nodes: int = 512,
        input_dim_dst_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim_dst_nodes: int = 512,
        output_dim_edges: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: Optional[nn.Module] = None,
        norm_type: Optional[str] = "LayerNorm",
    ):
        super().__init__()
        self.aggregation = aggregation
        mlp = dict(hidden_dim=hidden_dim, hidden_layers=hidden_layers, activation_fn=activation_fn, norm_type=norm_type)
        self.edge_mlp = MeshGraphEdgeMLPConcat(
            efeat_dim=input_dim_edges, src_dim=input_dim_src_nodes, dst_dim=input_dim_dst_nodes, output_dim=output_dim_edges, **mlp
        )
        self.node_mlp = MeshGraphMLP(input_dim=input_dim_dst_nodes + output_dim_edges, output_dim=output_dim_dst_nodes, **mlp)

    def forward(self, m2g_efeat: Tensor, grid_nfeat: Tensor, mesh_nfeat: Tensor, edges: _Edges) -> Tensor:
        efeat = self.edge_mlp(m2g_efeat, (mesh_nfeat, grid_nfeat), edges)
        cat_feat = _aggregate_and_concat(efeat, grid_nfeat, edges, self.aggregation)
        return self.node_mlp(cat_feat) + grid_nfeat


class MeshProcessorBlock(nn.Module):
    """One multi-mesh message-passing layer: residual edge update, then residual node update."""

    def __init__(
        self,
        aggregation: str = "sum",
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim_nodes: int = 512,
        output_dim_edges: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: Optional[nn.Module] = None,
        norm_type: Optional[str] = "LayerNorm",
    ):
        super().__init__()
        self.aggregation = aggregation
        self.input_dim_nodes = input_dim_nodes
        self.output_dim_nodes = output_dim_nodes
        self.input_dim_edges = input_dim_edges
        self.output_dim_edges = output_dim_edges
        mlp = dict(hidden_dim=hidden_dim, hidden_layers=hidden_layers, activation_fn=activation_fn, norm_type=norm_type)
        self.edge_mlp = MeshGraphMLP(input_dim=2 * input_dim_nodes + input_dim_edges, output_dim=output_dim_edges, **mlp)
        self.node_mlp = MeshGraphMLP(input_dim=input_dim_nodes + output_dim_edges, output_dim=output_dim_nodes, **mlp)

    def forward(self, efeat: Tensor, nfeat: Tensor, edges: _Edges) -> Tuple[Tensor, Tensor]:
        efeat_new = self.edge_mlp(_concat_efeat(efeat, nfeat, edges))
        if self.input_dim_edges == self.output_dim_edges:
            efeat_new = efeat_new + efeat
        nfeat_new = self.node_mlp(_aggregate_and_concat(efeat_new, nfeat, edges, self.aggregation))
        if self.input_dim_nodes == self.output_dim_nodes:
            nfeat_new = nfeat_new + nfeat
        return efeat_new, nfeat_new


class GraphCastMeshProcessor(nn.Module):
    """A stack of :class:`MeshProcessorBlock` layers on the multi-mesh."""

    def __init__(
        self,
        aggregation: str = "sum",
        processor_layers: int = 16,
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: Optional[nn.Module] = None,
        norm_type: Optional[str] = "LayerNorm",
    ):
        super().__init__()
        self.has_time_dim = False
        self.hidden_dim = hidden_dim
        self.processor_layers = nn.ModuleList(
            MeshProcessorBlock(
                aggregation, input_dim_nodes, input_dim_edges, input_dim_nodes, input_dim_edges, hidden_dim, hidden_layers, activation_fn, norm_type
            )
            for _ in range(processor_layers)
        )
        self.num_processor_layers = len(self.processor_layers)

    def forward(self, efeat: Tensor, nfeat: Tensor, edges: _Edges) -> Tuple[Tensor, Tensor]:
        for layer in self.processor_layers:
            efeat, nfeat = layer(efeat, nfeat, edges)
        return efeat, nfeat


class GraphCastNet(nn.Module):
    """GraphCast encoder-processor-decoder on fixed grid/mesh graphs.

    Takes grid-node features ``(num_grid_nodes, input_dim_grid_nodes)`` and returns
    ``(num_grid_nodes, output_dim_grid_nodes)``. The graphs are kept as non-persistent buffers, so
    they follow ``.to(device)`` but are not part of the ``state_dict``.
    """

    def __init__(
        self,
        graph: FireCastNetGraph,
        input_dim_grid_nodes: int = 10,
        input_dim_mesh_nodes: int = 3,
        input_dim_edges: int = 4,
        output_dim_grid_nodes: int = 1,
        processor_layers: int = 4,
        hidden_layers: int = 1,
        hidden_dim: int = 512,
        aggregation: str = "sum",
        norm_type: Optional[str] = "LayerNorm",
    ):
        super().__init__()
        if aggregation not in {"sum", "mean"}:
            raise ValueError(f"aggregation must be 'sum' or 'mean', got {aggregation!r}")
        if processor_layers <= 2:
            raise ValueError("Expected at least 3 processor layers")
        self.input_dim_grid_nodes = input_dim_grid_nodes
        self.num_grid_nodes = graph.num_grid_nodes
        self.num_mesh_nodes = graph.num_mesh_nodes
        for name in ("mesh_src", "mesh_dst", "g2m_src", "g2m_dst", "m2g_src", "m2g_dst"):
            self.register_buffer(name, getattr(graph, name).long(), persistent=False)
        for name in ("mesh_edata", "mesh_ndata", "g2m_edata", "m2g_edata"):
            self.register_buffer(name, getattr(graph, name).float(), persistent=False)

        activation_fn = nn.SiLU()
        common = dict(hidden_dim=hidden_dim, hidden_layers=hidden_layers, activation_fn=activation_fn, norm_type=norm_type)
        self.encoder_embedder = GraphCastEncoderEmbedder(
            input_dim_grid_nodes=input_dim_grid_nodes,
            input_dim_mesh_nodes=input_dim_mesh_nodes,
            input_dim_edges=input_dim_edges,
            output_dim=hidden_dim,
            **common,
        )
        self.encoder = MeshGraphEncoder(
            aggregation=aggregation,
            input_dim_src_nodes=hidden_dim,
            input_dim_dst_nodes=hidden_dim,
            input_dim_edges=hidden_dim,
            output_dim_src_nodes=hidden_dim,
            output_dim_dst_nodes=hidden_dim,
            output_dim_edges=hidden_dim,
            **common,
        )
        processor = dict(aggregation=aggregation, input_dim_nodes=hidden_dim, input_dim_edges=hidden_dim, **common)
        self.processor_encoder = GraphCastMeshProcessor(processor_layers=1, **processor)
        self.processor = GraphCastMeshProcessor(processor_layers=processor_layers - 2, **processor)
        self.processor_decoder = GraphCastMeshProcessor(processor_layers=1, **processor)
        self.decoder_embedder = GraphCastDecoderEmbedder(input_dim_edges=input_dim_edges, output_dim=hidden_dim, **common)
        self.decoder = MeshGraphDecoder(
            aggregation=aggregation,
            input_dim_src_nodes=hidden_dim,
            input_dim_dst_nodes=hidden_dim,
            input_dim_edges=hidden_dim,
            output_dim_dst_nodes=hidden_dim,
            output_dim_edges=hidden_dim,
            **common,
        )
        self.finale = MeshGraphMLP(
            input_dim=hidden_dim,
            output_dim=output_dim_grid_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=None,
        )

    def forward(self, grid_nfeat: Tensor) -> Tensor:
        if grid_nfeat.ndim != 2 or grid_nfeat.shape != (self.num_grid_nodes, self.input_dim_grid_nodes):
            raise ValueError(
                "GraphCastNet expects grid node features of shape "
                f"({self.num_grid_nodes}, {self.input_dim_grid_nodes}), got {tuple(grid_nfeat.shape)}."
            )
        g2m = (self.g2m_src, self.g2m_dst, self.num_mesh_nodes)
        mesh = (self.mesh_src, self.mesh_dst, self.num_mesh_nodes)
        m2g = (self.m2g_src, self.m2g_dst, self.num_grid_nodes)

        grid_emb, mesh_emb, g2m_emb, mesh_edge_emb = self.encoder_embedder(
            grid_nfeat, self.mesh_ndata.to(grid_nfeat), self.g2m_edata.to(grid_nfeat), self.mesh_edata.to(grid_nfeat)
        )
        grid_encoded, mesh_encoded = self.encoder(g2m_emb, grid_emb, mesh_emb, g2m)
        mesh_efeat, mesh_nfeat = self.processor_encoder(mesh_edge_emb, mesh_encoded, mesh)
        mesh_efeat, mesh_nfeat = self.processor(mesh_efeat, mesh_nfeat, mesh)
        _, mesh_nfeat = self.processor_decoder(mesh_efeat, mesh_nfeat, mesh)
        m2g_emb = self.decoder_embedder(self.m2g_edata.to(grid_nfeat))
        grid_decoded = self.decoder(m2g_emb, grid_encoded, mesh_nfeat, m2g)
        return self.finale(grid_decoded)


class CubeConv3d(nn.Module):
    """Space-time cube embedding: non-overlapping ``Conv3d`` patches, then LayerNorm.

    As in the official configuration the LayerNorm normalises over the whole embedded cube
    ``(C', T', H', W')`` of one sample, with an elementwise affine transform of that shape.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        origin_shape: Tuple[int, int, int],
        kernel_size: Tuple[int, int, int],
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.conv_3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size)
        embedded = tuple(size // patch for size, patch in zip(origin_shape, kernel_size))
        self.layer_norm = nn.LayerNorm((out_channels, *embedded)) if use_layer_norm else None

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv_3d(x)
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        return out


def _official_levels(path: str) -> Optional[Tuple[int, ...]]:
    match = re.fullmatch(r"icospheres_(\d+(?:_\d+)*)\.json(?:\.gz)?", Path(path).name)
    return tuple(int(level) for level in match.group(1).split("_")) if match else None


class FireCastNet(nn.Module):
    """FireCastNet for one latitude/longitude window (global by default).

    Input ``(batch, timeseries_len, in_channels, n_lat, n_lon)``: ``timeseries_len`` 8-day steps of
    ``in_channels`` standardised variables on the grid given by ``sp_res`` and the cell-centre
    bounds (latitudes from north to south, longitudes from west to east). The official models use
    the 10 SeasFire inputs plus the land-sea mask (:data:`OFFICIAL_INPUT_VARIABLES`) on the global
    0.25-degree grid (720 x 1440). ``cos(lat)``, ``sin(lon)`` and ``cos(lon)`` are appended
    internally when ``lat_lon_static_data`` is true.

    Output: logits ``(batch, output_dim_grid_nodes // r**2, n_lat, n_lon)`` with ``r`` the patch
    size, i.e. one burned-area logit (or regression value) per cell with the default 16 outputs.

    The mesh is either the icospheres of ``mesh_levels`` (default 0-6, as in the paper) or a mesh
    file in the official layout given by ``icospheres_graph_path`` (for example a LAM mesh).
    """

    def __init__(
        self,
        in_channels: int = 11,
        timeseries_len: int = 24,
        mesh_levels: Sequence[int] = (0, 1, 2, 3, 4, 5, 6),
        icospheres_graph_path: Optional[Union[str, Path]] = None,
        sp_res: float = 0.25,
        max_lat: float = 89.875,
        min_lat: float = -89.875,
        max_lon: float = 179.875,
        min_lon: float = -179.875,
        lat_lon_static_data: bool = True,
        embed_cube_width: int = 4,
        embed_cube_height: int = 4,
        embed_cube_time: Optional[int] = None,
        embed_cube_dim: int = 64,
        embed_cube_layer_norm: bool = True,
        embed_cube_sp_res: Optional[float] = None,
        embed_cube_max_lat: Optional[float] = None,
        embed_cube_min_lat: Optional[float] = None,
        embed_cube_max_lon: Optional[float] = None,
        embed_cube_min_lon: Optional[float] = None,
        output_dim_grid_nodes: int = 16,
        input_dim_mesh_nodes: int = 3,
        input_dim_edges: int = 4,
        processor_layers: int = 12,
        hidden_layers: int = 1,
        hidden_dim: int = 64,
        aggregation: str = "sum",
        norm_type: Optional[str] = "LayerNorm",
    ):
        super().__init__()
        for name, value in (("in_channels", in_channels), ("timeseries_len", timeseries_len), ("embed_cube_dim", embed_cube_dim)):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        embed_cube_time = timeseries_len if embed_cube_time is None else int(embed_cube_time)
        if embed_cube_time != timeseries_len:
            raise ValueError(
                "embed_cube_time must equal timeseries_len: the official recurrent processor used when the cube "
                f"embedding keeps a time dimension is not ported (got {embed_cube_time} and {timeseries_len})."
            )
        if embed_cube_width != embed_cube_height:
            raise ValueError("embed_cube_width and embed_cube_height must be equal (sub-pixel up-sampling).")
        patch = int(embed_cube_width)
        if output_dim_grid_nodes % (patch * patch) != 0:
            raise ValueError(f"output_dim_grid_nodes must be a multiple of {patch * patch}, got {output_dim_grid_nodes}")

        self.in_channels = int(in_channels)
        self.timeseries_len = int(timeseries_len)
        self.out_channels = output_dim_grid_nodes // (patch * patch)

        # Input grid (0.25 degree by default) and the grid of the embedded cube, on which the graph lives.
        self.latitudes, self.longitudes = latlon_grid(sp_res, max_lat, min_lat, max_lon, min_lon)
        offset = sp_res * (patch - 1) / 2
        g_sp_res = sp_res * patch if embed_cube_sp_res is None else embed_cube_sp_res
        g_max_lat = max_lat - offset if embed_cube_max_lat is None else embed_cube_max_lat
        g_min_lat = min_lat + offset if embed_cube_min_lat is None else embed_cube_min_lat
        g_max_lon = max_lon - offset if embed_cube_max_lon is None else embed_cube_max_lon
        g_min_lon = min_lon + offset if embed_cube_min_lon is None else embed_cube_min_lon
        self.graph_latitudes, self.graph_longitudes = latlon_grid(g_sp_res, g_max_lat, g_min_lat, g_max_lon, g_min_lon)
        self.lat_dim, self.lon_dim = self.latitudes.numel(), self.longitudes.numel()
        self.graph_lat_dim, self.graph_lon_dim = self.graph_latitudes.numel(), self.graph_longitudes.numel()
        if (self.lat_dim // patch, self.lon_dim // patch) != (self.graph_lat_dim, self.graph_lon_dim):
            raise ValueError(
                f"the {self.lat_dim}x{self.lon_dim} input grid embeds to {self.lat_dim // patch}x{self.lon_dim // patch} "
                f"cells, but the graph grid has shape {self.graph_lat_dim}x{self.graph_lon_dim}"
            )

        if icospheres_graph_path is not None:
            meshes = load_icospheres(icospheres_graph_path)
        else:
            meshes = icospheres(mesh_levels)
        graph = build_graph(self.graph_latitudes, self.graph_longitudes, meshes)

        self.lat_lon_static_data = bool(lat_lon_static_data)
        static = None
        if self.lat_lon_static_data:
            cos_lat = torch.cos(deg2rad(self.latitudes)).view(-1, 1).expand(self.lat_dim, self.lon_dim)
            sin_lon = torch.sin(deg2rad(self.longitudes)).view(1, -1).expand(self.lat_dim, self.lon_dim)
            cos_lon = torch.cos(deg2rad(self.longitudes)).view(1, -1).expand(self.lat_dim, self.lon_dim)
            static = torch.stack((cos_lat, sin_lon, cos_lon))
        self.register_buffer("static_data", static, persistent=False)
        grid_channels = self.in_channels + (3 if self.lat_lon_static_data else 0)

        # Module creation order follows the reference (GraphCastNet, then the cube embedding) so that
        # the same torch.manual_seed gives the same initial weights.
        self._net = GraphCastNet(
            graph,
            input_dim_grid_nodes=embed_cube_dim,
            input_dim_mesh_nodes=input_dim_mesh_nodes,
            input_dim_edges=input_dim_edges,
            output_dim_grid_nodes=output_dim_grid_nodes,
            processor_layers=processor_layers,
            hidden_layers=hidden_layers,
            hidden_dim=hidden_dim,
            aggregation=aggregation,
            norm_type=norm_type,
        )
        self._downsample = CubeConv3d(
            grid_channels,
            embed_cube_dim,
            (self.timeseries_len, self.lat_dim, self.lon_dim),
            (embed_cube_time, patch, patch),
            use_layer_norm=embed_cube_layer_norm,
        )
        self._upsample = nn.PixelShuffle(patch)

    def forward(self, x: Tensor) -> Tensor:
        expected = (self.timeseries_len, self.in_channels, self.lat_dim, self.lon_dim)
        if x.ndim != 5 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                "FireCastNet expects input shape (batch, timeseries_len, in_channels, n_lat, n_lon) = "
                f"(B, {', '.join(map(str, expected))}), got {tuple(x.shape)}."
            )
        batch, steps = x.size(0), x.size(1)
        x = x.transpose(1, 2)  # (B, C, T, H, W), the layout of the reference model
        if self.static_data is not None:
            static = self.static_data.to(x)[None, :, None].expand(batch, -1, steps, -1, -1)
            x = torch.cat((x, static), dim=1)
        else:
            x = x.contiguous()
        x = self._downsample(x)  # (B, C', 1, H', W')
        logits = []
        for sample in x:
            nodes = sample[:, 0].reshape(sample.size(0), -1).permute(1, 0)  # (H' * W', C')
            out = self._net(nodes)
            logits.append(out.view(self.graph_lat_dim, self.graph_lon_dim, -1).permute(2, 0, 1))
        return self._upsample(torch.stack(logits))


# Defaults of the official FireCastNetLit, used for keys missing from a checkpoint's hyper_parameters.
_LIGHTNING_DEFAULTS: Dict[str, Any] = dict(
    icospheres_graph_path="icospheres/icospheres_0_1_2_3.json.gz",
    sp_res=0.25,
    max_lat=89.875,
    min_lat=-89.875,
    max_lon=179.875,
    min_lon=-179.875,
    lat_lon_static_data=True,
    embed_cube=False,
    embed_cube_width=4,
    embed_cube_height=4,
    embed_cube_time=1,
    embed_cube_dim=128,
    embed_cube_layer_norm=True,
    embed_cube_sp_res=1.0,
    embed_cube_max_lat=89.5,
    embed_cube_min_lat=-89.5,
    embed_cube_max_lon=179.5,
    embed_cube_min_lon=-179.5,
    timeseries_len=1,
    input_dim_grid_nodes=11,
    output_dim_grid_nodes=1,
    input_dim_mesh_nodes=3,
    input_dim_edges=4,
    processor_layers=8,
    hidden_layers=1,
    hidden_dim=64,
    aggregation="sum",
    norm_type="LayerNorm",
    do_concat_trick=False,
)
_MODEL_HPARAMS = (
    "sp_res", "max_lat", "min_lat", "max_lon", "min_lon", "lat_lon_static_data", "embed_cube_width",
    "embed_cube_height", "embed_cube_time", "embed_cube_dim", "embed_cube_layer_norm", "embed_cube_sp_res",
    "embed_cube_max_lat", "embed_cube_min_lat", "embed_cube_max_lon", "embed_cube_min_lon",
    "output_dim_grid_nodes", "input_dim_mesh_nodes", "input_dim_edges", "processor_layers",
    "hidden_layers", "hidden_dim", "aggregation", "norm_type",
)


def lightning_state_dict_to_firecastnet(state_dict: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    """Map a ``FireCastNetLit`` checkpoint ``state_dict`` to :class:`FireCastNet` keys.

    The network weights sit under the LightningModule attribute ``_net``, so ``_net.<key>``
    becomes ``<key>``. Data-derived buffers of the LightningModule (``_lsm_mask``, the loss's
    ``_criterion.pixel_weights``) are not network weights and are dropped.
    """
    return {key[len("_net."):]: value for key, value in state_dict.items() if key.startswith("_net.")}


def load_official_checkpoint(
    path: Union[str, Path],
    icospheres_graph_path: Optional[Union[str, Path]] = None,
    map_location: Union[str, torch.device] = "cpu",
) -> FireCastNet:
    """Build :class:`FireCastNet` from an official checkpoint and load its weights (``strict=True``).

    The configuration comes from the checkpoint's ``hyper_parameters``. Meshes named
    ``icospheres_<levels>.json.gz`` are regenerated; any other mesh (the LAM meshes) must be given
    as a local file in ``icospheres_graph_path``.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    hparams: Dict[str, Any] = {**_LIGHTNING_DEFAULTS, **checkpoint.get("hyper_parameters", {})}
    if not hparams["embed_cube"]:
        raise ValueError("only checkpoints with embed_cube=True (FireCastNet) are supported")
    if hparams.get("embed_cube_vit_enable") or hparams.get("embed_cube_ltae_enable"):
        raise ValueError("ViT / L-TAE cube embedders are not ported")
    if hparams["do_concat_trick"]:
        raise ValueError("do_concat_trick=True is not ported")
    kwargs: Dict[str, Any] = {key: hparams[key] for key in _MODEL_HPARAMS}
    if icospheres_graph_path is not None:
        kwargs["icospheres_graph_path"] = icospheres_graph_path
    else:
        levels = _official_levels(str(hparams["icospheres_graph_path"]))
        if levels is None:
            raise ValueError(
                f"mesh {hparams['icospheres_graph_path']!r} is not a regular icosphere set; pass icospheres_graph_path"
            )
        kwargs["mesh_levels"] = levels
    model = FireCastNet(in_channels=hparams["input_dim_grid_nodes"], timeseries_len=hparams["timeseries_len"], **kwargs)
    model.load_state_dict(lightning_state_dict_to_firecastnet(checkpoint["state_dict"]), strict=True)
    return model


def firecastnet_builder(
    task: str,
    in_channels: int = 11,
    timeseries_len: int = 24,
    checkpoint: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> nn.Module:
    """Build FireCastNet; ``checkpoint`` loads an official ``.ckpt`` (its configuration wins)."""
    kwargs.pop("name", None)
    if task.lower() not in {"segmentation", "regression"}:
        raise ValueError(f"firecastnet supports task='segmentation' (burned-area presence) or 'regression', got {task!r}.")
    if checkpoint is not None:
        return load_official_checkpoint(checkpoint, icospheres_graph_path=kwargs.get("icospheres_graph_path"))
    if "mesh_levels" in kwargs:
        kwargs["mesh_levels"] = tuple(kwargs["mesh_levels"])
    return FireCastNet(in_channels=in_channels, timeseries_len=timeseries_len, **kwargs)


__all__ = [
    "CubeConv3d",
    "FireCastNet",
    "GraphCastNet",
    "OFFICIAL_INPUT_VARIABLES",
    "firecastnet_builder",
    "lightning_state_dict_to_firecastnet",
    "load_official_checkpoint",
]
