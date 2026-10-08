"""Icosahedral multi-mesh and grid/mesh graphs for FireCastNet, in plain PyTorch (no DGL).

FireCastNet (Michail et al., Scientific Reports 2025) runs a GraphCast encoder-processor-decoder
between a regular latitude/longitude grid and an icosahedral multi-mesh. This module builds the
three graphs it needs as ``(src, dst)`` index tensors plus their input features:

- the multi-mesh: union of the edges of icospheres of several refinement levels, made
  bidirectional, on the vertices of the finest level;
- grid -> mesh: each grid node to its 4 nearest mesh vertices, kept when closer than 0.6 times
  the longest edge of the finest icosphere;
- mesh -> grid: each grid node from its single nearest mesh vertex.

Provenance and licenses
-----------------------
- The coordinate helpers (``latlon2xyz``, ``xyz2latlon``, ``geospatial_rotation``,
  ``azimuthal_angle``, ``polar_angle``), ``edge_features``, ``node_features`` and the multi-mesh /
  grid-to-mesh construction are ported from NVIDIA Modulus v0.5.0
  (``modulus/utils/graphcast/graph_utils.py`` and ``modulus/utils/graphcast/graph.py``),
  Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES, licensed under the Apache License 2.0
  (http://www.apache.org/licenses/LICENSE-2.0). Changes: DGL graphs are replaced by index
  tensors, the scikit-learn KD-tree by a SciPy KD-tree with exact float64 re-ranking, and the
  mesh-to-grid graph links each grid node to its single nearest mesh vertex as FireCastNet does
  (Modulus links the three vertices of the nearest face).
- ``icosphere`` reimplements the icosphere layout of PyMesh ``generate_icosphere``, which produced
  the official ``icospheres/*.json.gz`` files of FireCastNet: midpoint subdivision of the flat
  icosahedron (new vertices numbered in order of first use, faces split as
  ``[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]``) followed by a single projection of all
  vertices onto the unit sphere. The oracle test checks that the generated meshes equal those files.
"""

from __future__ import annotations

import gzip
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

_PHI = (1.0 + math.sqrt(5.0)) / 2.0

# Regular icosahedron (12 vertices, 20 faces) in the layout used by PyMesh's generate_icosphere.
_ICOSAHEDRON_VERTICES = np.array(
    [
        [-1.0, _PHI, 0.0],
        [1.0, _PHI, 0.0],
        [-1.0, -_PHI, 0.0],
        [1.0, -_PHI, 0.0],
        [0.0, -1.0, _PHI],
        [0.0, 1.0, _PHI],
        [0.0, -1.0, -_PHI],
        [0.0, 1.0, -_PHI],
        [_PHI, 0.0, -1.0],
        [_PHI, 0.0, 1.0],
        [-_PHI, 0.0, -1.0],
        [-_PHI, 0.0, 1.0],
    ],
    dtype=np.float64,
)
_ICOSAHEDRON_FACES = np.array(
    [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [5, 4, 9], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ],
    dtype=np.int64,
)


def _subdivide(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split every triangle into four at its edge midpoints (no projection)."""
    midpoint: Dict[Tuple[int, int], int] = {}
    edges: List[Tuple[int, int]] = []
    base = len(vertices)

    def mid(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        index = midpoint.get(key)
        if index is None:
            index = base + len(edges)
            midpoint[key] = index
            edges.append(key)
        return index

    new_faces = np.empty((4 * len(faces), 3), dtype=np.int64)
    for i, (a, b, c) in enumerate(faces.tolist()):
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        new_faces[4 * i : 4 * i + 4] = [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
    pairs = np.asarray(edges, dtype=np.int64)
    midpoints = 0.5 * (vertices[pairs[:, 0]] + vertices[pairs[:, 1]])
    return np.concatenate((vertices, midpoints)), new_faces


def icosphere(level: int) -> Tuple[np.ndarray, np.ndarray]:
    """Unit icosphere after ``level`` refinements: float64 vertices ``(V, 3)``, int64 faces ``(F, 3)``.

    Level ``k`` has ``10 * 4**k + 2`` vertices and ``20 * 4**k`` faces; the vertices of level ``k``
    are the first vertices of every finer level.
    """
    if level < 0:
        raise ValueError(f"icosphere level must be >= 0, got {level}")
    vertices, faces = _ICOSAHEDRON_VERTICES, _ICOSAHEDRON_FACES
    for _ in range(level):
        vertices, faces = _subdivide(vertices, faces)
    return vertices / np.linalg.norm(vertices, axis=1, keepdims=True), faces.copy()


def icospheres(levels: Sequence[int] = (0, 1, 2, 3, 4, 5, 6)) -> Dict[str, np.ndarray]:
    """Icospheres of the given refinement levels in the layout of FireCastNet's icosphere files.

    Keys are ``order_{i}_vertices`` / ``order_{i}_faces`` with ``i = 0 .. len(levels) - 1`` in the
    given order, as in ``icospheres/icospheres_0_1_2_3_4_5_6.json.gz``.
    """
    levels = [int(level) for level in levels]
    if not levels:
        raise ValueError("at least one icosphere level is required")
    if any(b <= a for a, b in zip(levels, levels[1:])):
        raise ValueError(f"icosphere levels must be strictly increasing, got {levels}")
    out: Dict[str, np.ndarray] = {}
    vertices, faces = _ICOSAHEDRON_VERTICES, _ICOSAHEDRON_FACES
    current = 0
    for order, level in enumerate(levels):
        while current < level:
            vertices, faces = _subdivide(vertices, faces)
            current += 1
        out[f"order_{order}_vertices"] = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        out[f"order_{order}_faces"] = faces.copy()
    return out


def load_icospheres(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Read a mesh file in FireCastNet's icosphere JSON layout (``.json`` or ``.json.gz``).

    This is how the local-area-modelling (LAM) meshes of the official repository are used.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        raw = json.load(handle)
    return {key: np.asarray(value) for key, value in raw.items() if re.fullmatch(r"order_\d+_(vertices|faces)", key)}


def mesh_levels(meshes: Mapping[str, np.ndarray]) -> int:
    """Number of refinement levels (``order_{i}_faces`` entries) in an icosphere mapping."""
    count = 0
    while f"order_{count}_faces" in meshes:
        count += 1
    if count == 0 or f"order_{count - 1}_vertices" not in meshes:
        raise ValueError("icosphere mapping needs order_0_faces ... order_{k}_faces and order_{k}_vertices")
    return count


# --- coordinate helpers (ported from NVIDIA Modulus graph_utils, Apache-2.0) -------------------


def deg2rad(deg: Tensor) -> Tensor:
    return deg * np.pi / 180


def rad2deg(rad: Tensor) -> Tensor:
    return rad * 180 / np.pi


def latlon2xyz(latlon: Tensor, radius: float = 1, unit: str = "deg") -> Tensor:
    """``(N, 2)`` latitude/longitude to ``(N, 3)`` Cartesian coordinates on a sphere."""
    if unit == "deg":
        latlon = deg2rad(latlon)
    elif unit != "rad":
        raise ValueError("Not a valid unit")
    lat, lon = latlon[:, 0], latlon[:, 1]
    x = radius * torch.cos(lat) * torch.cos(lon)
    y = radius * torch.cos(lat) * torch.sin(lon)
    z = radius * torch.sin(lat)
    return torch.stack((x, y, z), dim=1)


def xyz2latlon(xyz: Tensor, radius: float = 1, unit: str = "deg") -> Tensor:
    """``(N, 3)`` Cartesian coordinates to ``(N, 2)`` latitude/longitude."""
    lat = torch.arcsin(xyz[:, 2] / radius)
    lon = torch.arctan2(xyz[:, 1], xyz[:, 0])
    if unit == "deg":
        return torch.stack((rad2deg(lat), rad2deg(lon)), dim=1)
    if unit == "rad":
        return torch.stack((lat, lon), dim=1)
    raise ValueError("Not a valid unit")


def geospatial_rotation(invar: Tensor, theta: Tensor, axis: str) -> Tensor:
    """Rotate ``(N, 3)`` points by per-point angles ``theta`` (radians) about ``axis`` (right-hand rule)."""
    invar = torch.unsqueeze(invar, -1)
    rotation = torch.zeros((theta.size(0), 3, 3), dtype=invar.dtype)
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    if axis == "x":
        rotation[:, 0, 0] += 1.0
        rotation[:, 1, 1] += cos
        rotation[:, 1, 2] -= sin
        rotation[:, 2, 1] += sin
        rotation[:, 2, 2] += cos
    elif axis == "y":
        rotation[:, 0, 0] += cos
        rotation[:, 0, 2] += sin
        rotation[:, 1, 1] += 1.0
        rotation[:, 2, 0] -= sin
        rotation[:, 2, 2] += cos
    elif axis == "z":
        rotation[:, 0, 0] += cos
        rotation[:, 0, 1] -= sin
        rotation[:, 1, 0] += sin
        rotation[:, 1, 1] += cos
        rotation[:, 2, 2] += 1.0
    else:
        raise ValueError("Invalid axis")
    return torch.matmul(rotation, invar).squeeze(-1)


def azimuthal_angle(lon: Tensor) -> Tensor:
    return torch.where(lon >= 0.0, 2 * np.pi - lon, -lon)


def polar_angle(lat: Tensor) -> Tensor:
    return torch.where(lat >= 0.0, lat, 2 * np.pi + lat)


def edge_features(src_pos: Tensor, dst_pos: Tensor, src: Tensor, dst: Tensor, normalize: bool = True) -> Tensor:
    """GraphCast edge features: source-minus-destination displacement in the destination's local frame.

    Both endpoints are rotated so that the destination lands on ``(1, 0, 0)``; the features are the
    rotated displacement and its length, divided by the longest displacement of the graph.
    """
    src_pos, dst_pos = src_pos[src.long()], dst_pos[dst.long()]
    dst_latlon = xyz2latlon(dst_pos, unit="rad")
    dst_lat, dst_lon = dst_latlon[:, 0], dst_latlon[:, 1]
    theta_azimuthal = azimuthal_angle(dst_lon)
    theta_polar = polar_angle(dst_lat)
    src_pos = geospatial_rotation(src_pos, theta=theta_azimuthal, axis="z")
    dst_pos = geospatial_rotation(dst_pos, theta=theta_azimuthal, axis="z")
    src_pos = geospatial_rotation(src_pos, theta=theta_polar, axis="y")
    dst_pos = geospatial_rotation(dst_pos, theta=theta_polar, axis="y")
    disp = src_pos - dst_pos
    disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)
    if normalize:
        max_disp_norm = torch.max(disp_norm)
        return torch.cat((disp / max_disp_norm, disp_norm / max_disp_norm), dim=-1)
    return torch.cat((disp, disp_norm), dim=-1)


def node_features(pos: Tensor) -> Tensor:
    """GraphCast mesh-node features ``(cos(lat), sin(lon), cos(lon))``.

    As in the reference code, latitude and longitude are taken in degrees and passed to the
    trigonometric functions without conversion to radians.
    """
    latlon = xyz2latlon(pos)
    lat, lon = latlon[:, 0], latlon[:, 1]
    return torch.stack((torch.cos(lat), torch.sin(lon), torch.cos(lon)), dim=-1)


# --- graph construction -------------------------------------------------------------------------


def latlon_grid(sp_res: float, max_lat: float, min_lat: float, max_lon: float, min_lon: float) -> Tuple[Tensor, Tensor]:
    """Cell-centre latitudes (north to south) and longitudes (west to east) as float32 tensors."""
    latitudes = torch.from_numpy(np.arange(max_lat, min_lat - (sp_res / 2), -sp_res, dtype=np.float32))
    longitudes = torch.from_numpy(np.arange(min_lon, max_lon + (sp_res / 2), sp_res, dtype=np.float32))
    return latitudes, longitudes


def _nearest_vertices(queries: Tensor, points: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    """Exact k nearest ``points`` of every query (float64 Euclidean distance), sorted by distance.

    A SciPy KD-tree proposes candidates; the final order uses the directly computed distance
    ``sqrt(dx^2 + dy^2 + dz^2)`` (the formula of the scikit-learn KD-tree used by the reference),
    with ties broken by vertex index.
    """
    from scipy.spatial import cKDTree

    queries = queries.to(torch.float64)
    points = points.to(torch.float64)
    n_candidates = min(points.size(0), k + 8)
    _, candidates = cKDTree(points.numpy()).query(queries.numpy(), k=n_candidates)
    candidates = torch.from_numpy(np.asarray(candidates, dtype=np.int64).reshape(queries.size(0), n_candidates))
    candidates, _ = candidates.sort(dim=1)
    diff = queries[:, None, :] - points[candidates]
    sq = diff[..., 0] * diff[..., 0] + diff[..., 1] * diff[..., 1] + diff[..., 2] * diff[..., 2]
    order = sq.argsort(dim=1, stable=True)[:, :k]
    return torch.sqrt(sq.gather(1, order)), candidates.gather(1, order)


@dataclass
class FireCastNetGraph:
    """Edge indices (int64) and float32 input features of the three FireCastNet graphs."""

    mesh_src: Tensor
    mesh_dst: Tensor
    mesh_edata: Tensor
    mesh_ndata: Tensor
    g2m_src: Tensor
    g2m_dst: Tensor
    g2m_edata: Tensor
    m2g_src: Tensor
    m2g_dst: Tensor
    m2g_edata: Tensor
    num_grid_nodes: int
    num_mesh_nodes: int


def build_graph(latitudes: Tensor, longitudes: Tensor, meshes: Mapping[str, np.ndarray]) -> FireCastNetGraph:
    """Build the multi-mesh, grid->mesh and mesh->grid graphs for a latitude/longitude grid.

    Grid nodes are numbered row-major over ``(latitude, longitude)``; mesh nodes are the vertices of
    the finest level in ``meshes`` (an :func:`icospheres` / :func:`load_icospheres` mapping).
    """
    max_order = mesh_levels(meshes) - 1
    vertices = np.asarray(meshes[f"order_{max_order}_vertices"], dtype=np.float64)
    finest_faces = np.asarray(meshes[f"order_{max_order}_faces"], dtype=np.int64)
    num_mesh = vertices.shape[0]
    mesh_pos = torch.tensor(vertices, dtype=torch.float32)

    # Multi-mesh: every face contributes (a, b), (b, c), (c, a); the union over all levels is made
    # bidirectional and simple, with edges sorted by (src, dst).
    faces = np.concatenate([np.asarray(meshes[f"order_{i}_faces"], dtype=np.int64) for i in range(max_order + 1)])
    if faces.min() < 0 or faces.max() >= num_mesh:
        raise ValueError("icosphere faces must index the vertices of the finest level")
    src = faces[:, [0, 1, 2]].reshape(-1)
    dst = faces[:, [1, 2, 0]].reshape(-1)
    pairs = np.unique(np.concatenate((src * num_mesh + dst, dst * num_mesh + src)))
    mesh_src = torch.from_numpy(pairs // num_mesh)
    mesh_dst = torch.from_numpy(pairs % num_mesh)

    # Grid nodes on the unit sphere (float32, as the reference computes them).
    grid = torch.stack(torch.meshgrid(latitudes, longitudes, indexing="ij"), dim=-1)
    grid_flat = grid.permute(2, 0, 1).reshape(2, -1).permute(1, 0)
    grid_pos = latlon2xyz(grid_flat)
    num_grid = grid_pos.size(0)

    # Grid -> mesh: up to 4 nearest vertices within 0.6 x the longest edge of the finest level.
    edge_len = max(
        np.max(np.linalg.norm(vertices[finest_faces[:, a]] - vertices[finest_faces[:, b]], axis=1))
        for a, b in ((0, 1), (0, 2), (1, 2))
    )
    distances, neighbours = _nearest_vertices(grid_pos, torch.from_numpy(vertices), k=min(4, num_mesh))
    keep = distances <= 0.6 * edge_len
    g2m_src = torch.arange(num_grid).unsqueeze(1).expand_as(neighbours)[keep]
    g2m_dst = neighbours[keep]

    # Mesh -> grid: the nearest mesh vertex of every grid node.
    _, nearest = _nearest_vertices(grid_pos, torch.from_numpy(vertices), k=1)
    m2g_src = nearest[:, 0]
    m2g_dst = torch.arange(num_grid)

    return FireCastNetGraph(
        mesh_src=mesh_src,
        mesh_dst=mesh_dst,
        mesh_edata=edge_features(mesh_pos, mesh_pos, mesh_src, mesh_dst),
        mesh_ndata=node_features(mesh_pos),
        g2m_src=g2m_src,
        g2m_dst=g2m_dst,
        g2m_edata=edge_features(grid_pos, mesh_pos, g2m_src, g2m_dst),
        m2g_src=m2g_src,
        m2g_dst=m2g_dst,
        m2g_edata=edge_features(mesh_pos, grid_pos, m2g_src, m2g_dst),
        num_grid_nodes=num_grid,
        num_mesh_nodes=num_mesh,
    )


__all__ = [
    "FireCastNetGraph",
    "build_graph",
    "edge_features",
    "icosphere",
    "icospheres",
    "latlon2xyz",
    "latlon_grid",
    "load_icospheres",
    "mesh_levels",
    "node_features",
    "xyz2latlon",
]
