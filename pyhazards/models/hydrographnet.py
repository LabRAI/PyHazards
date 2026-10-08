"""HydroGraphNet (Taghizadeh et al., Computer-Aided Civil and Infrastructure Engineering 2025).

HydroGraphNet is the MeshGraphKAN model of NVIDIA PhysicsNeMo: a MeshGraphNet encoder-processor-decoder
whose node encoder is a Fourier Kolmogorov-Arnold network, trained on a water-volume continuity loss
and rolled out autoregressively over an unstructured flood mesh.

Ported from NVIDIA PhysicsNeMo @ eb7a329c246a48b6fb5acd569eaabeac53f915f7
(https://github.com/NVIDIA/physicsnemo, Apache-2.0, Copyright (c) 2023 - 2026 NVIDIA CORPORATION &
AFFILIATES): ``physicsnemo/models/meshgraphnet/{meshgraphkan,meshgraphnet}.py``,
``physicsnemo/nn/module/kan_layers.py``, ``physicsnemo/nn/module/gnn_layers/{mesh_graph_mlp,
mesh_edge_block,mesh_node_block,utils}.py`` and the HydroGraphNet example
``examples/weather/flood_modeling/hydrographnet/{utils.py (compute_physics_loss), inference.py
(rollout)}``. The port is plain PyTorch: PhysicsNeMo's PyTorch Geometric graph and ``torch_scatter``
aggregation are replaced by an ``edge_index`` tensor and ``index_add_``.

Parameter names, initialisation order and outputs equal PhysicsNeMo's, so PhysicsNeMo state dicts load
with ``strict=True`` (``tests/oracle/test_hydrographnet_oracle.py``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# PhysicsNeMo's ``ACT2FN`` entries without parameters (one module instance is shared by every MLP).
_ACTIVATIONS = {
    "relu": lambda: nn.ReLU(),
    "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.1),
    "relu6": lambda: nn.ReLU6(),
    "elu": lambda: nn.ELU(),
    "celu": lambda: nn.CELU(alpha=1.0),
    "selu": lambda: nn.SELU(),
    "silu": lambda: nn.SiLU(),
    "gelu": lambda: nn.GELU(),
    "sigmoid": lambda: nn.Sigmoid(),
    "logsigmoid": lambda: nn.LogSigmoid(),
    "softplus": lambda: nn.Softplus(),
    "softsign": lambda: nn.Softsign(),
    "tanh": lambda: nn.Tanh(),
    "tanhshrink": lambda: nn.Tanhshrink(),
    "hardtanh": lambda: nn.Hardtanh(),
    "identity": lambda: nn.Identity(),
}

# HydroGraphNet node features (HydroGraphDataset.create_node_features): 12 static / forcing columns
# followed by the water-depth and volume windows.
HYDROGRAPHNET_STATIC_FEATURES = (
    "x",
    "y",
    "area",
    "elevation",
    "slope",
    "aspect",
    "curvature",
    "manning",
    "flow_accumulation",
    "infiltration",
    "inflow",
    "precipitation",
)


def _activation(name: str) -> nn.Module:
    key = str(name).lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"Unknown mlp_activation_fn {name!r}; choose one of {sorted(_ACTIVATIONS)}.")
    return _ACTIVATIONS[key]()


def _edge_index(graph: Any) -> torch.Tensor:
    """``(2, num_edges)`` source / destination indices from a tensor or a graph object with ``edge_index``."""
    edge_index = getattr(graph, "edge_index", graph)
    if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        shape = tuple(edge_index.shape) if isinstance(edge_index, torch.Tensor) else type(edge_index).__name__
        raise ValueError(f"edge_index must be a (2, num_edges) tensor of source / destination nodes, got {shape}.")
    return edge_index.long()


class MeshGraphMLP(nn.Module):
    """Linear layers with a shared activation and an optional LayerNorm after the last layer.

    ``hidden_layers=2``: Linear(in, hidden), act, Linear(hidden, hidden), act, Linear(hidden, out), LayerNorm.
    """

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
            layers: List[nn.Module] = [nn.Linear(input_dim, hidden_dim), activation_fn]
            self.hidden_layers = hidden_layers
            for _ in range(hidden_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), activation_fn]
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.norm_type = norm_type
            if norm_type is not None:
                if norm_type != "LayerNorm":
                    raise ValueError(f"norm_type must be 'LayerNorm' or None, got {norm_type!r}.")
                layers.append(nn.LayerNorm(output_dim))
            self.model = nn.Sequential(*layers)
        else:
            self.model = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MeshGraphEdgeMLPConcat(MeshGraphMLP):
    """Edge MLP on ``cat(edge, source node, destination node)``."""

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

    def forward(self, efeat: torch.Tensor, nfeat: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        return self.model(torch.cat((efeat, nfeat[src], nfeat[dst]), dim=1))


class MeshEdgeBlock(nn.Module):
    """Residual edge update: ``e + MLP(cat(e, n_src, n_dst))``."""

    def __init__(
        self,
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: Optional[nn.Module] = None,
        norm_type: Optional[str] = "LayerNorm",
    ):
        super().__init__()
        self.edge_mlp = MeshGraphEdgeMLPConcat(
            efeat_dim=input_dim_edges,
            src_dim=input_dim_nodes,
            dst_dim=input_dim_nodes,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

    def forward(
        self, efeat: torch.Tensor, nfeat: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.edge_mlp(efeat, nfeat, edge_index) + efeat, nfeat


class MeshNodeBlock(nn.Module):
    """Residual node update: ``n + MLP(cat(aggregate(incoming edges), n))``, summing (or averaging) at the destination."""

    def __init__(
        self,
        aggregation: str = "sum",
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: Optional[nn.Module] = None,
        norm_type: Optional[str] = "LayerNorm",
    ):
        super().__init__()
        if aggregation not in {"sum", "mean"}:
            raise ValueError(f"aggregation must be 'sum' or 'mean', got {aggregation!r}.")
        self.aggregation = aggregation
        self.node_mlp = MeshGraphMLP(
            input_dim=input_dim_nodes + input_dim_edges,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

    def forward(
        self, efeat: torch.Tensor, nfeat: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dst = edge_index[1]
        # torch_scatter.scatter(efeat, dst, dim=0, dim_size=num_nodes, reduce=aggregation).
        aggregated = efeat.new_zeros(nfeat.shape[0], efeat.shape[1]).index_add_(0, dst, efeat)
        if self.aggregation == "mean":
            count = efeat.new_zeros(nfeat.shape[0]).index_add_(0, dst, efeat.new_ones(dst.shape[0]))
            aggregated = aggregated / count.clamp(min=1).unsqueeze(-1)
        return efeat, self.node_mlp(torch.cat((aggregated, nfeat), dim=-1)) + nfeat


class MeshGraphNetProcessor(nn.Module):
    """``processor_size`` alternating edge and node blocks (``processor_layers`` = [edge, node, edge, node, ...])."""

    def __init__(
        self,
        processor_size: int = 15,
        input_dim_node: int = 128,
        input_dim_edge: int = 128,
        num_layers_node: int = 2,
        num_layers_edge: int = 2,
        aggregation: str = "sum",
        norm_type: Optional[str] = "LayerNorm",
        activation_fn: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.processor_size = processor_size
        self.input_dim_node = input_dim_node
        self.input_dim_edge = input_dim_edge
        activation_fn = nn.ReLU() if activation_fn is None else activation_fn
        # As in PhysicsNeMo, every edge block is created before the node blocks (initialisation order).
        edge_blocks = [
            MeshEdgeBlock(input_dim_node, input_dim_edge, input_dim_edge, input_dim_edge, num_layers_edge, activation_fn, norm_type)
            for _ in range(processor_size)
        ]
        node_blocks = [
            MeshNodeBlock(aggregation, input_dim_node, input_dim_edge, input_dim_edge, input_dim_edge, num_layers_node, activation_fn, norm_type)
            for _ in range(processor_size)
        ]
        self.processor_layers = nn.ModuleList([block for pair in zip(edge_blocks, node_blocks) for block in pair])
        self.num_processor_layers = len(self.processor_layers)

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if node_features.ndim != 2 or node_features.shape[1] != self.input_dim_node:
            raise ValueError(
                f"Expected tensor of shape (N_nodes, {self.input_dim_node}) but got tensor of shape {tuple(node_features.shape)}"
            )
        if edge_features.ndim != 2 or edge_features.shape[1] != self.input_dim_edge:
            raise ValueError(
                f"Expected tensor of shape (N_edges, {self.input_dim_edge}) but got tensor of shape {tuple(edge_features.shape)}"
            )
        for layer in self.processor_layers:
            edge_features, node_features = layer(edge_features, node_features, edge_index)
        return node_features


class KolmogorovArnoldNetwork(nn.Module):
    """Fourier KAN layer: ``y_o = sum_i sum_k a_oik cos(k x_i) + b_oik sin(k x_i) + bias_o``, k = 1..num_harmonics."""

    def __init__(self, input_dim: int, output_dim: int, num_harmonics: int = 5, add_bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_harmonics = num_harmonics
        self.add_bias = add_bias
        # [2, output_dim, input_dim, num_harmonics]: cosine and sine coefficients.
        self.fourier_coeffs = nn.Parameter(
            torch.randn(2, output_dim, input_dim, num_harmonics) / (np.sqrt(input_dim) * np.sqrt(num_harmonics))
        )
        if self.add_bias:
            self.bias = nn.Parameter(torch.zeros(1, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.view(batch_size, self.input_dim, 1)
        k = torch.arange(1, self.num_harmonics + 1, device=x.device).view(1, 1, self.num_harmonics)
        y_cos = torch.einsum("bij,oij->bo", torch.cos(k * x), self.fourier_coeffs[0])
        y_sin = torch.einsum("bij,oij->bo", torch.sin(k * x), self.fourier_coeffs[1])
        y = y_cos + y_sin
        if self.add_bias:
            y = y + self.bias
        return y


class HydroGraphNet(nn.Module):
    """HydroGraphNet: PhysicsNeMo's MeshGraphKAN.

    ``forward(node_features (N, input_dim_nodes), edge_features (E, input_dim_edges), edge_index (2, E))``
    returns ``(N, output_dim)``; for HydroGraphNet the outputs are the normalised changes of water depth
    and volume over one 20-minute step. ``edge_index`` holds source (row 0) and destination (row 1) node
    indices; messages are summed at the destination. A PyTorch Geometric ``Data`` (anything with an
    ``edge_index`` attribute) is accepted in its place, and a single mapping with the keys
    ``node_features``, ``edge_features`` and ``edge_index`` (the PyHazards mesh batches) as the only
    argument.

    Defaults are the HydroGraphNet configuration (``conf/config.yaml``): 16 node features, 3 edge
    features, 2 outputs, hidden width 128, 15 message-passing blocks, 2 hidden layers per MLP, ReLU,
    LayerNorm, sum aggregation and 5 harmonics (2,318,722 parameters).
    """

    def __init__(
        self,
        input_dim_nodes: int = 16,
        input_dim_edges: int = 3,
        output_dim: int = 2,
        processor_size: int = 15,
        mlp_activation_fn: str = "relu",
        num_layers_node_processor: int = 2,
        num_layers_edge_processor: int = 2,
        hidden_dim_processor: int = 128,
        hidden_dim_node_encoder: int = 128,
        num_layers_node_encoder: Optional[int] = 2,
        hidden_dim_edge_encoder: int = 128,
        num_layers_edge_encoder: Optional[int] = 2,
        hidden_dim_node_decoder: int = 128,
        num_layers_node_decoder: Optional[int] = 2,
        aggregation: str = "sum",
        num_harmonics: int = 5,
    ):
        super().__init__()
        if num_layers_node_encoder is None or num_layers_edge_encoder is None or num_layers_node_decoder is None:
            raise ValueError("num_layers_node_encoder, num_layers_edge_encoder and num_layers_node_decoder cannot be None")
        self.input_dim_nodes = input_dim_nodes
        self.input_dim_edges = input_dim_edges
        self.output_dim = output_dim
        activation_fn = _activation(mlp_activation_fn)

        self.edge_encoder = MeshGraphMLP(
            input_dim_edges,
            output_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_edge_encoder,
            hidden_layers=num_layers_edge_encoder,
            activation_fn=activation_fn,
            norm_type="LayerNorm",
        )
        # MeshGraphKAN builds MeshGraphNet's MLP node encoder and only then replaces it with the KAN;
        # creating (and dropping) it keeps the random number stream, and the position of ``node_encoder``
        # in the state dict, identical to PhysicsNeMo's.
        self.node_encoder = MeshGraphMLP(
            input_dim_nodes,
            output_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_node_encoder,
            hidden_layers=num_layers_node_encoder,
            activation_fn=activation_fn,
            norm_type="LayerNorm",
        )
        self.node_decoder = MeshGraphMLP(
            hidden_dim_processor,
            output_dim=output_dim,
            hidden_dim=hidden_dim_node_decoder,
            hidden_layers=num_layers_node_decoder,
            activation_fn=activation_fn,
            norm_type=None,
        )
        self.processor = MeshGraphNetProcessor(
            processor_size=processor_size,
            input_dim_node=hidden_dim_processor,
            input_dim_edge=hidden_dim_processor,
            num_layers_node=num_layers_node_processor,
            num_layers_edge=num_layers_edge_processor,
            aggregation=aggregation,
            norm_type="LayerNorm",
            activation_fn=activation_fn,
        )
        self.node_encoder = KolmogorovArnoldNetwork(
            input_dim=input_dim_nodes,
            output_dim=hidden_dim_processor,
            num_harmonics=num_harmonics,
            add_bias=True,
        )

    def forward(
        self,
        node_features: Union[torch.Tensor, Mapping[str, Any]],
        edge_features: Optional[torch.Tensor] = None,
        graph: Any = None,
    ) -> torch.Tensor:
        if isinstance(node_features, Mapping):
            batch = node_features
            missing = [key for key in ("node_features", "edge_features", "edge_index") if key not in batch]
            if missing:
                raise ValueError(f"HydroGraphNet mesh batches need the keys node_features, edge_features and edge_index; missing {missing}.")
            node_features, edge_features, graph = batch["node_features"], batch["edge_features"], batch["edge_index"]
        if edge_features is None or graph is None:
            raise ValueError("HydroGraphNet needs node_features (N, F), edge_features (E, F_edge) and edge_index (2, E).")
        if node_features.ndim != 2 or node_features.shape[1] != self.input_dim_nodes:
            raise ValueError(
                f"Expected tensor of shape (N_nodes, {self.input_dim_nodes}) but got tensor of shape {tuple(node_features.shape)}"
            )
        if edge_features.ndim != 2 or edge_features.shape[1] != self.input_dim_edges:
            raise ValueError(
                f"Expected tensor of shape (N_edges, {self.input_dim_edges}) but got tensor of shape {tuple(edge_features.shape)}"
            )
        edge_index = _edge_index(graph)
        if edge_index.shape[1] != edge_features.shape[0]:
            raise ValueError(
                f"edge_index has {edge_index.shape[1]} edges but edge_features has shape {tuple(edge_features.shape)}."
            )
        edge_features = self.edge_encoder(edge_features)
        node_features = self.node_encoder(node_features)
        x = self.processor(node_features, edge_features, edge_index)
        return self.node_decoder(x)

    @torch.no_grad()
    def rollout(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: Any,
        inflow: torch.Tensor,
        precipitation: torch.Tensor,
        n_time_steps: int = 2,
        n_static: int = 12,
    ) -> Dict[str, torch.Tensor]:
        """Autoregressive rollout of the HydroGraphNet example (``inference.py``).

        ``node_features`` (N, 12 + 2 * n_time_steps) is the first window of a test hydrograph; ``inflow``
        and ``precipitation`` (rollout_length,) are the normalised forcings from step ``n_time_steps`` on.
        Each step predicts the depth / volume change, appends ``last + change`` to the windows (dropping
        the oldest), and writes ``inflow[t]`` / ``precipitation[t]`` into columns 10 and 11. Returns the
        normalised ``water_depth`` and ``volume`` of every step, ``(rollout_length, N)`` each.
        """
        if node_features.ndim != 2 or node_features.shape[1] != n_static + 2 * n_time_steps:
            raise ValueError(
                f"rollout expects node_features of shape (N_nodes, {n_static + 2 * n_time_steps}), got {tuple(node_features.shape)}."
            )
        if inflow.ndim != 1 or precipitation.shape != inflow.shape:
            raise ValueError(
                f"inflow and precipitation must be 1-D tensors of equal length, got {tuple(inflow.shape)} and {tuple(precipitation.shape)}."
            )
        num_nodes = node_features.size(0)
        x_iter = node_features.clone()
        depths, volumes = [], []
        for t in range(inflow.shape[0]):
            static_part = x_iter[:, :n_static]
            water_depth_window = x_iter[:, n_static : n_static + n_time_steps]
            volume_window = x_iter[:, n_static + n_time_steps : n_static + 2 * n_time_steps]
            x_input = torch.cat([static_part, water_depth_window, volume_window], dim=1)
            pred = self(x_input, edge_features, edge_index)
            new_wd = water_depth_window[:, -1:] + pred[:, 0:1]
            new_vol = volume_window[:, -1:] + pred[:, 1:2]
            water_depth_updated = torch.cat([water_depth_window[:, 1:], new_wd], dim=1)
            volume_updated = torch.cat([volume_window[:, 1:], new_vol], dim=1)
            new_flow = inflow[t].unsqueeze(0).expand(num_nodes, 1)
            new_precip = precipitation[t].unsqueeze(0).expand(num_nodes, 1)
            static_part_updated = static_part.clone()
            static_part_updated[:, 10:12] = torch.cat([new_flow, new_precip], dim=1)
            x_iter = torch.cat([static_part_updated, water_depth_updated, volume_updated], dim=1)
            depths.append(new_wd.squeeze(1))
            volumes.append(new_vol.squeeze(1))
        return {"water_depth": torch.stack(depths), "volume": torch.stack(volumes)}


MeshGraphKAN = HydroGraphNet

PHYSICS_KEYS = (
    "past_volume",
    "future_volume",
    "avg_inflow",
    "avg_precipitation",
    "next_inflow",
    "next_precip",
    "volume_mean",
    "volume_std",
    "num_nodes",
    "area_sum",
    "infiltration_area_sum",
)


def hydrographnet_physics_loss(
    pred: torch.Tensor,
    physics_data: Mapping[str, torch.Tensor],
    batch: Optional[torch.Tensor] = None,
    delta_t: float = 1200.0,
) -> torch.Tensor:
    """Volume-continuity loss of the HydroGraphNet example (``compute_physics_loss``).

    For each graph of the batch (``batch`` gives the graph index of every node; ``None`` means one graph)
    the predicted total volume ``V_pred = V_past + volume_std * sum(pred[:, 1])`` is compared, in
    physical units, with the volume entering over ``delta_t`` seconds:

    ``term1 = relu((V_pred - (V_past + delta_t (avg_inflow + avg_precipitation * A_inf))) / A)^2``,
    ``term2 = relu((V_future - V_pred - delta_t (next_inflow + next_precip * A_inf)) / A)^2``,

    with ``V = volume_norm * volume_std + num_nodes * volume_mean``, ``A`` the total cell area and
    ``A_inf`` the infiltration-weighted area. Returns the mean of ``term1 + term2`` over graphs.
    ``physics_data`` holds one value per graph for each of :data:`PHYSICS_KEYS`.
    """
    if pred.ndim != 2 or pred.shape[1] < 2:
        raise ValueError(f"pred must have shape (N_nodes, >=2) with the volume change in column 1, got {tuple(pred.shape)}.")
    missing = [key for key in PHYSICS_KEYS if key not in physics_data]
    if missing:
        raise ValueError(f"physics_data is missing {missing}.")
    if batch is None:
        batch = torch.zeros(pred.shape[0], dtype=torch.long, device=pred.device)
    unique_ids = torch.unique(batch)
    predicted_diff = pred[:, 1]
    physics_losses = []
    for uid in unique_ids:
        mask = batch == uid
        pred_diff_sum = predicted_diff[mask].sum()
        idx = (unique_ids == uid).nonzero(as_tuple=False).item()
        values = {key: torch.as_tensor(physics_data[key]).reshape(-1)[idx] for key in PHYSICS_KEYS}
        past_volume_denorm = values["past_volume"] * values["volume_std"] + values["num_nodes"] * values["volume_mean"]
        future_volume_denorm = values["future_volume"] * values["volume_std"] + values["num_nodes"] * values["volume_mean"]
        pred_total_volume = past_volume_denorm + values["volume_std"] * pred_diff_sum
        new_precip_term = values["avg_precipitation"] * values["infiltration_area_sum"]
        new_next_precip_term = values["next_precip"] * values["infiltration_area_sum"]
        term1 = (
            F.relu(
                (pred_total_volume - (past_volume_denorm + delta_t * (values["avg_inflow"] + new_precip_term)))
                / values["area_sum"]
            )
            ** 2
        )
        term2 = (
            F.relu(
                (future_volume_denorm - pred_total_volume - delta_t * (values["next_inflow"] + new_next_precip_term))
                / values["area_sum"]
            )
            ** 2
        )
        physics_losses.append(term1 + term2)
    if physics_losses:
        return torch.stack(physics_losses).mean()
    return torch.tensor(0.0, device=pred.device)


class HydroGraphNetLoss(nn.Module):
    """Training loss of the HydroGraphNet example (``train.py``, ``noise_type: none``).

    ``MSE(pred, target) + physics_loss_weight * hydrographnet_physics_loss(pred, physics_data, batch)``;
    the physics term is skipped when ``physics_data`` is ``None`` (``use_physics_loss: false``). Defaults
    are the example's ``physics_loss_weight=1.0`` and ``delta_t=1200`` s. Returns the total loss and a
    dict of its parts.
    """

    def __init__(self, physics_loss_weight: float = 1.0, delta_t: float = 1200.0, use_physics_loss: bool = True):
        super().__init__()
        self.physics_loss_weight = float(physics_loss_weight)
        self.delta_t = float(delta_t)
        self.use_physics_loss = bool(use_physics_loss)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        physics_data: Optional[Mapping[str, torch.Tensor]] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mse_loss = F.mse_loss(pred, target)
        loss = mse_loss
        parts = {"total_loss": loss, "mse_loss": mse_loss}
        if self.use_physics_loss and physics_data is not None:
            phy_loss = hydrographnet_physics_loss(pred, physics_data, batch, delta_t=self.delta_t)
            loss = loss + self.physics_loss_weight * phy_loss
            parts = {"total_loss": loss, "mse_loss": mse_loss, "physics_loss": phy_loss}
        return loss, parts


def hydrographnet_builder(
    task: str,
    input_dim_nodes: int = 16,
    input_dim_edges: int = 3,
    output_dim: int = 2,
    processor_size: int = 15,
    mlp_activation_fn: str = "relu",
    num_layers_node_processor: int = 2,
    num_layers_edge_processor: int = 2,
    hidden_dim_processor: int = 128,
    hidden_dim_node_encoder: int = 128,
    num_layers_node_encoder: int = 2,
    hidden_dim_edge_encoder: int = 128,
    num_layers_edge_encoder: int = 2,
    hidden_dim_node_decoder: int = 128,
    num_layers_node_decoder: int = 2,
    aggregation: str = "sum",
    num_harmonics: int = 5,
    **kwargs: Any,
) -> HydroGraphNet:
    """HydroGraphNet (MeshGraphKAN) at the HydroGraphNet example configuration by default."""
    if task.lower() != "regression":
        raise ValueError(f"HydroGraphNet only supports task='regression', got {task!r}.")
    kwargs.pop("name", None)
    if kwargs:
        raise ValueError(f"Unknown HydroGraphNet arguments: {sorted(kwargs)}.")
    return HydroGraphNet(
        input_dim_nodes=input_dim_nodes,
        input_dim_edges=input_dim_edges,
        output_dim=output_dim,
        processor_size=processor_size,
        mlp_activation_fn=mlp_activation_fn,
        num_layers_node_processor=num_layers_node_processor,
        num_layers_edge_processor=num_layers_edge_processor,
        hidden_dim_processor=hidden_dim_processor,
        hidden_dim_node_encoder=hidden_dim_node_encoder,
        num_layers_node_encoder=num_layers_node_encoder,
        hidden_dim_edge_encoder=hidden_dim_edge_encoder,
        num_layers_edge_encoder=num_layers_edge_encoder,
        hidden_dim_node_decoder=hidden_dim_node_decoder,
        num_layers_node_decoder=num_layers_node_decoder,
        aggregation=aggregation,
        num_harmonics=num_harmonics,
    )


__all__ = [
    "HYDROGRAPHNET_STATIC_FEATURES",
    "HydroGraphNet",
    "HydroGraphNetLoss",
    "KolmogorovArnoldNetwork",
    "MeshGraphKAN",
    "PHYSICS_KEYS",
    "hydrographnet_builder",
    "hydrographnet_physics_loss",
]
