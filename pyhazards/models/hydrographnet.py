from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class KAN(nn.Module):
    """
    Lightweight KAN-style harmonic basis encoder for node features.
    """

    def __init__(self, in_dim: int, harmonics: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.in_dim = in_dim
        self.harmonics = harmonics
        self.feature_proj = nn.ModuleList(
            [nn.Linear(2 * harmonics + 1, hidden_dim) for _ in range(in_dim)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F)
        outputs = []
        for i in range(self.in_dim):
            xi = x[:, :, i].unsqueeze(-1)
            basis = [torch.ones_like(xi)]
            for k in range(1, self.harmonics + 1):
                basis.append(torch.sin(k * xi))
                basis.append(torch.cos(k * xi))
            basis = torch.cat(basis, dim=-1)
            outputs.append(self.feature_proj[i](basis))
        return torch.stack(outputs, dim=0).sum(dim=0)


class GNBlock(nn.Module):
    """
    Message-passing block with residual edge and node updates.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.edge_mlp = MLP(3 * hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.node_mlp = MLP(2 * hidden_dim, hidden_dim, hidden_dim, dropout=dropout)

    def forward(
        self,
        node: torch.Tensor,
        edge: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sender_feat = node[:, senders, :]
        receiver_feat = node[:, receivers, :]

        edge_input = torch.cat([edge, sender_feat, receiver_feat], dim=-1)
        edge = edge + self.edge_mlp(edge_input)

        agg = torch.zeros_like(node)
        agg.index_add_(1, receivers, edge)

        # Degree-normalized aggregation improves stability when graph density changes.
        deg = torch.zeros(node.size(1), device=node.device, dtype=node.dtype)
        deg.index_add_(0, receivers, torch.ones_like(receivers, dtype=node.dtype))
        agg = agg / deg.clamp(min=1.0).view(1, -1, 1)

        node_input = torch.cat([node, agg], dim=-1)
        node = node + self.node_mlp(node_input)
        return node, edge


class HydroGraphNet(nn.Module):
    """
    PhysicsNeMo-inspired HydroGraphNet:
    encoder -> message-passing processor -> residual delta-state decoder.

    Supports one-step forward prediction and autoregressive rollout.
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        out_dim: int,
        hidden_dim: int = 64,
        harmonics: int = 5,
        num_gn_blocks: int = 5,
        state_dim: Optional[int] = None,
        rollout_steps: int = 1,
        enforce_nonnegative: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_in_dim = int(node_in_dim)
        self.edge_in_dim = int(edge_in_dim)
        self.out_dim = int(out_dim)
        self.state_dim = int(state_dim) if state_dim is not None else min(2, self.node_in_dim)
        self.state_dim = max(1, min(self.state_dim, self.node_in_dim))
        if self.out_dim > self.state_dim:
            raise ValueError(
                f"out_dim={self.out_dim} cannot exceed residual state_dim={self.state_dim}."
            )
        self.rollout_steps = max(1, int(rollout_steps))
        self.enforce_nonnegative = bool(enforce_nonnegative)

        # Encoder
        self.node_encoder = KAN(
            in_dim=self.node_in_dim,
            hidden_dim=hidden_dim,
            harmonics=harmonics,
        )
        self.edge_encoder = MLP(
            in_dim=self.edge_in_dim,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Processor
        self.processor = nn.ModuleList(
            [GNBlock(hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_gn_blocks)]
        )

        # Decoder predicts delta of physically meaningful states.
        self.decoder = MLP(
            in_dim=hidden_dim,
            out_dim=self.state_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def _edge_index(self, adj: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if adj.dim() == 2:
            a = adj
        elif adj.dim() == 3:
            if adj.size(0) != batch_size:
                raise ValueError(f"adj batch size mismatch: got {adj.size(0)}, expected {batch_size}")
            a = adj[0]
            for i in range(1, batch_size):
                if not torch.allclose(adj[i], a):
                    raise ValueError(
                        "Per-sample varying adjacency is not supported yet. "
                        "Provide a shared (N, N) adjacency or identical (B, N, N) adjacency."
                    )
        else:
            raise ValueError("adj must be shaped (N, N) or (B, N, N).")

        a = (a > 0).to(dtype=torch.bool)
        a.fill_diagonal_(True)
        return a.nonzero(as_tuple=True)

    def _match_edge_dim(self, edge_feat: torch.Tensor) -> torch.Tensor:
        # edge_feat: (B, E, F_edge_raw)
        f_raw = edge_feat.size(-1)
        if f_raw == self.edge_in_dim:
            return edge_feat
        if f_raw > self.edge_in_dim:
            return edge_feat[..., : self.edge_in_dim]
        pad = torch.zeros(
            edge_feat.size(0),
            edge_feat.size(1),
            self.edge_in_dim - f_raw,
            device=edge_feat.device,
            dtype=edge_feat.dtype,
        )
        return torch.cat([edge_feat, pad], dim=-1)

    def _prepare_edge_inputs(
        self,
        batch: Dict[str, torch.Tensor],
        senders: torch.Tensor,
        receivers: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        edge_attr = batch.get("edge_attr")
        if edge_attr is not None:
            edge_attr = edge_attr.to(device=device, dtype=dtype)
            if edge_attr.dim() == 2:
                edge_attr = edge_attr.unsqueeze(0).expand(batch_size, -1, -1)
            elif edge_attr.dim() == 3 and edge_attr.size(0) == 1 and batch_size > 1:
                edge_attr = edge_attr.expand(batch_size, -1, -1)
            if edge_attr.dim() != 3:
                raise ValueError("edge_attr must be shaped (E, F_edge) or (B, E, F_edge).")
            if edge_attr.size(1) != senders.numel():
                raise ValueError(
                    f"edge_attr edge count mismatch: got {edge_attr.size(1)}, expected {senders.numel()}."
                )
            return self._match_edge_dim(edge_attr)

        # Derive geometric edge features from coords: [dx, dy, distance]
        coords = batch.get("coords")
        if coords is None:
            edge_feat = torch.zeros(batch_size, senders.numel(), 3, device=device, dtype=dtype)
            return self._match_edge_dim(edge_feat)

        coords = coords.to(device=device, dtype=dtype)
        if coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(batch_size, -1, -1)
        elif coords.dim() == 3 and coords.size(0) == 1 and batch_size > 1:
            coords = coords.expand(batch_size, -1, -1)
        if coords.dim() != 3:
            raise ValueError("coords must be shaped (N, 2) or (B, N, 2).")

        src = coords[:, senders, :]
        dst = coords[:, receivers, :]
        delta = src - dst
        dist = torch.norm(delta, dim=-1, keepdim=True)
        edge_feat = torch.cat([delta, dist], dim=-1)
        return self._match_edge_dim(edge_feat)

    def _one_step(
        self,
        node_x: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # node_x: (B, N, F)
        if node_x.ndim != 3:
            raise ValueError(f"Expected node_x with shape (B,N,F), got {tuple(node_x.shape)}")
        if node_x.size(-1) < self.state_dim:
            raise ValueError(
                f"Input feature dim {node_x.size(-1)} is smaller than state_dim {self.state_dim}."
            )

        adj = batch.get("adj")
        if adj is None:
            raise ValueError("HydroGraphNet requires `adj` in the batch.")
        adj = adj.to(device=node_x.device)

        senders, receivers = self._edge_index(adj, batch_size=node_x.size(0))

        # ---- encoder ----
        node = self.node_encoder(node_x)
        edge_in = self._prepare_edge_inputs(
            batch=batch,
            senders=senders,
            receivers=receivers,
            batch_size=node.size(0),
            device=node.device,
            dtype=node.dtype,
        )
        edge = self.edge_encoder(edge_in)

        # ---- processor ----
        for gn in self.processor:
            node, edge = gn(node, edge, senders, receivers)

        # ---- decoder: residual state update ----
        delta_state = self.decoder(node)  # (B, N, state_dim)
        prev_state = node_x[..., : self.state_dim]
        next_state = prev_state + delta_state
        if self.enforce_nonnegative:
            next_state = next_state.clamp_min(0.0)

        # Return requested targets from the evolved state.
        return next_state[..., : self.out_dim]

    def rollout(self, batch: Dict[str, torch.Tensor], predict_steps: int) -> torch.Tensor:
        batch_roll = dict(batch)
        batch_roll["predict_steps"] = int(predict_steps)
        out = self.forward(batch_roll)
        if out.ndim != 4:
            raise RuntimeError("rollout expected stacked output with shape (B, S, N, out_dim).")
        return out

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # batch["x"]: (B, T, N, F)
        x = batch["x"]
        if x.ndim != 4:
            raise ValueError(
                f"HydroGraphNet expects x shaped (B, T, N, F), got {tuple(x.shape)}"
            )

        predict_steps = int(batch.get("predict_steps", self.rollout_steps))
        predict_steps = max(1, predict_steps)

        history = x
        preds = []
        for _ in range(predict_steps):
            node_x = history[:, -1]
            y_next = self._one_step(node_x=node_x, batch=batch)  # (B, N, out_dim)
            preds.append(y_next)

            if predict_steps > 1:
                next_frame = history[:, -1].clone()
                next_frame[..., : self.out_dim] = y_next
                history = torch.cat([history[:, 1:], next_frame.unsqueeze(1)], dim=1)

        if predict_steps == 1:
            return preds[0]
        return torch.stack(preds, dim=1)


class HydroGraphNetLoss(nn.Module):
    """
    Supervised regression loss with optional continuity regularization.
    """

    def __init__(self, supervised_weight: float = 1.0, continuity_weight: float = 0.0):
        super().__init__()
        self.supervised_weight = float(supervised_weight)
        self.continuity_weight = float(continuity_weight)

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        prev_state: Optional[torch.Tensor] = None,
        cell_area: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        supervised = F.mse_loss(preds, targets)
        total = self.supervised_weight * supervised
        metrics: Dict[str, float] = {"mse": float(supervised.detach().cpu())}

        if (
            self.continuity_weight > 0
            and prev_state is not None
            and cell_area is not None
            and preds.size(-1) >= 2
            and prev_state.size(-1) >= 2
        ):
            # Approximate local continuity: depth-change * area ~= volume-change
            depth_delta = preds[..., 0] - prev_state[..., 0]
            volume_delta = preds[..., 1] - prev_state[..., 1]
            area = cell_area.to(device=preds.device, dtype=preds.dtype)
            if area.dim() == 1:
                area = area.unsqueeze(0)
            continuity = F.mse_loss(depth_delta * area, volume_delta)
            total = total + self.continuity_weight * continuity
            metrics["continuity"] = float(continuity.detach().cpu())

        metrics["total"] = float(total.detach().cpu())
        return total, metrics


def hydrographnet_builder(
    task: str,
    node_in_dim: int,
    edge_in_dim: int,
    out_dim: int,
    **kwargs,
) -> HydroGraphNet:
    if task != "regression":
        raise ValueError("HydroGraphNet only supports regression")

    return HydroGraphNet(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        out_dim=out_dim,
        hidden_dim=kwargs.get("hidden_dim", 64),
        harmonics=kwargs.get("harmonics", 5),
        num_gn_blocks=kwargs.get("num_gn_blocks", 5),
        state_dim=kwargs.get("state_dim"),
        rollout_steps=kwargs.get("rollout_steps", 1),
        enforce_nonnegative=kwargs.get("enforce_nonnegative", False),
        dropout=kwargs.get("dropout", 0.0),
    )


__all__ = ["HydroGraphNet", "HydroGraphNetLoss", "hydrographnet_builder"]
