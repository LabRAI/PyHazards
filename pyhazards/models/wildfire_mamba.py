from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """
    Row-normalize an adjacency matrix and ensure self-loops.
    Accepts (N, N) or (B, N, N) and returns the same rank.
    """
    if adj.dim() == 2:
        adj = adj.unsqueeze(0)
    eye = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype)
    adj = adj.float() + eye.unsqueeze(0)
    return adj / adj.sum(-1, keepdim=True).clamp(min=1e-6)


class SelectiveSSMBlock(nn.Module):
    """
    Lightweight selective state-space block inspired by Mamba.

    Operates over a single temporal stream: (batch, time, features) -> (batch, time, hidden_dim).
    """

    def __init__(self, in_dim: int, hidden_dim: int, state_dim: int = 64, conv_kernel: int = 5, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.dwconv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=conv_kernel, padding=conv_kernel // 2, groups=hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.A = nn.Parameter(torch.randn(hidden_dim, state_dim) * 0.02)
        self.B = nn.Parameter(torch.randn(state_dim, hidden_dim) * 0.02)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        h = self.in_proj(x)  # (B, T, H)
        h_conv = self.dwconv(h.transpose(1, 2)).transpose(1, 2)
        g = torch.sigmoid(self.gate(h_conv))
        B, T, H = h_conv.shape
        state = torch.zeros(B, H, device=h_conv.device, dtype=h_conv.dtype)
        outputs = []
        for t in range(T):
            # selective update: gates decide how much new signal to mix into the running state
            state = g[:, t, :] * (state @ self.A @ self.B + h_conv[:, t, :]) + (1 - g[:, t, :]) * state
            outputs.append(state)
        y = torch.stack(outputs, dim=1)
        y = self.out_proj(self.drop(y)) + h_conv
        return self.norm(y)


class MambaTemporalEncoder(nn.Module):
    """Stack of selective SSM blocks; returns the last hidden state."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2, state_dim: int = 64, conv_kernel: int = 5, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SelectiveSSMBlock(
                    in_dim=in_dim if i == 0 else hidden_dim,
                    hidden_dim=hidden_dim,
                    state_dim=state_dim,
                    conv_kernel=conv_kernel,
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.blocks:
            h = block(h)
        return h[:, -1, :]


class SimpleGCN(nn.Module):
    """Two-layer GCN that mixes counties with a fixed adjacency."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, H: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # H: (B, N, D); adj: (B, N, N)
        z = torch.matmul(adj, H)
        z = F.relu(self.lin1(z))
        z = self.drop(z)
        z = torch.matmul(adj, z)
        return F.relu(self.lin2(z))


class WildfireMamba(nn.Module):
    """
    Mamba-based spatio-temporal wildfire model for county-day ERA5 features.

    Input shape: (batch, past_days, num_counties, num_features)
    Output: logits per county for the next day (use sigmoid for probabilities)
    """

    def __init__(
        self,
        in_dim: int,
        num_counties: int,
        past_days: int,
        hidden_dim: int = 128,
        gcn_hidden: int = 64,
        mamba_layers: int = 2,
        state_dim: int = 64,
        conv_kernel: int = 5,
        dropout: float = 0.1,
        adjacency: Optional[torch.Tensor] = None,
        with_count_head: bool = False,
    ):
        super().__init__()
        self.num_counties = num_counties
        self.past_days = past_days
        self.with_count_head = with_count_head
        self.temporal = MambaTemporalEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=mamba_layers,
            state_dim=state_dim,
            conv_kernel=conv_kernel,
            dropout=dropout,
        )
        # differential branch is shallower and gates how much change to inject
        self.delta_temporal = MambaTemporalEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=max(1, mamba_layers - 1),
            state_dim=state_dim,
            conv_kernel=conv_kernel,
            dropout=dropout,
        )
        self.delta_gate = nn.Linear(hidden_dim, hidden_dim)
        self.gcn = SimpleGCN(hidden_dim, hidden_dim=gcn_hidden, out_dim=gcn_hidden, dropout=dropout)
        self.cls_head = nn.Linear(gcn_hidden, 1)
        if self.with_count_head:
            self.count_head = nn.Linear(gcn_hidden, 1)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("_adjacency", None)
        if adjacency is not None:
            self.set_adjacency(adjacency)

    def set_adjacency(self, adj: torch.Tensor) -> None:
        """Set/override the spatial adjacency."""
        adj = _normalize_adjacency(adj.detach())
        self._adjacency = adj

    def _get_adjacency(self, batch_size: int) -> torch.Tensor:
        if self._adjacency is None:
            eye = torch.eye(self.num_counties, device=self.cls_head.weight.device)
            adj = _normalize_adjacency(eye)
        else:
            adj = self._adjacency
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        if adj.size(0) == 1 and batch_size > 1:
            adj = adj.expand(batch_size, -1, -1)
        return adj

    @staticmethod
    def _temporal_delta(x: torch.Tensor) -> torch.Tensor:
        # prepend zeros so delta has the same length as the input sequence
        zeros = torch.zeros(x.size(0), 1, x.size(2), device=x.device, dtype=x.dtype)
        return torch.cat([zeros, x[:, 1:] - x[:, :-1]], dim=1)

    def forward(self, x: torch.Tensor, adjacency: Optional[torch.Tensor] = None):
        """
        Args:
            x: Tensor shaped (batch, past_days, num_counties, in_dim)
            adjacency: Optional (N, N) or (B, N, N) adjacency override.
        Returns:
            - logits: (batch, num_counties)
            - optional counts: (batch, num_counties) if with_count_head is enabled.
        """
        B, T, N, F = x.shape
        if T != self.past_days:
            raise ValueError(f"Expected past_days={self.past_days}, got {T}.")
        if N != self.num_counties:
            raise ValueError(f"Expected num_counties={self.num_counties}, got {N}.")

        # flatten counties into the batch for temporal encoding
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, F)
        base = self.temporal(x_flat)
        delta = self.delta_temporal(self._temporal_delta(x_flat))
        gate = torch.sigmoid(self.delta_gate(delta))
        fused = base * gate + delta
        fused = fused.view(B, N, -1)

        adj = _normalize_adjacency(adjacency) if adjacency is not None else self._get_adjacency(B)
        spatial = self.gcn(fused, adj)
        spatial = self.dropout(spatial)
        logits = self.cls_head(spatial).squeeze(-1)
        if self.with_count_head:
            counts = F.relu(self.count_head(spatial)).squeeze(-1)
            return logits, counts
        return logits


def wildfire_mamba_builder(
    task: str,
    in_dim: int,
    num_counties: int,
    past_days: int,
    **kwargs,
) -> WildfireMamba:
    """
    Builder used by the model registry.
    """
    if task.lower() not in {"classification", "binary_classification"}:
        raise ValueError("WildfireMamba is designed for binary per-county classification.")
    return WildfireMamba(
        in_dim=in_dim,
        num_counties=num_counties,
        past_days=past_days,
        hidden_dim=kwargs.get("hidden_dim", 128),
        gcn_hidden=kwargs.get("gcn_hidden", 64),
        mamba_layers=kwargs.get("mamba_layers", 2),
        state_dim=kwargs.get("state_dim", 64),
        conv_kernel=kwargs.get("conv_kernel", 5),
        dropout=kwargs.get("dropout", 0.1),
        adjacency=kwargs.get("adjacency"),
        with_count_head=kwargs.get("with_count_head", False),
    )


__all__ = ["WildfireMamba", "wildfire_mamba_builder"]
