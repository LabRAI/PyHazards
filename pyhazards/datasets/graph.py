from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class GraphTemporalDataset(Dataset):
    """
    Simple container for county/day style tensors with an optional adjacency.

    Each sample is a window of shape (past_days, num_counties, num_features) and a label
    of shape (num_counties,).
    """

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: Tensor (samples, past_days, num_counties, num_features)
            y: Tensor (samples, num_counties) or (samples, num_counties, targets)
            adjacency: Optional Tensor
                - (num_counties, num_counties) global adjacency
                - (samples, num_counties, num_counties) per-sample adjacency
        """
        if x.ndim != 4:
            raise ValueError("x must be (samples, past_days, num_counties, num_features)")
        if y.ndim not in (2, 3):
            raise ValueError("y must be (samples, num_counties) or (samples, num_counties, targets)")
        if adjacency is not None and adjacency.ndim not in (2, 3):
            raise ValueError("adjacency must be None, (N,N), or (B,N,N)")
        if adjacency is not None and adjacency.ndim == 2 and adjacency.size(0) != x.size(2):
            raise ValueError("adjacency size mismatch with num_counties")
        if adjacency is not None and adjacency.ndim == 3 and adjacency.size(1) != x.size(2):
            raise ValueError("adjacency size mismatch with num_counties")

        self.x = x
        self.y = y
        self.adj = adjacency

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], torch.Tensor]:
        adj = None
        if self.adj is not None:
            adj = self.adj if self.adj.ndim == 2 else self.adj[idx]
        return {"x": self.x[idx], "adj": adj}, self.y[idx]


def graph_collate(batch: List[Tuple[Dict[str, Any], torch.Tensor]]):
    """
    Collate function that stacks x and adjacency if provided.
    """
    xs, ys = zip(*batch)
    x_tensor = torch.stack([item["x"] for item in xs], dim=0)
    adj_list = [item["adj"] for item in xs]
    adj = None
    if any(a is not None for a in adj_list):
        # If some entries are None, replace with first non-None
        first = next(a for a in adj_list if a is not None)
        adj = torch.stack([a if a is not None else first for a in adj_list], dim=0)
    y_tensor = torch.stack(ys, dim=0)
    return {"x": x_tensor, "adj": adj}, y_tensor


__all__ = ["GraphTemporalDataset", "graph_collate"]
