import torch
import torch.nn as nn 
import torch.nn.functional as F
import math

#inital code

class HydroGraphNet(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        out_dim: int,
        hidden_dim: int = 64,
        harmonics: int = 5,
        num_gn_blocks: int = 5,
    ):
        super().__init__()

        #-------encoder-------
        self.node_encoder = KAN(in_dim = node_in_dim, hidden_dim = hidden_dim, harmonics = harmonics)
        self.edge_encoder = MLP(in_dim= edge_in_dim, out_dim = hidden_dim , hidden_dim= hidden_dim)
        #------processor-------
        self.processor = nn.ModuleList(
            [
                GN(hidden_dim = hidden_dim)
                for i in range(num_gn_blocks)
            ]
        )
        #-------decoder-------
        self.decoder = MLP(in_dim = hidden_dim, out_dim = out_dim)
    def forward(self, batch):
        # batch is a dict from graph_collate
        x = batch["x"]        # (B, past_days, N, F)
        adj = batch["adj"]    # (B, N, N) or None

        node_x = x[:, -1]     # (B, N, F)

        if adj is None:
            raise ValueError("HydroGraphNet requires adjacency")

        A = adj[0]            # (N, N)
        senders, receivers = A.nonzero(as_tuple=True)

        # ---- encoder ----
        node = self.node_encoder(node_x)

        # edge features from geometry: [Δx, Δy, distance]
        coords = batch.get("coords")
        if coords is None:
            coords = torch.zeros(node.size(1), 2, device=node.device)
        else:
            coords = coords.to(node.device)

        src = coords[senders]      # (E, 2)
        dst = coords[receivers]    # (E, 2)

        delta = src - dst          # (E, 2)
        dist = torch.norm(delta, dim=-1, keepdim=True)  # (E, 1)

        edge_feat = torch.cat([delta, dist], dim=-1)    # (E, 3)

        edge_x = edge_feat.unsqueeze(0).repeat(node.size(0), 1, 1)


        edge = self.edge_encoder(edge_x)

        # ---- GN processor ----
        for gn in self.processor:
            node, edge = gn(node, edge, senders, receivers)

        # ---- decoder ----
        # autoregressive update 
        delta = self.decoder(node)          # Δy
        y_prev = node_x[..., :delta.size(-1)]  # y_t (assumes target is part of node features)
        y_next = y_prev + delta             # y_{t+1}

        return y_next



class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64): 
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class KAN(nn.Module):
    def __init__(
        self,
        in_dim: int, 
        harmonics: int = 5,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.harmonics = harmonics

        self.feature_proj = nn.ModuleList(
            [
                nn.Linear(2*harmonics + 1, hidden_dim)
                for i in range(in_dim)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []

        for i in range(self.in_dim):
            xi = x[:, : ,i].unsqueeze(-1)

           
            basis = [torch.ones_like(xi)]
            for k in range(1, self.harmonics + 1):
                basis.append(torch.sin(k * xi))
                basis.append(torch.cos(k * xi))

            basis = torch.cat(basis, dim=-1)

          
            outputs.append(self.feature_proj[i](basis))
      
        return torch.stack(outputs, dim=0).sum(dim=0)
 
class GN(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.edge_mlp = MLP(3 * hidden_dim, hidden_dim, hidden_dim)
        self.node_mlp = MLP(2 * hidden_dim, hidden_dim, hidden_dim)

    def forward(self, node, edge, senders, receivers):
        sender_feat = node[:, senders, :]
        receiver_feat = node[:, receivers, :]

        edge_input = torch.cat([edge, sender_feat, receiver_feat], dim=-1)
        edge = edge + self.edge_mlp(edge_input)

        agg = torch.zeros_like(node)
        agg.index_add_(1, receivers, edge)

        node_input = torch.cat([node, agg], dim=-1)
        node = node + self.node_mlp(node_input)

        return node, edge

def hydrographnet_builder(
    task: str,
    node_in_dim: int,
    edge_in_dim: int,
    out_dim: int,
    **kwargs,
):
    if task != "regression":
        raise ValueError("HydroGraphNet only supports regression")

    return HydroGraphNet(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        out_dim=out_dim,
        hidden_dim=kwargs.get("hidden_dim", 64),
        harmonics=kwargs.get("harmonics", 5),
        num_gn_blocks=kwargs.get("num_gn_blocks", 5),
    )
