import torch
import torch.nn as nn 
import torch.nn.functional as F
import math

#inital code

class hydrographnet(nn.Module):
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


    def forward(self, node_x: torch.Tensor, edge_x: torch.Tensor) -> torch.Tensor:
        #--encoder--
        node = self.node_encoder(node_x)
        edge = self.edge_encoder(edge_x)
        #--processor--
        for gn in self.processor:
            node, edge = gn(node, edge)
        #--decoder--
        out = self.decoder(node) 
        return out


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
        return self.net(x)


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
        batch_size = x.shape[0]
        outputs = []

        for i in range(self.in_dim):
            xi = x[:, i].unsqueeze(1)

           
            basis = [torch.ones_like(xi)]
            for k in range(1, self.harmonics + 1):
                basis.append(torch.sin(k * xi))
                basis.append(torch.cos(k * xi))

            basis = torch.cat(basis, dim=1)

          
            outputs.append(self.feature_proj[i](basis))
      
        return torch.stack(outputs, dim=0).sum(dim=0)
    
class GN(nn.Module):
    def __init__(
        self,
        #parameters,
        #parameters,
        #parameters,
        #parameters,
        #parameters,
        #parameters,
    ):
        super().__init__()
        #setting all of the values to defaults and parameters


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #forwarding process
        return 1