import torch
import xarray as xr
import numpy as np
from pathlib import Path

from pyhazards.datasets import DataBundle, DataSplit, FeatureSpec, LabelSpec
from pyhazards.datasets.graph import GraphTemporalDataset


def knn_adjacency(coords: torch.Tensor, k: int = 4):
    """
    Build symmetric k-NN adjacency from mesh coordinates.

    coords: (N, 2) tensor of (lon, lat)
    returns: (N, N) adjacency matrix
    """
    N = coords.shape[0]
    dist = torch.cdist(coords, coords)

    adj = torch.zeros(N, N)
    knn = dist.topk(k + 1, largest=False).indices

    for i in range(N):
        adj[i, knn[i, 1:]] = 1.0
        adj[knn[i, 1:], i] = 1.0

    return adj


def load_hydrograph_data(
    era5_path: str,
    max_nodes: int = 50,
):
    """
    Load ERA5 NetCDF files, use ERA5 grid as mesh,
    build kNN adjacency, and return a DataBundle.
    """

    files = sorted(Path(era5_path).glob("*.nc"))
    assert len(files) > 0, "No ERA5 NetCDF files found"


    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        chunks={},
    )


    lats = ds["latitude"].values
    lons = ds["longitude"].values

    if lats[0] > lats[-1]:
        ds = ds.sortby("latitude")
        lats = ds["latitude"].values

    # Build mesh coordinates from ERA5 grid
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    mesh_coords = torch.tensor(
        np.stack([lon_grid.ravel(), lat_grid.ravel()], axis=1),
        dtype=torch.float,
    )

    mesh_coords = mesh_coords[:max_nodes]

    # ERA5 variables
    precip = ds["tp"].values    # total precipitation
    temp = ds["t2m"].values     # 2m temperature

    if precip.ndim == 3:
        precip = precip.mean(axis=0)
        temp = temp.mean(axis=0)

    node_feats = []
    for lon, lat in mesh_coords.numpy():
        i = np.argmin((lats - lat) ** 2)
        j = np.argmin((lons - lon) ** 2)
        node_feats.append([precip[i, j], temp[i, j]])

    X = torch.tensor(node_feats, dtype=torch.float).unsqueeze(0).unsqueeze(0)

    Y = X[:, 0, :, 0:1]  # (1, num_nodes)

    adjacency = knn_adjacency(mesh_coords, k=4)

    dataset = GraphTemporalDataset(X, Y, adjacency=adjacency)

    return DataBundle(
        splits={
            "train": DataSplit(inputs=dataset, targets=None),
        },
        feature_spec=FeatureSpec(
            input_dim=2,
            description="ERA5 precipitation + temperature on ERA5-derived mesh",
        ),
        label_spec=LabelSpec(
            num_targets=1,
            task_type="regression",
        ),
    )
