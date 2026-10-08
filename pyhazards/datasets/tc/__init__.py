"""Tropical cyclone datasets: real readers (IBTrACS, Xu et al. 2021 SHIPS predictors, TCND, Hurricast IBTrACS + ERA5) and synthetic smoke data."""

from __future__ import annotations

from .ibtracs import (
    IBTrACSTropicalCycloneDataset,
    build_track_samples,
    download_ibtracs,
    ibtracs_url,
    read_ibtracs,
    read_ibtracs_csv,
    read_ibtracs_netcdf,
)
from .hurricast import HurricastDataset, era5_maps, hurricast_storms, hurricast_windows, open_era5
from .ships import SHIPSXu2021Dataset, read_xu2021_table
from .tcnd import TropiCycloneNetDataset, gph_to_image, read_tcnd_track
from .synthetic import (
    SyntheticHurricastDataset,
    SyntheticSAFNetDataset,
    SyntheticSHIPSDataset,
    SyntheticTCIFFusionDataset,
    SyntheticTCNDDataset,
    SyntheticTropicalCycloneDataset,
    synthetic_tcnd_batch,
)

__all__ = [
    "HurricastDataset",
    "IBTrACSTropicalCycloneDataset",
    "SHIPSXu2021Dataset",
    "SyntheticHurricastDataset",
    "SyntheticSAFNetDataset",
    "SyntheticSHIPSDataset",
    "SyntheticTCIFFusionDataset",
    "SyntheticTCNDDataset",
    "SyntheticTropicalCycloneDataset",
    "TropiCycloneNetDataset",
    "build_track_samples",
    "download_ibtracs",
    "era5_maps",
    "hurricast_storms",
    "hurricast_windows",
    "ibtracs_url",
    "open_era5",
    "read_ibtracs",
    "read_ibtracs_csv",
    "read_ibtracs_netcdf",
    "gph_to_image",
    "read_tcnd_track",
    "read_xu2021_table",
    "synthetic_tcnd_batch",
]
