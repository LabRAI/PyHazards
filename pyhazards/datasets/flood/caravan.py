"""Caravan reader (Kratzert et al., Scientific Data 10:61, 2023), as NeuralHydrology reads it.

The file readers follow NeuralHydrology, ``neuralhydrology/datasetzoo/caravan.py`` at commit ea94a40
(https://github.com/neuralhydrology/neuralhydrology, BSD-3-Clause, Copyright (c) 2021, NeuralHydrology).
They expect the layout of the official Caravan release (Zenodo, CC BY 4.0; per-source licence notes in
its ``licenses/`` folder) under ``data_dir``::

    attributes/<source>/attributes_{caravan,hydroatlas,other}_<source>.csv   (column gauge_id)
    timeseries/netcdf/<source>/<source>_<id>.nc                               (dimension date)
    timeseries/csv/<source>/<source>_<id>.csv                                  (column date)

Basin ids have the form ``<source>_<id>`` (e.g. ``camelsgb_28015``). Time series are daily ERA5-Land
forcings and states (e.g. ``total_precipitation_sum``, ``temperature_2m_mean``,
``potential_evaporation_sum``) and observed ``streamflow`` in mm/day. PyHazards downloads nothing;
point ``data_dir`` at a local copy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd
import xarray

from ..base import DataBundle, Dataset
from .streamflow import build_streamflow_bundle, read_basin_list

CARAVAN_TARGET = "streamflow"


def load_caravan_attributes(
    data_dir: Union[str, Path],
    basins: Optional[Sequence[str]] = None,
    subdataset: Optional[str] = None,
) -> pd.DataFrame:
    """Basin-indexed table merging every attribute CSV of the requested source datasets."""
    data_dir = Path(data_dir)
    if subdataset:
        subdataset_dir = data_dir / "attributes" / subdataset
        if not subdataset_dir.is_dir():
            raise FileNotFoundError(f"No subdataset {subdataset} found at {subdataset_dir}.")
        subdataset_dirs = [subdataset_dir]
    else:
        subdataset_dirs = sorted(d for d in (data_dir / "attributes").glob("*") if d.is_dir())

    if basins:
        subdataset_names = sorted(set(x.split("_")[0] for x in basins))
        if subdataset:
            if len(subdataset_names) > 1 or subdataset_names[0] != subdataset:
                raise ValueError("At least one of the passed basins is not part of the passed subdataset.")
        else:
            missing_subdatasets = [s for s in subdataset_names if not (data_dir / "attributes" / s).is_dir()]
            if missing_subdatasets:
                raise FileNotFoundError(f"Could not find subdataset directories for {missing_subdatasets}.")
        subdataset_dirs = [s for s in subdataset_dirs if s.name in subdataset_names]

    dfs = []
    for subdataset_dir in subdataset_dirs:
        files = [pd.read_csv(csv_file, index_col="gauge_id") for csv_file in sorted(subdataset_dir.glob("*.csv"))]
        if files:
            dfs.append(pd.concat(files, axis=1))
    if not dfs:
        raise FileNotFoundError(f"No attribute CSV files under {data_dir / 'attributes'}.")
    df = pd.concat(dfs, axis=0)

    if basins:
        missing = [b for b in basins if b not in df.index]
        if missing:
            raise ValueError(f"Some basins are missing static attributes: {missing}")
        df = df.loc[list(basins)]
    return df


def load_caravan_timeseries(data_dir: Union[str, Path], basin: str, filetype: str = "netcdf") -> pd.DataFrame:
    """Date-indexed time series of one basin from ``timeseries/netcdf`` (default) or ``timeseries/csv``."""
    subdataset_name = basin.split("_")[0]
    if filetype == "netcdf":
        filepath = Path(data_dir) / "timeseries" / "netcdf" / subdataset_name / f"{basin}.nc"
    elif filetype == "csv":
        filepath = Path(data_dir) / "timeseries" / "csv" / subdataset_name / f"{basin}.csv"
    else:
        raise ValueError("filetype has to be either 'csv' or 'netcdf'.")
    if not filepath.is_file():
        raise FileNotFoundError(f"No basin file found at {filepath}.")
    if filetype == "netcdf":
        with xarray.open_dataset(filepath) as ds:
            df = ds.to_dataframe()
    else:
        df = pd.read_csv(filepath, parse_dates=["date"]).set_index("date")
    return df


class CaravanStreamflowDataset(Dataset):
    """Caravan daily streamflow, read from a local copy of the official release.

    Caravan has no single published input set or split, so ``basins``, ``dynamic_inputs`` and
    ``periods`` must be given; ``static_attributes`` may be any columns of the attribute CSVs.
    """

    name = "caravan_streamflow"

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        basins: Any = None,
        dynamic_inputs: Optional[Sequence[str]] = None,
        static_attributes: Sequence[str] = (),
        target_variables: Sequence[str] = (CARAVAN_TARGET,),
        periods: Optional[Dict[str, Sequence[str]]] = None,
        seq_length: int = 365,
        predict_last_n: int = 1,
        filetype: str = "netcdf",
        cache_dir: Optional[str] = None,
    ):
        super().__init__(cache_dir=cache_dir)
        if data_dir is None:
            raise ValueError("caravan_streamflow needs data_dir, the root of a local Caravan copy.")
        if not dynamic_inputs:
            raise ValueError("caravan_streamflow needs dynamic_inputs, e.g. ['total_precipitation_sum', ...].")
        if not periods:
            raise ValueError("caravan_streamflow needs periods, e.g. {'train': (start, end), 'test': (start, end)}.")
        if filetype not in ("netcdf", "csv"):
            raise ValueError("filetype has to be either 'csv' or 'netcdf'.")
        self.data_dir = Path(data_dir)
        self.basins: List[str] = read_basin_list(basins)
        self.dynamic_inputs = list(dynamic_inputs)
        self.static_attributes = list(static_attributes)
        self.target_variables = list(target_variables)
        self.periods = dict(periods)
        self.seq_length = int(seq_length)
        self.predict_last_n = int(predict_last_n)
        self.filetype = filetype

    def _load(self) -> DataBundle:
        frames = {basin: load_caravan_timeseries(self.data_dir, basin, self.filetype) for basin in self.basins}
        attributes = load_caravan_attributes(self.data_dir, self.basins) if self.static_attributes else None
        return build_streamflow_bundle(
            frames,
            attributes,
            self.dynamic_inputs,
            self.static_attributes,
            self.target_variables,
            self.periods,
            self.seq_length,
            self.predict_last_n,
            dataset_name=self.name,
            metadata={"data_dir": str(self.data_dir), "filetype": self.filetype},
        )


__all__ = [
    "CARAVAN_TARGET",
    "CaravanStreamflowDataset",
    "load_caravan_attributes",
    "load_caravan_timeseries",
]
