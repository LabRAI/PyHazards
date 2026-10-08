"""CAMELS-US reader (Addor et al., HESS 2017; Newman et al., HESS 2015), as NeuralHydrology reads it.

The file readers are ported from NeuralHydrology, ``neuralhydrology/datasetzoo/camelsus.py`` at commit
ea94a40 (https://github.com/neuralhydrology/neuralhydrology, BSD-3-Clause, Copyright (c) 2021,
NeuralHydrology). They expect the original CAMELS-US directory layout under ``data_dir``::

    basin_mean_forcing/<forcing>/[<huc>/]<basin>_*_forcing_leap.txt   (daymet, maurer, nldas, maurer_extended, ...)
    usgs_streamflow/[<huc>/]<basin>_streamflow_qc.txt
    camels_attributes_v2.0/camels_*.txt                                 (';'-separated, column gauge_id)

Discharge is converted from cubic feet per second to mm/day with the catchment area in the forcing
file header (``QObs(mm/d)``); negative discharge (missing-value flags) becomes NaN. PyHazards does not
download CAMELS-US (CC BY 4.0, Zenodo record 15529996); point ``data_dir`` at a local copy. The extended
Maurer forcings used by Kratzert et al. (2019) are a separate download (HydroShare,
doi 10.4211/hs.17c896843cf940339c3c3496d0c1c077).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..base import DataBundle, Dataset
from .streamflow import build_streamflow_bundle, read_basin_list

# Kratzert et al. (HESS 2019), Sect. 2.6 and Appendix A: the five extended-Maurer forcings and the 27
# CAMELS attributes (frac_snow is "frac_snow_daily" in the paper's table), train 1999-10-01 to
# 2008-09-30, test 1989-10-01 to 1999-09-30, 270-day input sequences. The validation period is the
# one of NeuralHydrology's example configuration (the paper used none).
KRATZERT2019_DYNAMIC_INPUTS = ["prcp(mm/day)", "srad(W/m2)", "tmax(C)", "tmin(C)", "vp(Pa)"]
KRATZERT2019_STATIC_ATTRIBUTES = [
    "elev_mean", "slope_mean", "area_gages2", "frac_forest", "lai_max", "lai_diff", "gvf_max", "gvf_diff",
    "soil_depth_pelletier", "soil_depth_statsgo", "soil_porosity", "soil_conductivity", "max_water_content",
    "sand_frac", "silt_frac", "clay_frac", "carbonate_rocks_frac", "geol_permeability", "p_mean", "pet_mean",
    "aridity", "frac_snow", "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur", "p_seasonality",
]
# Column order of the 27 attributes in the official 2019 checkpoints (the attributes.db of every HydroShare
# run). PyHazards, like NeuralHydrology, feeds static attributes in alphabetical order; to apply a 2019
# checkpoint to real data, permute x_s with KRATZERT2019_CHECKPOINT_STATIC_ORDER.
KRATZERT2019_CHECKPOINT_STATIC_ORDER = [
    "carbonate_rocks_frac", "geol_permeability", "frac_forest", "lai_max", "lai_diff", "gvf_max", "gvf_diff",
    "p_mean", "pet_mean", "p_seasonality", "frac_snow", "aridity", "high_prec_freq", "high_prec_dur",
    "low_prec_freq", "low_prec_dur", "elev_mean", "slope_mean", "area_gages2", "soil_depth_pelletier",
    "soil_depth_statsgo", "soil_porosity", "soil_conductivity", "max_water_content", "sand_frac", "silt_frac",
    "clay_frac",
]
KRATZERT2019_PERIODS = {
    "train": ("1999-10-01", "2008-09-30"),
    "val": ("1980-10-01", "1989-09-30"),
    "test": ("1989-10-01", "1999-09-30"),
}
CAMELS_US_TARGET = "QObs(mm/d)"


def load_camels_us_attributes(data_dir: Union[str, Path], basins: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Basin-indexed table of all CAMELS-US attributes (``camels_attributes_v2.0/camels_*.txt``)."""
    attributes_path = Path(data_dir) / "camels_attributes_v2.0"
    if not attributes_path.exists():
        raise FileNotFoundError(f"Attribute folder not found at {attributes_path}")
    dfs = []
    for txt_file in sorted(attributes_path.glob("camels_*.txt")):
        df_temp = pd.read_csv(txt_file, sep=";", header=0, dtype={"gauge_id": str})
        dfs.append(df_temp.set_index("gauge_id"))
    if not dfs:
        raise FileNotFoundError(f"No camels_*.txt attribute files in {attributes_path}")
    df = pd.concat(dfs, axis=1)
    # convert huc column to double digit strings
    df["huc"] = df["huc_02"].apply(lambda x: str(x).zfill(2))
    df = df.drop("huc_02", axis=1)
    if basins:
        missing = [b for b in basins if b not in df.index]
        if missing:
            raise ValueError(f"Some basins are missing static attributes: {missing}")
        df = df.loc[list(basins)]
    return df


def load_camels_us_forcings(data_dir: Union[str, Path], basin: str, forcings: str) -> Tuple[pd.DataFrame, int]:
    """Forcing table of one basin (date-indexed) and the catchment area (m2) from the file header."""
    forcing_path = Path(data_dir) / "basin_mean_forcing" / forcings
    if not forcing_path.is_dir():
        raise OSError(f"{forcing_path} does not exist")
    file_paths = sorted(forcing_path.glob(f"**/{basin}_*_forcing_leap.txt"))
    if not file_paths:
        raise FileNotFoundError(f"No file for Basin {basin} at {forcing_path}")
    with open(file_paths[0], "r") as fp:
        # load area from header
        fp.readline()
        fp.readline()
        area = int(fp.readline())
        # load the dataframe from the rest of the stream
        df = pd.read_csv(fp, sep=r"\s+")
    df["date"] = pd.to_datetime(
        df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str), format="%Y/%m/%d"
    )
    return df.set_index("date"), area


def load_camels_us_discharge(data_dir: Union[str, Path], basin: str, area: int) -> pd.Series:
    """Daily discharge of one basin in mm/day (USGS cubic feet per second divided by the area)."""
    discharge_path = Path(data_dir) / "usgs_streamflow"
    file_paths = sorted(discharge_path.glob(f"**/{basin}_streamflow_qc.txt"))
    if not file_paths:
        raise FileNotFoundError(f"No file for Basin {basin} at {discharge_path}")
    col_names = ["basin", "Year", "Mnth", "Day", "QObs", "flag"]
    df = pd.read_csv(file_paths[0], sep=r"\s+", header=None, names=col_names)
    df["date"] = pd.to_datetime(
        df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str), format="%Y/%m/%d"
    )
    df = df.set_index("date")
    # normalize discharge from cubic feet per second to mm per day
    return 28316846.592 * df.QObs * 86400 / (area * 10**6)


def load_camels_us_basin(data_dir: Union[str, Path], basin: str, forcings: Sequence[str]) -> pd.DataFrame:
    """Forcings (suffixed ``_<forcing>`` when several products are used) and ``QObs(mm/d)`` of one basin."""
    dfs = []
    area = None
    for forcing in forcings:
        df, area = load_camels_us_forcings(data_dir, basin, forcing)
        if len(forcings) > 1:
            df = df.rename(columns={col: f"{col}_{forcing}" for col in df.columns})
        dfs.append(df)
    df = pd.concat(dfs, axis=1)
    df[CAMELS_US_TARGET] = load_camels_us_discharge(data_dir, basin, area)
    # replace invalid discharge values by NaNs
    for col in [c for c in df.columns if "qobs" in c.lower()]:
        df.loc[df[col] < 0, col] = np.nan
    return df


class CamelsUSStreamflowDataset(Dataset):
    """CAMELS-US daily streamflow in the Kratzert et al. (2019) setup, read from a local copy.

    ``basins`` is a list of 8-digit USGS ids or a basin file (one id per line). Defaults reproduce the
    paper: extended Maurer forcings, the 27 static attributes, 270-day windows, train 1999-10-01 to
    2008-09-30 and test 1989-10-01 to 1999-09-30.
    """

    name = "camels_us_streamflow"

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        basins: Any = None,
        forcings: Union[str, Sequence[str]] = "maurer_extended",
        dynamic_inputs: Sequence[str] = tuple(KRATZERT2019_DYNAMIC_INPUTS),
        static_attributes: Sequence[str] = tuple(KRATZERT2019_STATIC_ATTRIBUTES),
        target_variables: Sequence[str] = (CAMELS_US_TARGET,),
        periods: Optional[Dict[str, Sequence[str]]] = None,
        seq_length: int = 270,
        predict_last_n: int = 1,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(cache_dir=cache_dir)
        if data_dir is None:
            raise ValueError("camels_us_streamflow needs data_dir, the root of a local CAMELS-US copy.")
        self.data_dir = Path(data_dir)
        self.basins = read_basin_list(basins)
        self.forcings = [forcings] if isinstance(forcings, str) else list(forcings)
        self.dynamic_inputs = list(dynamic_inputs)
        self.static_attributes = list(static_attributes)
        self.target_variables = list(target_variables)
        self.periods = dict(periods or KRATZERT2019_PERIODS)
        self.seq_length = int(seq_length)
        self.predict_last_n = int(predict_last_n)

    def _load(self) -> DataBundle:
        frames = {basin: load_camels_us_basin(self.data_dir, basin, self.forcings) for basin in self.basins}
        attributes = (
            load_camels_us_attributes(self.data_dir, self.basins) if self.static_attributes else None
        )
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
            metadata={"forcings": self.forcings, "data_dir": str(self.data_dir)},
        )


__all__ = [
    "CAMELS_US_TARGET",
    "KRATZERT2019_CHECKPOINT_STATIC_ORDER",
    "CamelsUSStreamflowDataset",
    "KRATZERT2019_DYNAMIC_INPUTS",
    "KRATZERT2019_PERIODS",
    "KRATZERT2019_STATIC_ATTRIBUTES",
    "load_camels_us_attributes",
    "load_camels_us_basin",
    "load_camels_us_discharge",
    "load_camels_us_forcings",
]
