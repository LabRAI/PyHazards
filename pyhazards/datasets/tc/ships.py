"""SHIPS predictor table of Xu et al. (2021) for 24-hour intensity-change forecasting.

Data: "Supporting data for Xu et al. 2021 - Weather and Forecasting" (Zenodo
doi:10.5281/zenodo.4784610, CC BY 4.0), built from the SHIPS developmental data (NHC / CIRA). The
file read here, ``train_global_fill_REA_na_wo_img_scaled.csv`` (392 MB), holds one row per storm
and 6-hourly time with the standardised SHIPS predictors, the identifiers ``name``, ``year``,
``basin`` and ``type`` (``rea``: predictors from reanalysis, global 1982-2017; ``opr``:
operational predictors, Atlantic 2010-2018) and the target ``dvs24``, the change of the maximum
sustained wind over the next 24 hours in knots. PyHazards reads the file the user downloaded; it
never redistributes it.

The split reproduces ``utils.load_loyo_data`` of the official code (wenweixu/tropicalcyclone_MLP,
BSD-2-Clause) with the options of ``loyo_testing.py`` for the MLP (``remove_oprfortraining=True``):
for a left-out year ``Y``, training rows are all ``rea`` rows except Atlantic (``AL``) rows of
``Y``; test rows are the ``opr`` rows of ``Y``. Validation is a random 10 % of the training rows
(``ShuffleSplit(test_size=0.1)`` in the official script, seeded here). The official script reads a
``..._w2020.csv`` variant of the file that is not in the Zenodo record.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch

from ...models.tropicalcyclone_mlp import SHIPS_PREDICTORS
from ..base import DataBundle, DataSplit, Dataset, FeatureSpec, LabelSpec

XU2021_ZENODO_RECORD = "https://doi.org/10.5281/zenodo.4784610"
XU2021_TRAIN_FILE = "train_global_fill_REA_na_wo_img_scaled.csv"
_ID_COLUMNS = ("name", "year", "basin", "type", "dvs24")


def read_xu2021_table(path: Union[str, Path], predictors: Sequence[str] = SHIPS_PREDICTORS):
    """Read the identifier, target and predictor columns of the Xu et al. (2021) CSV."""
    import pandas as pd

    path = Path(path)
    header = pd.read_csv(path, nrows=0).columns
    wanted = list(_ID_COLUMNS) + list(predictors)
    missing = [column for column in wanted if column not in header]
    if missing:
        raise ValueError(f"{path} lacks the Xu et al. (2021) column(s) {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    frame = pd.read_csv(path, usecols=wanted)
    frame["year"] = frame["year"].astype(int)
    return frame


class SHIPSXu2021Dataset(Dataset):
    """Leave-one-year-out fold of the Xu et al. (2021) SHIPS predictor table.

    Inputs ``(samples, 121)`` are the standardised predictors of
    :data:`pyhazards.models.tropicalcyclone_mlp.SHIPS_PREDICTORS`; targets ``(samples,)`` are
    ``dvs24`` in knots. ``leave_out_year`` is the test year (2010-2018 in the paper).
    """

    name = "ships_xu2021"

    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        cache_dir: Optional[str] = None,
        leave_out_year: int = 2018,
        val_fraction: float = 0.1,
        seed: int = 0,
        drop_missing: bool = True,
    ):
        super().__init__(cache_dir=cache_dir)
        if path is None:
            if cache_dir is None:
                raise ValueError(f"ships_xu2021 needs path= to {XU2021_TRAIN_FILE} (download it from {XU2021_ZENODO_RECORD})")
            path = Path(cache_dir) / XU2021_TRAIN_FILE
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} not found; download {XU2021_TRAIN_FILE} from {XU2021_ZENODO_RECORD}")
        if not 0.0 <= float(val_fraction) < 1.0:
            raise ValueError("val_fraction must be in [0, 1)")
        self.leave_out_year = int(leave_out_year)
        self.val_fraction = float(val_fraction)
        self.seed = int(seed)
        self.drop_missing = drop_missing

    def _load(self) -> DataBundle:
        frame = read_xu2021_table(self.path)
        if self.drop_missing:
            frame = frame.dropna(subset=["dvs24", *SHIPS_PREDICTORS])
        year = self.leave_out_year
        train = frame[~((frame["basin"] == "AL") & (frame["year"] == year)) & (frame["type"] != "opr")]
        test = frame[(frame["year"] == year) & (frame["type"] == "opr")]
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(train))
        n_val = int(np.ceil(self.val_fraction * len(train))) if self.val_fraction > 0 else 0
        val_index, train_index = order[:n_val], order[n_val:]

        def split(part, index=None):
            rows = part if index is None else part.iloc[np.sort(index)]
            return DataSplit(
                torch.as_tensor(rows[list(SHIPS_PREDICTORS)].to_numpy(dtype=np.float32)),
                torch.as_tensor(rows["dvs24"].to_numpy(dtype=np.float32)),
                metadata={"groups": rows["year"].tolist(), "name": rows["name"].tolist(), "basin": rows["basin"].tolist()},
            )

        return DataBundle(
            splits={"train": split(train, train_index), "val": split(train, val_index), "test": split(test)},
            feature_spec=FeatureSpec(
                input_dim=len(SHIPS_PREDICTORS),
                description="121 standardised SHIPS predictors (Xu et al. 2021).",
                extra={"predictors": list(SHIPS_PREDICTORS)},
            ),
            label_spec=LabelSpec(num_targets=1, task_type="regression", description="24-hour change of the maximum sustained wind (kt)."),
            metadata={
                "dataset": self.name,
                "source_dataset": "Xu et al. (2021) SHIPS predictors, Zenodo 4784610",
                "hazard_task": "tc.intensity",
                "lead_hours": [24],
                "units": {"wind": "kt"},
                "intensity_target": "dvs24 (24-hour intensity change)",
                "leave_out_year": year,
                "source_file": str(self.path),
                "synthetic": False,
            },
        )


__all__ = ["SHIPSXu2021Dataset", "XU2021_TRAIN_FILE", "read_xu2021_table"]
