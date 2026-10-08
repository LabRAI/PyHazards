"""Reader for the TropiCycloneNet Dataset (TCND) in the layout used to train and test TropiCycloneNet.

Dataset: Huang et al., "Benchmark dataset and deep learning method for global tropical cyclone
forecasting", Nature Communications 16, 5923 (2025). Releases (CC BY 4.0): the dataset itself
(Zenodo doi:10.5281/zenodo.15009527: ``TCND_Data1D``, ``TCND_Env-Data``, ``TCND_Data3D_<basin>``)
and the preprocessed training data of the model (Zenodo record 17104690: ``data1d.zip``,
``env_data.zip``, ``data_3d.zip`` with 500 hPa geopotential crops). PyHazards reads the files the
user downloaded and extracted; it never redistributes them.

Expected directories (the official test loader's names under one ``root``; each can also be
passed separately):

* ``BST_data/<area>/<split>/<AREA><YEAR>BST<NAME>.txt`` (Data1d): tab-separated rows
  ``frame, id, lon, lat, pres, wind, YYYYMMDDHH, name`` with the normalised values of
  :data:`pyhazards.models.tropicyclonenet.TCND_NORMALIZATION`;
* ``Env_data/<area>/<year>/<NAME>/<YYYYMMDDHH>.npy``: pickled dicts of the Env-Data features;
* ``ERA5_gph500/<area>/<year>/<NAME>/<YYYYMMDDHH>.npy``: 500 hPa geopotential crops (100 x 100).

Samples follow the official ``TrajectoryDataset`` (TCNM/data/trajectoriesWithMe_unet.py) and
``seq_collate``: windows of ``obs_len + pred_len`` consecutive records of a file (stride ``skip``);
``obs_traj`` / ``obs_traj_rel`` hold the observed normalised values and their steps (0 at the first
step); GPH crops are resized to 64 x 64 (bilinear, as ``cv2.resize``), scaled with the range
(44490.578125, 58768.4486860389) and clipped to [0, 1]; Env-Data values of ``-1`` (feature not
available yet, e.g. the 24-hour history at a track's start) are replaced by the first available
value of the same feature in the window. Targets are the future positions (degrees), central
pressure (hPa) and maximum sustained wind (m/s) at 6, 12, 18 and 24 h.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ...models.tropicyclonenet import ENV_FEATURES, TCND_NORMALIZATION
from ..base import DataBundle, DataSplit, Dataset, FeatureSpec, LabelSpec

TCND_AREAS = ("EP", "NA", "NI", "SI", "SP", "WP")
GPH_RANGE = (44490.578125, 58768.4486860389)
IMAGE_SIZE = 64


def read_tcnd_track(path: Union[str, Path]) -> Tuple[np.ndarray, List[str], List[str]]:
    """Rows of a Data1d file: values ``(n, 6)`` (frame, id, lon, lat, pres, wind), dates, names."""
    values, dates, names = [], [], []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.strip().split("\t")
            if len(fields) < 8:
                if line.strip():
                    raise ValueError(f"{path}: expected 8 tab-separated fields, got {len(fields)}: {line!r}")
                continue
            values.append([float(field) for field in fields[:-2]])
            dates.append(fields[-2])
            names.append(fields[-1])
    return np.asarray(values, dtype=np.float64).reshape(-1, 6), dates, names


def gph_to_image(gph: np.ndarray) -> np.ndarray:
    """Resize a GPH crop to 64 x 64 (bilinear, half-pixel centres) and scale it to [0, 1]."""
    tensor = torch.as_tensor(np.asarray(gph, dtype=np.float64))[None, None]
    resized = F.interpolate(tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)[0, 0].numpy()
    low, high = GPH_RANGE
    return np.clip((resized - low) / (high - low), 0.0, 1.0)


def _storm_dir(base: Path, area: str, year: str, name: str) -> Path:
    for candidate in (name, name.lower().capitalize(), name.upper()):
        path = base / area / year / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"no directory for storm {name} ({area} {year}) under {base}")


def _missing(value) -> bool:
    return type(value) is int and value == -1


def _env_window(env_dir: Path, area: str, year: str, name: str, dates: Sequence[str]) -> Dict[str, np.ndarray]:
    storm = _storm_dir(env_dir, area, year, name)
    records = [np.load(storm / f"{date}.npy", allow_pickle=True).item() for date in dates]
    window: Dict[str, np.ndarray] = {}
    for key, width in ENV_FEATURES:
        values = [record[key] for record in records]
        # The official loader tests ``value is -1``: only Python int -1 marks a missing feature.
        available = [value for value in values if not _missing(value)]
        if not available:
            raise ValueError(f"Env-Data feature {key!r} is -1 at every observed step of {name} {dates[0]}")
        filled = [available[0] if _missing(value) else value for value in values]
        window[key] = np.asarray(filled, dtype=np.float32).reshape(len(dates), width)
    return window


class TropiCycloneNetDataset(Dataset):
    """TCND samples in the input layout of :class:`pyhazards.models.TropiCycloneNet`.

    ``inputs`` of each split is a dict with ``obs_traj`` and ``obs_traj_rel`` ``(obs_len, N, 4)``,
    ``image_obs`` ``(N, 1, obs_len, 64, 64)`` and ``env_data`` (nine arrays ``(N, obs_len, width)``);
    ``targets`` is ``(N, pred_len, 4)`` with latitude, longitude, pressure and wind. The whole
    split is loaded into memory (about 0.13 MB per sample); use ``areas``, ``splits`` and
    ``max_samples`` to limit it.
    """

    name = "tropicyclonenet_dataset"

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        data1d_dir: Optional[Union[str, Path]] = None,
        env_dir: Optional[Union[str, Path]] = None,
        gph_dir: Optional[Union[str, Path]] = None,
        cache_dir: Optional[str] = None,
        areas: Sequence[str] = TCND_AREAS,
        splits: Sequence[str] = ("train", "val", "test"),
        obs_len: int = 8,
        pred_len: int = 4,
        skip: int = 1,
        max_samples: Optional[int] = None,
    ):
        super().__init__(cache_dir=cache_dir)
        base = Path(root) if root is not None else None
        resolved = {}
        for label, given, default in (("data1d_dir", data1d_dir, "BST_data"), ("env_dir", env_dir, "Env_data"), ("gph_dir", gph_dir, "ERA5_gph500")):
            if given is None and base is None:
                raise ValueError("tropicyclonenet_dataset needs root= (with BST_data, Env_data, ERA5_gph500) or the three directories")
            resolved[label] = Path(given) if given is not None else base / default
            if not resolved[label].is_dir():
                raise FileNotFoundError(f"{label} {resolved[label]} does not exist")
        self.data1d_dir, self.env_dir, self.gph_dir = resolved["data1d_dir"], resolved["env_dir"], resolved["gph_dir"]
        unknown = sorted(set(areas) - set(TCND_AREAS))
        if unknown:
            raise ValueError(f"unknown TCND areas {unknown}; choose from {TCND_AREAS}")
        self.areas = tuple(areas)
        self.splits = tuple(splits)
        self.obs_len, self.pred_len, self.skip = int(obs_len), int(pred_len), int(skip)
        self.max_samples = max_samples

    def _windows(self, split: str):
        seq_len = self.obs_len + self.pred_len
        for area in self.areas:
            folder = self.data1d_dir / area / split
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob("*.txt")):
                values, dates, names = read_tcnd_track(path)
                if len(values) > 1 and not np.all(np.diff(values[:, 0]) > 0):
                    raise ValueError(f"{path}: frame numbers (first column) must increase")
                frames = np.unique(values[:, 0]).tolist()
                count = int(math.ceil((len(frames) - seq_len + 1) / self.skip))
                for start in range(0, count * self.skip + 1, self.skip):
                    rows = values[start : start + seq_len]
                    if len(rows) != seq_len or len(np.unique(rows[:, 0])) != seq_len:
                        continue
                    yield path.stem, area, rows, dates[start : start + seq_len], names[start + self.obs_len - 1]

    def _load_split(self, split: str):
        obs, obs_rel, targets, images = [], [], [], []
        env: Dict[str, List[np.ndarray]] = {key: [] for key, _ in ENV_FEATURES}
        meta: Dict[str, List[str]] = {"storm": [], "area": [], "origin": []}
        for stem, area, rows, dates, name in self._windows(split):
            if self.max_samples is not None and len(obs) >= int(self.max_samples):
                break
            year = stem[2:6]
            track = np.round(rows[:, 2:6], 4)  # (seq_len, 4): lon, lat, pres, wind (normalised)
            rel = np.zeros_like(track)
            rel[1:] = track[1:] - track[:-1]
            gph_storm = _storm_dir(self.gph_dir, area, year, name)
            frames = [gph_to_image(np.load(gph_storm / f"{date}.npy")) for date in dates[: self.obs_len]]
            for key, value in _env_window(self.env_dir, area, year, name, dates[: self.obs_len]).items():
                env[key].append(value)
            obs.append(track[: self.obs_len])
            obs_rel.append(rel[: self.obs_len])
            future = track[self.obs_len :]
            physical = {var: future[:, i] * TCND_NORMALIZATION[var][0] + TCND_NORMALIZATION[var][1] for i, var in enumerate(("lon", "lat", "pres", "wind"))}
            targets.append(np.stack([physical["lat"], physical["lon"], physical["pres"], physical["wind"]], axis=-1))
            images.append(np.stack(frames))
            meta["storm"].append(stem)
            meta["area"].append(area)
            meta["origin"].append(dates[self.obs_len - 1])
        n = len(obs)
        inputs = {
            "obs_traj": torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(n, self.obs_len, 4)).permute(1, 0, 2).contiguous(),
            "obs_traj_rel": torch.as_tensor(np.asarray(obs_rel, dtype=np.float32).reshape(n, self.obs_len, 4)).permute(1, 0, 2).contiguous(),
            "image_obs": torch.as_tensor(np.asarray(images, dtype=np.float32).reshape(n, self.obs_len, IMAGE_SIZE, IMAGE_SIZE)).unsqueeze(1),
            "env_data": {key: torch.as_tensor(np.asarray(values, dtype=np.float32).reshape(n, self.obs_len, width)) for (key, width), values in zip(ENV_FEATURES, env.values())},
        }
        return DataSplit(inputs, torch.as_tensor(np.asarray(targets, dtype=np.float32).reshape(n, self.pred_len, 4)), metadata=meta)

    def _load(self) -> DataBundle:
        splits = {split: self._load_split(split) for split in self.splits}
        return DataBundle(
            splits=splits,
            feature_spec=FeatureSpec(
                description="TCND Data1d tracks, Env-Data features and 64x64 500 hPa GPH frames (TropiCycloneNet layout).",
                extra={"obs_len": self.obs_len, "pred_len": self.pred_len},
            ),
            label_spec=LabelSpec(num_targets=4, task_type="regression", description="Latitude, longitude, pressure (hPa) and wind (m/s) at 6-24 h."),
            metadata={
                "dataset": self.name,
                "source_dataset": "TropiCycloneNet Dataset (TCND)",
                "hazard_task": "tc.track_intensity",
                "lead_hours": [6 * (step + 1) for step in range(self.pred_len)],
                "target_variables": ["lat", "lon", "pres", "wind"],
                "units": {"wind": "m/s", "pres": "hPa"},
                "batch_dims": {"obs_traj": 1, "obs_traj_rel": 1},
                "synthetic": False,
            },
        )


__all__ = ["GPH_RANGE", "TCND_AREAS", "TropiCycloneNetDataset", "gph_to_image", "read_tcnd_track"]
