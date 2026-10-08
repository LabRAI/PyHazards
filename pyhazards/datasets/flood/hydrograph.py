"""HydroGraphNet mesh flood data: the White River (Indiana) HEC-RAS dataset and a synthetic stand-in.

The reading, normalisation, graph construction and sample layout follow PhysicsNeMo's
``HydroGraphDataset`` (``physicsnemo/datapipes/gnn/hydrographnet_dataset.py`` @ eb7a329, Apache-2.0,
Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES) used by the HydroGraphNet example; the
oracle test compares every sample with it.

File layout (Zenodo record 14969507, ``HydroGraphNet.zip``, CC BY 4.0; tab-separated text, prefix ``M80``):

- static, one value per mesh cell: ``M80_XY.txt`` (x, y), ``M80_CA.txt`` (cell area), ``M80_CE.txt``
  (elevation), ``M80_CS.txt`` (slope), ``M80_A.txt`` (aspect), ``M80_CU.txt`` (curvature), ``M80_N.txt``
  (Manning's n), ``M80_FA.txt`` (flow accumulation), ``M80_IP.txt`` (infiltration); rows beyond the
  number of XY points are ignored;
- per hydrograph ``<id>`` (``H1`` ... ``H500``): ``M80_WD_<id>.txt`` / ``M80_V_<id>.txt`` (water depth /
  volume, one row per 20-minute step, one column per cell), ``M80_US_InF_<id>.txt`` (upstream inflow in
  column 1), ``M80_Pr_<id>.txt`` (precipitation);
- ``train.txt``: hydrograph ids, one per line.

Each hydrograph drops its first 72 steps and is cut 25 steps after the inflow peak; precipitation is
multiplied by 2.7778e-7. Static features are standardised per column, the dynamic series with one mean
and standard deviation each over the training hydrographs (``std + 1e-8`` in the denominator). The
graph links every cell to its 4 nearest neighbours (k-d tree on standardised coordinates, edges point
from the cell to its neighbours); edge features are the standardised offsets and distance.

Training samples are sliding windows: 16 node features (x, y, area, elevation, slope, aspect,
curvature, Manning, flow accumulation, infiltration, inflow and precipitation at the window start,
depth and volume over the ``n_time_steps=2`` window) and the next change of depth and volume as target,
with the physics data of the continuity loss. Test samples are whole hydrographs for autoregressive
rollouts (the first window, the forcings and the true depth / volume of the next ``rollout_length``
steps).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from ..base import DataBundle, DataSplit, Dataset, FeatureSpec, LabelSpec

EPSILON = 1e-8
HYDROGRAPH_PREFIX = "M80"
# PyHazards key -> HydroGraphDataset static key and file code.
STATIC_FILES = {
    "xy_coords": "XY",
    "area": "CA",
    "elevation": "CE",
    "slope": "CS",
    "aspect": "A",
    "curvature": "CU",
    "manning": "N",
    "flow_accum": "FA",
    "infiltration": "IP",
}
STATIC_ORDER = tuple(STATIC_FILES)
DYNAMIC_KEYS = ("water_depth", "volume", "precipitation", "inflow_hydrograph")
STATIC_NORM_STATS_FILE = "static_norm_stats.json"
DYNAMIC_NORM_STATS_FILE = "dynamic_norm_stats.json"


def _normalize(data: np.ndarray, mean, std) -> np.ndarray:
    mean = np.array(mean) if isinstance(mean, list) else mean
    std = np.array(std) if isinstance(std, list) else std
    return (data - mean) / (std + EPSILON)


def _denormalize(data: np.ndarray, mean, std) -> np.ndarray:
    mean = np.array(mean) if isinstance(mean, list) else mean
    std = np.array(std) if isinstance(std, list) else std
    return data * (std + EPSILON) + mean


def read_static(data_dir: Union[str, Path], prefix: str = HYDROGRAPH_PREFIX) -> Dict[str, np.ndarray]:
    """Raw static mesh arrays: ``xy_coords`` (N, 2) and one (N, 1) column per other attribute."""
    root = Path(data_dir)
    path = root / f"{prefix}_XY.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; data_dir must hold the HydroGraphNet files ({prefix}_XY.txt, ...).")
    xy = np.loadtxt(path, delimiter="\t")
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"{path} must have two tab-separated columns (x, y); got shape {xy.shape}.")
    raw: Dict[str, np.ndarray] = {"xy_coords": xy}
    for key, code in list(STATIC_FILES.items())[1:]:
        raw[key] = np.loadtxt(root / f"{prefix}_{code}.txt", delimiter="\t")[: xy.shape[0]].reshape(-1, 1)
    return raw


def standardize_static(
    raw: Mapping[str, np.ndarray], stats: Optional[Mapping[str, Mapping[str, Any]]] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, list]]]:
    """Standardise each static array with ``stats`` (or its own column means / standard deviations)."""
    stats = {key: dict(value) for key, value in (stats or {}).items()}
    out: Dict[str, np.ndarray] = {}
    for key in STATIC_ORDER:
        data = np.asarray(raw[key], dtype=np.float64)
        if key in stats:
            mean_val = np.array(stats[key]["mean"])
            std_val = np.array(stats[key]["std"])
        else:
            mean_val = np.mean(data, axis=0)
            std_val = np.std(data, axis=0)
            stats[key] = {"mean": mean_val.tolist(), "std": std_val.tolist()}
        out[key] = (data - mean_val) / (std_val + EPSILON)
    return out, stats


def read_hydrograph(
    data_dir: Union[str, Path],
    hydrograph_id: str,
    num_points: int,
    prefix: str = HYDROGRAPH_PREFIX,
    interval: int = 1,
    skip: int = 72,
) -> Dict[str, np.ndarray]:
    """Raw series of one hydrograph: ``water_depth`` / ``volume`` (T, N), ``inflow_hydrograph`` / ``precipitation`` (T,)."""
    root = Path(data_dir)
    water_depth = np.loadtxt(root / f"{prefix}_WD_{hydrograph_id}.txt", delimiter="\t")[skip::interval, :num_points]
    inflow = np.loadtxt(root / f"{prefix}_US_InF_{hydrograph_id}.txt", delimiter="\t")[skip::interval, 1]
    volume = np.loadtxt(root / f"{prefix}_V_{hydrograph_id}.txt", delimiter="\t")[skip::interval, :num_points]
    precipitation = np.loadtxt(root / f"{prefix}_Pr_{hydrograph_id}.txt", delimiter="\t")[skip::interval]
    peak = int(np.argmax(inflow))
    return {
        "water_depth": water_depth[: peak + 25],
        "volume": volume[: peak + 25],
        "precipitation": precipitation[: peak + 25] * 2.7778e-7,
        "inflow_hydrograph": inflow[: peak + 25],
    }


def list_hydrograph_ids(data_dir: Union[str, Path], prefix: str = HYDROGRAPH_PREFIX) -> List[str]:
    """Ids of every ``<prefix>_WD_<id>.txt`` file in ``data_dir`` (directory order, like HydroGraphDataset)."""
    ids = []
    for path in Path(data_dir).iterdir():
        name = path.name
        if name.startswith(f"{prefix}_WD_") and name.endswith(".txt"):
            parts = name.split("_")
            if len(parts) >= 3:
                ids.append(Path(parts[2]).stem)
    return ids


def read_hydrograph_ids(spec: Union[str, Path, Sequence[str]], data_dir: Union[str, Path]) -> List[str]:
    """Hydrograph ids from a list or from a file (absolute, or relative to ``data_dir``), one id per line."""
    if isinstance(spec, (str, Path)):
        path = Path(spec)
        if not path.is_absolute():
            path = Path(data_dir) / path
        if not path.exists():
            raise FileNotFoundError(f"Hydrograph IDs file not found: {path}")
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [str(item) for item in spec]


def knn_edge_index(xy_coords: np.ndarray, k: int = 4) -> np.ndarray:
    """``(2, E)`` edges from every node to its ``k`` nearest other nodes (k-d tree query of ``k + 1``)."""
    from scipy.spatial import KDTree

    _, neighbors = KDTree(xy_coords).query(xy_coords, k=k + 1)
    return np.vstack([(i, nbr) for i, nbrs in enumerate(neighbors) for nbr in nbrs if nbr != i]).T


def mesh_edge_features(xy_coords: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """Standardised (dx, dy, distance) of every edge (source minus destination)."""
    row, col = edge_index
    relative = xy_coords[row] - xy_coords[col]
    distance = np.linalg.norm(relative, axis=1)
    relative = (relative - np.mean(relative, axis=0)) / (np.std(relative, axis=0) + EPSILON)
    distance = (distance - np.mean(distance)) / (np.std(distance) + EPSILON)
    return np.hstack([relative, distance[:, None]])


def mesh_node_features(
    static: Mapping[str, np.ndarray],
    water_depth: np.ndarray,
    volume: np.ndarray,
    precipitation: np.ndarray,
    inflow: np.ndarray,
    time_step: int,
) -> np.ndarray:
    """HydroGraphDataset.create_node_features: 12 static / forcing columns, then depth and volume windows."""
    num_nodes = static["xy_coords"].shape[0]
    return np.hstack(
        [static[key] for key in STATIC_ORDER]
        + [
            np.full((num_nodes, 1), inflow[time_step]),
            np.full((num_nodes, 1), precipitation[time_step]),
            water_depth.T,
            volume.T,
        ]
    )


def dynamic_stats(hydrographs: Iterable[Mapping[str, np.ndarray]]) -> Dict[str, Dict[str, float]]:
    """Mean and standard deviation of every dynamic variable over all values of the given hydrographs."""
    series = list(hydrographs)
    if not series:
        raise ValueError("Dynamic normalisation statistics need at least one training hydrograph.")
    stats: Dict[str, Dict[str, float]] = {}
    for key in DYNAMIC_KEYS:
        values = np.concatenate([np.asarray(item[key]).flatten() for item in series])
        stats[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return stats


class HydrographMesh:
    """Normalised static fields, graph and normalised hydrographs shared by the training and test samples."""

    def __init__(
        self,
        static: Mapping[str, np.ndarray],
        area_denorm: np.ndarray,
        hydrographs: Mapping[str, Mapping[str, np.ndarray]],
        static_stats: Mapping[str, Any],
        dynamic_stats: Mapping[str, Mapping[str, float]],
        k: int = 4,
    ):
        self.static = dict(static)
        self.area_denorm = np.asarray(area_denorm)
        self.static_stats = {key: dict(value) for key, value in static_stats.items()}
        self.dynamic_stats = {key: dict(value) for key, value in dynamic_stats.items()}
        self.num_nodes = int(self.static["xy_coords"].shape[0])
        self.edge_index = knn_edge_index(self.static["xy_coords"], k=k)
        self.edge_features = mesh_edge_features(self.static["xy_coords"], self.edge_index)
        self.edge_index_tensor = torch.from_numpy(np.ascontiguousarray(self.edge_index)).long()
        self.edge_features_tensor = torch.tensor(self.edge_features, dtype=torch.float)
        self.hydrographs = {
            hid: {
                key: _normalize(np.asarray(series[key]), self.dynamic_stats[key]["mean"], self.dynamic_stats[key]["std"])
                for key in DYNAMIC_KEYS
            }
            for hid, series in hydrographs.items()
        }
        infiltration = _denormalize(
            self.static["infiltration"], self.static_stats["infiltration"]["mean"], self.static_stats["infiltration"]["std"]
        )
        self.area_sum = float(np.sum(self.area_denorm))
        self.infiltration_area_sum = float(np.sum(infiltration * self.area_denorm)) / 100.0

    def node_features(self, hydrograph_id: str, start: int, end: int, time_step: int) -> np.ndarray:
        dyn = self.hydrographs[hydrograph_id]
        return mesh_node_features(
            self.static,
            dyn["water_depth"][start:end, :],
            dyn["volume"][start:end, :],
            dyn["precipitation"],
            dyn["inflow_hydrograph"],
            time_step,
        )

    def _denorm(self, value: float, key: str) -> float:
        return value * self.dynamic_stats[key]["std"] + self.dynamic_stats[key]["mean"]

    def physics_data(self, hydrograph_id: str, t_idx: int, n_time_steps: int) -> Dict[str, float]:
        """Physics dict of the training sample starting at ``t_idx`` (HydroGraphDataset, ``return_physics=True``)."""
        dyn = self.hydrographs[hydrograph_id]
        inflow, precipitation, volume = dyn["inflow_hydrograph"], dyn["precipitation"], dyn["volume"]
        target_time = t_idx + n_time_steps
        prev_time = target_time - 1
        has_next = target_time + 1 < volume.shape[0]
        future_volume = float(np.sum(volume[target_time + 1, :])) if has_next else float(np.sum(volume[target_time, :]))
        next_index = target_time + 1 if target_time + 1 < inflow.shape[0] else target_time
        stats = self.dynamic_stats
        return {
            "flow_future": float(self._denorm(inflow[t_idx + n_time_steps], "inflow_hydrograph")),
            "precip_future": float(self._denorm(precipitation[t_idx + n_time_steps], "precipitation")),
            "past_volume": float(np.sum(volume[prev_time, :])),
            "future_volume": future_volume,
            "avg_inflow": self._denorm(float((inflow[prev_time] + inflow[target_time]) / 2), "inflow_hydrograph"),
            "avg_precipitation": self._denorm(float((precipitation[prev_time] + precipitation[target_time]) / 2), "precipitation"),
            "next_inflow": float(self._denorm(inflow[next_index], "inflow_hydrograph")),
            "next_precip": float(self._denorm(precipitation[next_index], "precipitation")),
            "volume_mean": float(stats["volume"]["mean"]),
            "volume_std": float(stats["volume"]["std"]),
            "inflow_mean": float(stats["inflow_hydrograph"]["mean"]),
            "inflow_std": float(stats["inflow_hydrograph"]["std"]),
            "precip_mean": float(stats["precipitation"]["mean"]),
            "precip_std": float(stats["precipitation"]["std"]),
            "num_nodes": float(self.num_nodes),
            "area_sum": self.area_sum,
            "infiltration_area_sum": self.infiltration_area_sum,
        }


class HydrographWindows(TorchDataset):
    """Training samples: ``({"node_features", "edge_features", "edge_index", "physics"}, target (N, 2))``.

    The target is the normalised change of water depth and volume from the last window step to the next.
    Batch several samples with :func:`hydrograph_collate`.
    """

    def __init__(self, mesh: HydrographMesh, hydrograph_ids: Sequence[str], n_time_steps: int = 2):
        self.mesh = mesh
        self.hydrograph_ids = list(hydrograph_ids)
        self.n_time_steps = int(n_time_steps)
        self.sample_index: List[Tuple[str, int]] = []
        for hid in self.hydrograph_ids:
            length = mesh.hydrographs[hid]["water_depth"].shape[0]
            self.sample_index.extend((hid, t) for t in range(length - self.n_time_steps))

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int):
        hid, t_idx = self.sample_index[idx]
        dyn = self.mesh.hydrographs[hid]
        n = self.n_time_steps
        features = self.mesh.node_features(hid, t_idx, t_idx + n, t_idx)
        target_time = t_idx + n
        target = np.stack(
            [
                dyn["water_depth"][target_time, :] - dyn["water_depth"][target_time - 1, :],
                dyn["volume"][target_time, :] - dyn["volume"][target_time - 1, :],
            ],
            axis=1,
        )
        inputs = {
            "node_features": torch.tensor(features, dtype=torch.float),
            "edge_features": self.mesh.edge_features_tensor,
            "edge_index": self.mesh.edge_index_tensor,
            "physics": self.mesh.physics_data(hid, t_idx, n),
        }
        return inputs, torch.tensor(target, dtype=torch.float)


class HydrographRollouts(TorchDataset):
    """Test hydrographs for autoregressive rollouts.

    Item ``i``: ``({"node_features" (N, 16) of the first window, "edge_features", "edge_index", "inflow",
    "precipitation" (rollout_length,)}, {"water_depth", "volume"} (rollout_length, N))``, all normalised;
    the forcings and targets cover the steps ``n_time_steps ... n_time_steps + rollout_length - 1``.
    """

    def __init__(self, mesh: HydrographMesh, hydrograph_ids: Sequence[str], n_time_steps: int = 2, rollout_length: int = 30):
        self.mesh = mesh
        self.hydrograph_ids = list(hydrograph_ids)
        self.n_time_steps = int(n_time_steps)
        self.rollout_length = int(rollout_length)
        for hid in self.hydrograph_ids:
            length = mesh.hydrographs[hid]["water_depth"].shape[0]
            if length < self.n_time_steps + self.rollout_length:
                raise ValueError(
                    f"Hydrograph {hid} does not have enough time steps for the specified rollout_length "
                    f"({length} < {self.n_time_steps} + {self.rollout_length})."
                )

    @property
    def depth_stats(self) -> Dict[str, float]:
        return dict(self.mesh.dynamic_stats["water_depth"])

    @property
    def volume_stats(self) -> Dict[str, float]:
        return dict(self.mesh.dynamic_stats["volume"])

    def __len__(self) -> int:
        return len(self.hydrograph_ids)

    def __getitem__(self, idx: int):
        hid = self.hydrograph_ids[idx]
        dyn = self.mesh.hydrographs[hid]
        n, length = self.n_time_steps, self.rollout_length
        features = self.mesh.node_features(hid, 0, n, 0)
        window = slice(n, n + length)
        inputs = {
            "node_features": torch.tensor(features, dtype=torch.float),
            "edge_features": self.mesh.edge_features_tensor,
            "edge_index": self.mesh.edge_index_tensor,
            "inflow": torch.tensor(dyn["inflow_hydrograph"][window], dtype=torch.float),
            "precipitation": torch.tensor(dyn["precipitation"][window], dtype=torch.float),
        }
        targets = {
            "water_depth": torch.tensor(dyn["water_depth"][window], dtype=torch.float),
            "volume": torch.tensor(dyn["volume"][window], dtype=torch.float),
        }
        return inputs, targets


def hydrograph_collate(batch: Sequence[Tuple[Mapping[str, Any], torch.Tensor]]):
    """Concatenate mesh samples into one graph (like PyTorch Geometric ``Batch.from_data_list``).

    Returns ``({"node_features", "edge_features", "edge_index" (offset per sample), "batch" (graph index
    per node), "physics" (float32 tensors, one value per sample)}, targets (sum of N, 2))``.
    """
    inputs, targets = zip(*batch)
    offsets, offset = [], 0
    for item in inputs:
        offsets.append(offset)
        offset += item["node_features"].shape[0]
    collated: Dict[str, Any] = {
        "node_features": torch.cat([item["node_features"] for item in inputs], dim=0),
        "edge_features": torch.cat([item["edge_features"] for item in inputs], dim=0),
        "edge_index": torch.cat([item["edge_index"] + shift for item, shift in zip(inputs, offsets)], dim=1),
        "batch": torch.cat(
            [torch.full((item["node_features"].shape[0],), i, dtype=torch.long) for i, item in enumerate(inputs)]
        ),
    }
    if all("physics" in item for item in inputs):
        collated["physics"] = {
            key: torch.tensor([item["physics"][key] for item in inputs], dtype=torch.float) for key in inputs[0]["physics"]
        }
    return collated, torch.cat(list(targets), dim=0)


def build_hydrograph_bundle(
    static_raw: Mapping[str, np.ndarray],
    hydrographs: Mapping[str, Mapping[str, np.ndarray]],
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    *,
    n_time_steps: int = 2,
    k: int = 4,
    rollout_length: int = 30,
    static_stats: Optional[Mapping[str, Any]] = None,
    dyn_stats: Optional[Mapping[str, Mapping[str, float]]] = None,
    val_ids: Sequence[str] = (),
    dataset_name: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> DataBundle:
    """Normalise raw mesh data and build the train (windows), optional val and test (rollouts) splits."""
    train_ids, val_ids, test_ids = list(train_ids), list(val_ids), list(test_ids)
    if not train_ids:
        raise ValueError("At least one training hydrograph is needed (normalisation statistics come from them).")
    unknown = [hid for hid in train_ids + val_ids + test_ids if hid not in hydrographs]
    if unknown:
        raise ValueError(f"Hydrographs {unknown} were not loaded.")
    static, static_stats = standardize_static(static_raw, static_stats)
    if dyn_stats is None:
        dyn_stats = dynamic_stats(hydrographs[hid] for hid in train_ids)
    mesh = HydrographMesh(static, np.asarray(static_raw["area"]), hydrographs, static_stats, dyn_stats, k=k)
    splits = {"train": DataSplit(inputs=HydrographWindows(mesh, train_ids, n_time_steps), targets=None)}
    if val_ids:
        splits["val"] = DataSplit(inputs=HydrographRollouts(mesh, val_ids, n_time_steps, rollout_length), targets=None)
    splits["test"] = DataSplit(inputs=HydrographRollouts(mesh, test_ids, n_time_steps, rollout_length), targets=None)
    return DataBundle(
        splits=splits,
        feature_spec=FeatureSpec(
            input_dim=12 + 2 * int(n_time_steps),
            description=(
                "Mesh node features: x, y, area, elevation, slope, aspect, curvature, Manning's n, flow "
                "accumulation, infiltration, inflow, precipitation, then the water depth and volume windows; "
                "edge features: standardised dx, dy, distance of the k-nearest-neighbour graph."
            ),
            extra={"edge_dim": 3, "num_nodes": mesh.num_nodes, "num_edges": int(mesh.edge_index.shape[1]), "n_time_steps": int(n_time_steps)},
        ),
        label_spec=LabelSpec(
            num_targets=2,
            task_type="regression",
            description="Normalised change of water depth and volume per mesh node over one 20-minute step.",
            extra={"rollout_length": int(rollout_length)},
        ),
        metadata={
            "dataset": dataset_name,
            "source_dataset": dataset_name,
            "hazard_task": "flood.inundation",
            "inundation_layout": "mesh_rollout",
            "static_stats": mesh.static_stats,
            "dynamic_stats": mesh.dynamic_stats,
            "n_time_steps": int(n_time_steps),
            "rollout_length": int(rollout_length),
            "train_ids": train_ids,
            "val_ids": val_ids,
            "test_ids": test_ids,
            **dict(metadata or {}),
        },
    )


class HydroGraphNetWhiteRiverDataset(Dataset):
    """Reader for a local copy of the HydroGraphNet White River dataset (Zenodo 14969507).

    ``data_dir`` holds the unzipped ``HydroGraphNet.zip`` (4,787-cell HEC-RAS mesh of the White River near
    Muncie, Indiana, 500 hydrographs). ``train_ids`` defaults to the release's ``train.txt``; the release
    has no test list (the example's ``Test/test.txt`` is not published and ``train.txt`` lists all 500
    hydrographs), so ``test_ids`` (a list or a file) must be given; by default it is every hydrograph not in
    ``train_ids``. Ids in ``test_ids`` are removed from the training list. Normalisation statistics are
    computed from the training hydrographs as the example's training run does (``norm_stats="compute"``);
    ``norm_stats="files"`` uses the released ``static_norm_stats.json`` / ``dynamic_norm_stats.json``
    instead, as the example's inference script does. PyHazards downloads nothing.
    """

    name = "hydrographnet_white_river"

    def __init__(
        self,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        prefix: str = HYDROGRAPH_PREFIX,
        train_ids: Union[str, Sequence[str], None] = "train.txt",
        test_ids: Union[str, Sequence[str], None] = None,
        val_ids: Union[str, Sequence[str], None] = None,
        n_time_steps: int = 2,
        k: int = 4,
        rollout_length: int = 30,
        norm_stats: str = "compute",
        skip: int = 72,
        interval: int = 1,
    ):
        super().__init__(cache_dir=cache_dir)
        if data_dir is None:
            raise ValueError(
                "hydrographnet_white_river needs data_dir: the unzipped HydroGraphNet.zip from "
                "https://zenodo.org/records/14969507 (8.3 GB, CC BY 4.0)."
            )
        if norm_stats not in {"compute", "files"}:
            raise ValueError("norm_stats must be 'compute' or 'files'.")
        self.data_dir = Path(data_dir).expanduser()
        self.prefix = prefix
        self.train_ids, self.test_ids, self.val_ids = train_ids, test_ids, val_ids
        self.n_time_steps = int(n_time_steps)
        self.k = int(k)
        self.rollout_length = int(rollout_length)
        self.norm_stats = norm_stats
        self.skip = int(skip)
        self.interval = int(interval)

    def _ids(self) -> Tuple[List[str], List[str], List[str]]:
        available = list_hydrograph_ids(self.data_dir, self.prefix)
        train = read_hydrograph_ids(self.train_ids, self.data_dir) if self.train_ids is not None else list(available)
        val = read_hydrograph_ids(self.val_ids, self.data_dir) if self.val_ids is not None else []
        if self.test_ids is not None:
            test = read_hydrograph_ids(self.test_ids, self.data_dir)
        else:
            test = [hid for hid in available if hid not in set(train) | set(val)]
        if not test:
            raise ValueError(
                "No test hydrographs: the HydroGraphNet release lists all 500 hydrographs in train.txt and ships "
                "no test list, so pass test_ids (a list or a file of ids)."
            )
        held_out = set(test) | set(val)
        return [hid for hid in train if hid not in held_out], val, test

    def _load(self) -> DataBundle:
        static_raw = read_static(self.data_dir, self.prefix)
        num_points = static_raw["xy_coords"].shape[0]
        train_ids, val_ids, test_ids = self._ids()
        hydrographs = {
            hid: read_hydrograph(self.data_dir, hid, num_points, self.prefix, interval=self.interval, skip=self.skip)
            for hid in dict.fromkeys(train_ids + val_ids + test_ids)
        }
        static_stats = dyn_stats = None
        if self.norm_stats == "files":
            static_stats = json.loads((self.data_dir / STATIC_NORM_STATS_FILE).read_text(encoding="utf-8"))
            dyn_stats = json.loads((self.data_dir / DYNAMIC_NORM_STATS_FILE).read_text(encoding="utf-8"))
        return build_hydrograph_bundle(
            static_raw,
            hydrographs,
            train_ids,
            test_ids,
            val_ids=val_ids,
            n_time_steps=self.n_time_steps,
            k=self.k,
            rollout_length=self.rollout_length,
            static_stats=static_stats,
            dyn_stats=dyn_stats,
            dataset_name=self.name,
            metadata={"data_dir": str(self.data_dir), "prefix": self.prefix, "norm_stats": self.norm_stats},
        )


def synthetic_hydrograph_mesh(
    nodes: int = 64,
    hydrographs: int = 5,
    steps: int = 48,
    seed: int = 0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
    """Random static fields on scattered points and toy hydrographs, in the layout :func:`read_static` /
    :func:`read_hydrograph` return. The numbers mean nothing hydrologically."""
    rng = np.random.default_rng(seed)
    xy = np.column_stack([406000 + rng.uniform(0, 2000, nodes), 1802000 + rng.uniform(0, 1000, nodes)])
    elevation = 285 + 0.002 * (xy[:, 0] - xy[:, 0].min()) + rng.normal(0, 0.5, nodes)
    static = {
        "xy_coords": xy,
        "area": rng.uniform(5000, 8000, (nodes, 1)),
        "elevation": elevation.reshape(-1, 1),
        "slope": rng.gamma(1.0, 2.0, (nodes, 1)),
        "aspect": rng.uniform(0, 360, (nodes, 1)),
        "curvature": rng.normal(0, 1.5, (nodes, 1)),
        "manning": rng.choice([0.035, 0.06, 0.1, 0.15], (nodes, 1)),
        "flow_accum": rng.integers(0, 200, (nodes, 1)).astype(float),
        "infiltration": rng.uniform(1, 100, (nodes, 1)),
    }
    series: Dict[str, Dict[str, np.ndarray]] = {}
    low = elevation - elevation.min()
    for h in range(hydrographs):
        peak = rng.integers(steps // 3, steps - 26) if steps > 30 else steps // 2
        rise = np.clip(np.arange(steps) / max(peak, 1), 0, None)
        inflow = 20 + rng.uniform(150, 400) * np.where(np.arange(steps) <= peak, rise, np.exp(-(np.arange(steps) - peak) / 15))
        precipitation = np.clip(rng.normal(1.5e-6, 1e-6, steps), 0, None)
        stage = (inflow - inflow.min()) / (np.ptp(inflow) + 1e-9) * 2.5
        depth = np.clip(stage[:, None] - low[None, :] + rng.normal(0, 0.05, (steps, nodes)), 0, None)
        series[f"H{h + 1}"] = {
            "water_depth": depth,
            "volume": depth * static["area"].reshape(1, -1),
            "precipitation": precipitation,
            "inflow_hydrograph": inflow,
        }
    return static, series


def write_hydrograph_files(
    directory: Union[str, Path],
    static_raw: Mapping[str, np.ndarray],
    hydrographs: Mapping[str, Mapping[str, np.ndarray]],
    prefix: str = HYDROGRAPH_PREFIX,
    skip: int = 72,
    extra_cells: int = 0,
    train_ids: Optional[Sequence[str]] = None,
) -> Path:
    """Write mesh data in the HydroGraphNet release layout (e.g. synthetic data for the official code).

    ``skip`` copies of the first step are prepended to every series (the reader drops them again),
    precipitation is written in the release's units (divided by 2.7778e-7), the inflow file gets a
    leading time column, and ``extra_cells`` padding rows / columns imitate the release's files, which
    hold more cells than ``M80_XY.txt``. ``train_ids`` are written to ``train.txt``.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    fmt = "%.9f"
    xy = np.asarray(static_raw["xy_coords"], dtype=np.float64)
    np.savetxt(root / f"{prefix}_XY.txt", xy, delimiter="\t", fmt=fmt)
    for key, code in list(STATIC_FILES.items())[1:]:
        column = np.asarray(static_raw[key], dtype=np.float64).reshape(-1)
        column = np.concatenate([column, np.full(extra_cells, column[-1])])
        np.savetxt(root / f"{prefix}_{code}.txt", column, fmt=fmt)

    def _pad(series: np.ndarray) -> np.ndarray:
        return np.concatenate([np.repeat(series[:1], skip, axis=0), series], axis=0)

    for hid, series in hydrographs.items():
        depth = np.asarray(series["water_depth"], dtype=np.float64)
        volume = np.asarray(series["volume"], dtype=np.float64)
        if extra_cells:
            depth = np.hstack([depth, np.zeros((depth.shape[0], extra_cells))])
            volume = np.hstack([volume, np.zeros((volume.shape[0], extra_cells))])
        inflow = _pad(np.asarray(series["inflow_hydrograph"], dtype=np.float64))
        np.savetxt(root / f"{prefix}_WD_{hid}.txt", _pad(depth), delimiter="\t", fmt=fmt)
        np.savetxt(root / f"{prefix}_V_{hid}.txt", _pad(volume), delimiter="\t", fmt=fmt)
        np.savetxt(
            root / f"{prefix}_US_InF_{hid}.txt",
            np.column_stack([np.arange(inflow.shape[0]) * 1200.0, inflow]),
            delimiter="\t",
            fmt=fmt,
        )
        np.savetxt(
            root / f"{prefix}_Pr_{hid}.txt", _pad(np.asarray(series["precipitation"], dtype=np.float64) / 2.7778e-7), fmt=fmt
        )
    if train_ids is not None:
        (root / "train.txt").write_text("".join(f"{hid}\n" for hid in train_ids), encoding="utf-8")
    return root


class SyntheticFloodMeshDataset(Dataset):
    """Synthetic mesh hydrographs in the HydroGraphNet layout (random fields; for smoke tests only).

    Same pipeline as ``hydrographnet_white_river`` (normalisation, k-nearest-neighbour graph, 16 node
    features, windows with physics data, rollout test split) on random points with toy hydrographs.
    """

    name = "flood_mesh_synthetic"

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        nodes: int = 64,
        train_hydrographs: int = 4,
        test_hydrographs: int = 2,
        steps: int = 48,
        n_time_steps: int = 2,
        k: int = 4,
        rollout_length: int = 30,
        seed: int = 0,
        micro: bool = False,
    ):
        super().__init__(cache_dir=cache_dir)
        self.nodes = 24 if micro else int(nodes)
        self.train_hydrographs = 2 if micro else int(train_hydrographs)
        self.test_hydrographs = 1 if micro else int(test_hydrographs)
        self.steps = 16 if micro else int(steps)
        self.rollout_length = 6 if micro else int(rollout_length)
        self.n_time_steps = int(n_time_steps)
        self.k = int(k)
        self.seed = int(seed)
        if self.nodes <= self.k:
            raise ValueError(f"nodes ({self.nodes}) must exceed k ({self.k}).")
        if self.steps < self.n_time_steps + self.rollout_length:
            raise ValueError(f"steps ({self.steps}) must be at least n_time_steps + rollout_length.")

    def _load(self) -> DataBundle:
        total = self.train_hydrographs + self.test_hydrographs
        static, series = synthetic_hydrograph_mesh(self.nodes, total, self.steps, self.seed)
        ids = list(series)
        return build_hydrograph_bundle(
            static,
            series,
            ids[: self.train_hydrographs],
            ids[self.train_hydrographs :],
            n_time_steps=self.n_time_steps,
            k=self.k,
            rollout_length=self.rollout_length,
            dataset_name=self.name,
            metadata={"synthetic": True},
        )


__all__ = [
    "DYNAMIC_KEYS",
    "HydroGraphNetWhiteRiverDataset",
    "HydrographMesh",
    "HydrographRollouts",
    "HydrographWindows",
    "STATIC_FILES",
    "SyntheticFloodMeshDataset",
    "build_hydrograph_bundle",
    "dynamic_stats",
    "hydrograph_collate",
    "knn_edge_index",
    "list_hydrograph_ids",
    "mesh_edge_features",
    "mesh_node_features",
    "read_hydrograph",
    "read_hydrograph_ids",
    "read_static",
    "standardize_static",
    "synthetic_hydrograph_mesh",
    "write_hydrograph_files",
]
