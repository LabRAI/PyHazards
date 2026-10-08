"""Flood benchmark: per-basin streamflow metrics and inundation depth / extent metrics.

``flood.streamflow`` follows Kratzert et al. (HESS 2019) and the NeuralHydrology tester: every model
predicts the discharge of each day of the evaluation period from the ``seq_length`` days before it (the
last step of its output sequence); predictions and observations are put back into physical units with
the training scaler, negative predictions are set to zero (``clip_negative``, as in the 2019 code and
NeuralHydrology's CAMELS configurations), and NeuralHydrology's metrics are computed per basin over the
period (:mod:`pyhazards.metrics.hydrology`). Reported values are the median over basins (``nse``,
``kge``, ...), the mean over basins (``nse_mean``, ...), ``n_basins`` and ``n_basins_nse_le_0``; the
per-basin values are in ``metadata["per_basin"]``.

``flood.inundation`` compares predicted and simulated water depth (:mod:`pyhazards.metrics.inundation`)
on three kinds of data:

- rasters (``(batch, 1, H, W)`` depth, or channels-last multi-variable outputs such as UrbanFloodCast's
  ``(batch, Sy, Sx, T, 3)`` depth and discharges with ``metadata["depth_channel"]`` and a NaN mask in the
  split metadata): the model maps the split inputs to the targets in batches;
- mesh hydrographs (``hydrographnet_white_river``, ``flood_mesh_synthetic``): every test hydrograph is
  rolled out autoregressively with the model's ``rollout`` method (HydroGraphNet's ``inference.py``)
  and the predicted depths are denormalised to metres; ``rollout_rmse`` is the depth RMSE per step
  averaged over steps and hydrographs (the per-step curve is in the report metadata);
- graph-temporal node series (``GraphTemporalDataset``), scored per node.

Reported: ``pixel_mae`` and ``rmse`` (depth errors over all cells), ``iou`` / ``f1`` (wet cells: prediction
>= 0.5 m, target > 0), the depth CSI at 1 / 10 / 50 cm and the per-sample relative L2, NSE and Pearson r
of the UrbanFloodCast evaluation, averaged over samples (events or hydrographs).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..configs import ExperimentConfig
from ..datasets.base import DataBundle
from ..datasets.flood.hydrograph import HydrographRollouts
from ..datasets.flood.streamflow import StreamflowWindows, scaler_from_metadata
from ..datasets.graph import GraphTemporalDataset, graph_collate
from ..metrics.hydrology import STREAMFLOW_METRICS, aggregate_basin_metrics, calculate_metrics, streamflow_metric_names
from ..metrics.inundation import INUNDATION_METRICS, inundation_metrics, rollout_rmse
from .base import Benchmark
from .registry import register_benchmark
from .schemas import BenchmarkResult


def _model_device(model: nn.Module):
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else None


def _to_device(obj: Any, device) -> Any:
    if device is None:
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: _to_device(value, device) for key, value in obj.items()}
    return obj


def point_prediction(model: nn.Module, outputs: Any) -> torch.Tensor:
    """Deterministic prediction ``(batch, time, n_targets)`` from a streamflow model's output.

    NeuralHydrology-style models return a dict: regression heads give ``y_hat``; probabilistic heads
    (e.g. CMAL) are reduced with the model's ``point_prediction`` (the mixture mean). Plain tensors are
    used as they are.
    """
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, dict):
        if "y_hat" in outputs:
            return outputs["y_hat"]
        if hasattr(model, "point_prediction"):
            return model.point_prediction(outputs)
    raise ValueError(
        "Streamflow models must return a tensor of shape (batch, time, n_targets) or a dict with 'y_hat' "
        f"(or a model with point_prediction); got {type(outputs).__name__}."
    )


def _last_step(prediction: torch.Tensor, n_targets: int) -> torch.Tensor:
    if prediction.ndim == 3:
        return prediction[:, -1, :]
    if prediction.ndim == 2 and prediction.shape[-1] == n_targets:
        return prediction
    raise ValueError(
        "Streamflow predictions must have shape (batch, time, n_targets) or (batch, n_targets); "
        f"got {tuple(prediction.shape)} for {n_targets} target(s)."
    )


def streamflow_predictions(
    model: nn.Module, windows: StreamflowWindows, n_targets: int, batch_size: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalised (simulation, observation) arrays ``(n_samples, n_targets)`` at the last window step."""
    loader = DataLoader(windows, batch_size=batch_size, shuffle=False)
    device = _model_device(model)
    sims: List[np.ndarray] = []
    obs: List[np.ndarray] = []
    with torch.no_grad():
        for inputs, target in loader:
            outputs = model(_to_device(inputs, device))
            prediction = _last_step(point_prediction(model, outputs), n_targets)
            sims.append(prediction.detach().float().cpu().numpy())
            obs.append(target[:, -1, :].numpy())
    if not sims:
        return np.zeros((0, n_targets)), np.zeros((0, n_targets))
    return np.concatenate(sims), np.concatenate(obs)


def evaluate_streamflow(
    model: nn.Module,
    data: DataBundle,
    split_name: str = "test",
    batch_size: int = 256,
    clip_negative: bool = True,
    metrics: Any = None,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Summary metrics and per-basin metrics of ``model`` on a streamflow split."""
    split = data.get_split(split_name)
    windows = split.inputs
    if not isinstance(windows, StreamflowWindows):
        raise ValueError(
            "flood.streamflow needs a streamflow dataset (StreamflowWindows splits, e.g. camels_us_streamflow, "
            f"caravan_streamflow or flood_streamflow_synthetic); got {type(windows).__name__}."
        )
    targets = list(data.metadata.get("target_variables", ["discharge"]))
    scaler = scaler_from_metadata(data.metadata)
    sim, obs = streamflow_predictions(model, windows, len(targets), batch_size=batch_size)
    sim = scaler.rescale_target(sim)
    obs = scaler.rescale_target(obs)
    if clip_negative:
        sim = np.where(sim < 0, 0.0, sim)

    names = list(STREAMFLOW_METRICS if metrics is None else metrics)
    per_basin: Dict[str, Dict[str, float]] = {}
    for basin_index, basin in enumerate(windows.basins):
        rows = np.flatnonzero(windows.sample_basin == basin_index)
        if rows.size == 0:
            continue
        rows = rows[np.argsort(windows.sample_date[rows], kind="stable")]
        dates = windows.sample_date[rows]
        values: Dict[str, float] = {}
        for t, target in enumerate(targets):
            basin_obs = obs[rows, t]
            if np.isnan(basin_obs).all():
                continue
            result = calculate_metrics(basin_obs, sim[rows, t], dates=dates, metrics=names)
            if len(targets) > 1:
                result = {f"{target}_{key}": value for key, value in result.items()}
            values.update(result)
        if values:
            per_basin[basin] = values
    return aggregate_basin_metrics(per_basin), per_basin


def _batched_predictions(model: nn.Module, inputs: torch.Tensor, batch_size: int) -> torch.Tensor:
    device = _model_device(model)
    preds = []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            batch = inputs[start : start + batch_size]
            preds.append(model(batch if device is None else batch.to(device)).detach().cpu())
    return torch.cat(preds, dim=0)


def evaluate_mesh_rollouts(model: nn.Module, rollouts: HydrographRollouts) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Roll out every test hydrograph with ``model.rollout`` and score the depths in metres."""
    if not hasattr(model, "rollout"):
        raise ValueError(
            "flood.inundation on mesh hydrographs rolls models out autoregressively; "
            f"{type(model).__name__} has no rollout(...) method (see HydroGraphNet)."
        )
    device = _model_device(model)
    stats = rollouts.depth_stats
    scale = stats["std"] + 1e-8
    preds, targets, curves = [], [], []
    with torch.no_grad():
        for idx in range(len(rollouts)):
            inputs, target = rollouts[idx]
            inputs = _to_device(inputs, device)
            out = model.rollout(
                inputs["node_features"],
                inputs["edge_features"],
                inputs["edge_index"],
                inputs["inflow"],
                inputs["precipitation"],
                n_time_steps=rollouts.n_time_steps,
            )
            pred = out["water_depth"].detach().cpu().float()
            truth = target["water_depth"].float()
            curves.append(rollout_rmse(pred, truth))
            preds.append(pred * scale + stats["mean"])
            targets.append(truth * scale + stats["mean"])
    metrics = inundation_metrics(torch.stack(preds), torch.stack(targets))
    curve = torch.stack(curves).mean(dim=0) if curves else torch.zeros(0)
    metrics["rollout_rmse"] = float(curve.mean() * scale) if curves else float("nan")
    metadata = {
        "rollout_rmse_per_step_m": [float(v) * scale for v in curve],
        "rollout_rmse_per_step_normalized": [float(v) for v in curve],
        "hydrographs": list(rollouts.hydrograph_ids),
        "rollout_length": rollouts.rollout_length,
    }
    return metrics, metadata


def evaluate_inundation(
    model: nn.Module, data: DataBundle, split_name: str = "test", batch_size: int = 1
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Inundation metrics of ``model`` on a raster, mesh-rollout or graph-temporal split."""
    split = data.get_split(split_name)
    if isinstance(split.inputs, HydrographRollouts):
        return evaluate_mesh_rollouts(model, split.inputs)
    if isinstance(split.inputs, GraphTemporalDataset):
        loader = DataLoader(split.inputs, batch_size=4, shuffle=False, collate_fn=graph_collate)
        preds, targets = [], []
        with torch.no_grad():
            for batch, target in loader:
                preds.append(model(batch))
                targets.append(target)
        return inundation_metrics(torch.cat(preds, dim=0), torch.cat(targets, dim=0)), {}
    if not isinstance(split.inputs, torch.Tensor) or not isinstance(split.targets, torch.Tensor):
        raise ValueError(
            "flood.inundation needs tensor splits (rasters), mesh hydrographs (HydrographRollouts) or a "
            f"GraphTemporalDataset; got inputs of type {type(split.inputs).__name__}."
        )
    preds = _batched_predictions(model, split.inputs, max(1, int(batch_size)))
    metrics = inundation_metrics(
        preds,
        split.targets,
        mask=split.metadata.get("mask"),
        depth_index=data.metadata.get("depth_channel"),
    )
    return metrics, {}


class FloodBenchmark(Benchmark):
    name = "flood"
    hazard_task = "flood.streamflow"
    metric_names_by_task = {
        "flood.streamflow": streamflow_metric_names(),
        "flood.inundation": list(INUNDATION_METRICS) + ["rollout_rmse"],
    }

    def evaluate(self, model: nn.Module, data: DataBundle, config: ExperimentConfig) -> BenchmarkResult:
        params = config.benchmark.params
        metadata: Dict[str, Any] = {
            "split": config.benchmark.eval_split,
            "dataset_name": data.metadata.get("dataset"),
            "source_dataset": data.metadata.get("source_dataset", data.metadata.get("dataset")),
        }
        if config.benchmark.hazard_task == "flood.inundation":
            metrics, extra = evaluate_inundation(
                model, data, split_name=config.benchmark.eval_split, batch_size=int(params.get("batch_size", 1))
            )
            metadata.update(extra)
        else:
            metrics, per_basin = evaluate_streamflow(
                model,
                data,
                split_name=config.benchmark.eval_split,
                batch_size=int(params.get("batch_size", 256)),
                clip_negative=bool(params.get("clip_negative", True)),
            )
            metadata["per_basin"] = per_basin
            metadata["aggregation"] = "median over basins (<metric>), mean over basins (<metric>_mean)"
        return BenchmarkResult(
            benchmark_name=self.name,
            hazard_task=config.benchmark.hazard_task,
            metrics=metrics,
            metadata=metadata,
        )


register_benchmark(FloodBenchmark.name, FloodBenchmark)

__all__ = [
    "FloodBenchmark",
    "evaluate_inundation",
    "evaluate_mesh_rollouts",
    "evaluate_streamflow",
    "inundation_metrics",
    "point_prediction",
    "streamflow_predictions",
]
