"""Tropical cyclone evaluator: track error in km and intensity errors per lead time.

Two hazard tasks are scored:

``tc.track_intensity``
    Forecast positions (and intensity) at several lead times. The data bundle's metadata gives
    ``lead_hours`` (one per forecast step), ``target_variables`` (the order of the last axis of
    tensor targets, from ``lat``, ``lon``, ``wind``, ``pres``) and ``units`` (e.g.
    ``{"wind": "kt", "pres": "hPa"}``). Models either define ``forecast(inputs)`` returning a dict
    of ``lat``, ``lon`` (degrees), ``wind`` and/or ``pres`` arrays shaped ``(storms, lead)`` or
    ``(storms, samples, lead)``, or return a tensor ``(storms, lead, variables)`` /
    ``(storms, samples, lead, variables)`` in the order of ``target_variables``.

    Metrics (NaN targets are skipped):

    * ``track_error_km`` and ``track_error_km_<h>h``: great-circle distance (haversine on a sphere
      of radius 6371.0 km) between forecast and observed centres, averaged over storms (and over
      lead times for the aggregate). ``params: {track_distance: tcn_equirectangular}`` switches to
      the TropiCycloneNet evaluation formula instead (111 km per degree, longitude difference
      scaled by the cosine of the observed latitude).
    * ``intensity_mae`` / ``intensity_mae_<h>h``: mean absolute error of the maximum sustained wind
      in the dataset's wind unit (``units["wind"]``, knots for IBTrACS, m/s for TCND / CMA);
      ``pressure_mae`` / ``pressure_mae_<h>h`` likewise for central pressure (hPa).
    * With several samples per storm, the deterministic metrics above score the sample mean (mean
      position computed on the unit sphere), and ``best_of_k_*`` metrics follow the
      TropiCycloneNet paper (Huang et al. 2025, evaluate_model_Me.py): for every storm, lead time
      and variable separately, the smallest error over the ``k`` samples, averaged over storms.
      Track, wind and pressure minima may come from different samples; these are oracle
      (best-case) errors and are not comparable with single-forecast errors.

    Benchmark ``params``: ``track_distance`` (above), ``batch_size`` (score in chunks) and ``seed``
    (default: the experiment seed; sampling models draw their noise from it).

``tc.intensity``
    One intensity value per storm and time (e.g. the 24-hour intensity or 24-hour intensity
    change). Targets are in physical units; ``metadata["prediction_transform"]`` (``{"kind":
    "minmax", "min": a, "max": b}``) maps model outputs back to them when the model predicts a
    scaled target. Metrics: ``intensity_mae``, ``intensity_rmse`` and ``intensity_mae_<h>h`` for the
    single lead time; when the split metadata has ``groups`` (e.g. the test year of each sample),
    also ``intensity_mae_<group>`` per group and their unweighted mean ``intensity_mae_group_mean``
    (the way Xu et al. 2021 and SAF-Net report yearly test MAE).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from ..configs import ExperimentConfig
from ..datasets.base import DataBundle
from .base import Benchmark
from .registry import register_benchmark
from .schemas import BenchmarkResult

EARTH_RADIUS_KM = 6371.0
TCN_KM_PER_DEGREE = 111.0
TRACK_VARIABLES = ("lat", "lon", "wind", "pres")


def great_circle_distance_km(lat1, lon1, lat2, lon2, radius_km: float = EARTH_RADIUS_KM) -> torch.Tensor:
    """Haversine distance in km between points given in degrees (broadcasting tensors)."""
    lat1, lon1, lat2, lon2 = (torch.deg2rad(torch.as_tensor(v, dtype=torch.float64)) for v in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    return 2 * radius_km * torch.asin(torch.sqrt(h.clamp(0.0, 1.0)))


def tcn_equirectangular_distance_km(lat_pred, lon_pred, lat_true, lon_true) -> torch.Tensor:
    """Track error of the official TropiCycloneNet evaluation (``trajectory_displacement_error``).

    111 km per degree, with the longitude difference scaled by the cosine of the observed latitude.
    """
    lat_pred, lon_pred, lat_true, lon_true = (torch.as_tensor(v, dtype=torch.float64) for v in (lat_pred, lon_pred, lat_true, lon_true))
    dx = (lon_true - lon_pred) * TCN_KM_PER_DEGREE * torch.cos(torch.deg2rad(lat_true))
    dy = (lat_true - lat_pred) * TCN_KM_PER_DEGREE
    return torch.sqrt(dx**2 + dy**2)


def _distance_fn(name: str):
    if name == "great_circle":
        return great_circle_distance_km
    if name == "tcn_equirectangular":
        return lambda lat_p, lon_p, lat_t, lon_t: tcn_equirectangular_distance_km(lat_p, lon_p, lat_t, lon_t)
    raise ValueError(f"unknown track_distance {name!r}; use 'great_circle' or 'tcn_equirectangular'")


def _spherical_mean(lat: torch.Tensor, lon: torch.Tensor, dim: int):
    """Mean position over ``dim`` computed from unit vectors (safe across the dateline)."""
    lat_r, lon_r = torch.deg2rad(lat), torch.deg2rad(lon)
    x = (torch.cos(lat_r) * torch.cos(lon_r)).mean(dim)
    y = (torch.cos(lat_r) * torch.sin(lon_r)).mean(dim)
    z = torch.sin(lat_r).mean(dim)
    mean_lat = torch.rad2deg(torch.atan2(z, torch.sqrt(x**2 + y**2)))
    mean_lon = torch.rad2deg(torch.atan2(y, x))
    return mean_lat, mean_lon


def _lead_label(hours: float) -> str:
    return f"{int(hours)}h" if float(hours).is_integer() else f"{hours:g}h".replace(".", "p")


def _nanmean(values: torch.Tensor) -> float:
    valid = ~torch.isnan(values)
    if not bool(valid.any()):
        return float("nan")
    return float(values[valid].mean())


def _as_dict(values: Any, variables: Sequence[str]) -> Dict[str, torch.Tensor]:
    if isinstance(values, Mapping):
        return {name: torch.as_tensor(values[name]).detach().cpu().double() for name in TRACK_VARIABLES if name in values}
    tensor = torch.as_tensor(values).detach().cpu().double()
    if tensor.size(-1) != len(variables):
        raise ValueError(
            f"tensor forecasts must end with one entry per target variable {list(variables)}, got shape {tuple(tensor.shape)}"
        )
    return {name: tensor[..., i] for i, name in enumerate(variables)}


def _call_model(model: nn.Module, inputs: Any) -> Any:
    forecast = getattr(model, "forecast", None)
    if callable(forecast):
        return forecast(inputs)
    return model(inputs)


def _batch_slice(inputs: Any, start: int, end: int, batch_dims: Mapping[str, int], key: Optional[str] = None) -> Any:
    if isinstance(inputs, Mapping):
        return {name: _batch_slice(value, start, end, batch_dims, name) for name, value in inputs.items()}
    if isinstance(inputs, torch.Tensor):
        return inputs.narrow(int(batch_dims.get(key, 0)), start, end - start)
    return inputs


def _concat_outputs(chunks: Sequence[Any]) -> Any:
    first = chunks[0]
    if isinstance(first, Mapping):
        return {name: _concat_outputs([chunk[name] for chunk in chunks]) for name in first}
    if isinstance(first, (tuple, list)):
        return type(first)(_concat_outputs([chunk[i] for chunk in chunks]) for i in range(len(first)))
    return torch.cat([torch.as_tensor(chunk).detach().cpu() for chunk in chunks], dim=0)


def _batched_model_outputs(model: nn.Module, inputs: Any, count: int, batch_size: Optional[int], batch_dims: Mapping[str, int]) -> Any:
    """Run the model on chunks of ``batch_size`` samples (all at once when ``batch_size`` is None).

    Mapping inputs are sliced along each entry's batch axis (``batch_dims``, default 0; e.g.
    TropiCycloneNet's ``obs_traj`` is time-first). Forecast dicts must be batch-first.
    """
    if not batch_size or batch_size >= count:
        return _call_model(model, inputs)
    chunks = [
        _call_model(model, _batch_slice(inputs, start, min(count, start + int(batch_size)), batch_dims))
        for start in range(0, count, int(batch_size))
    ]
    return _concat_outputs(chunks)


def track_intensity_metrics(
    forecast: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    lead_hours: Sequence[float],
    track_distance: str = "great_circle",
) -> Dict[str, float]:
    """Per-lead and aggregate track / intensity errors; see the module docstring."""
    distance = _distance_fn(track_distance)
    lead_hours = list(lead_hours)
    sample = forecast["lat"].ndim == 3
    if not sample and forecast["lat"].ndim != 2:
        raise ValueError(f"forecasts must be (storms, lead) or (storms, samples, lead), got {tuple(forecast['lat'].shape)}")
    if target["lat"].ndim != 2 or target["lat"].size(1) != len(lead_hours):
        raise ValueError(f"targets must be (storms, {len(lead_hours)} lead times), got {tuple(target['lat'].shape)}")
    metrics: Dict[str, float] = {}

    if sample:
        mean_lat, mean_lon = _spherical_mean(forecast["lat"], forecast["lon"], dim=1)
        point = {"lat": mean_lat, "lon": mean_lon}
        for name in ("wind", "pres"):
            if name in forecast:
                point[name] = forecast[name].mean(dim=1)
    else:
        point = dict(forecast)

    errors = {"track_error_km": distance(point["lat"], point["lon"], target["lat"], target["lon"])}
    for name, metric in (("wind", "intensity_mae"), ("pres", "pressure_mae")):
        if name in point and name in target:
            errors[metric] = (point[name] - target[name]).abs()
    if sample:
        lat_t, lon_t = target["lat"].unsqueeze(1), target["lon"].unsqueeze(1)
        per_sample = {"best_of_k_track_error_km": distance(forecast["lat"], forecast["lon"], lat_t, lon_t)}
        for name, metric in (("wind", "best_of_k_intensity_mae"), ("pres", "best_of_k_pressure_mae")):
            if name in forecast and name in target:
                per_sample[metric] = (forecast[name] - target[name].unsqueeze(1)).abs()
        for metric, values in per_sample.items():
            errors[metric] = values.min(dim=1).values

    for metric, values in errors.items():
        metrics[metric] = _nanmean(values)
        for step, hours in enumerate(lead_hours):
            metrics[f"{metric}_{_lead_label(hours)}"] = _nanmean(values[:, step])
    if sample:
        metrics["num_samples"] = float(forecast["lat"].size(1))
    return metrics


def intensity_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    lead_hours: Optional[Sequence[float]] = None,
    groups: Optional[Sequence[Any]] = None,
) -> Dict[str, float]:
    preds = torch.as_tensor(predictions).detach().cpu().double().reshape(-1)
    truth = torch.as_tensor(targets).detach().cpu().double().reshape(-1)
    if preds.shape != truth.shape:
        raise ValueError(f"expected one prediction per target, got {tuple(preds.shape)} and {tuple(truth.shape)}")
    valid = ~torch.isnan(truth)
    error = preds[valid] - truth[valid]
    metrics = {
        "intensity_mae": float(error.abs().mean()),
        "intensity_rmse": float(torch.sqrt((error**2).mean())),
    }
    if lead_hours and len(lead_hours) == 1:
        metrics[f"intensity_mae_{_lead_label(lead_hours[0])}"] = metrics["intensity_mae"]
    if groups is not None:
        labels = list(groups)
        if len(labels) != preds.numel():
            raise ValueError("split metadata 'groups' needs one label per sample")
        group_maes = []
        for label in sorted(set(labels), key=str):
            mask = torch.tensor([item == label for item in labels]) & valid
            if bool(mask.any()):
                mae = float((preds[mask] - truth[mask]).abs().mean())
                metrics[f"intensity_mae_{label}"] = mae
                group_maes.append(mae)
        if group_maes:
            metrics["intensity_mae_group_mean"] = sum(group_maes) / len(group_maes)
    return metrics


def _apply_transform(values: torch.Tensor, transform: Optional[Mapping[str, Any]]) -> torch.Tensor:
    if not transform:
        return values
    kind = transform.get("kind")
    if kind == "minmax":
        return values * (float(transform["max"]) - float(transform["min"])) + float(transform["min"])
    if kind == "affine":
        return values * float(transform["scale"]) + float(transform["offset"])
    raise ValueError(f"unknown prediction_transform kind {kind!r}")


class TropicalCycloneBenchmark(Benchmark):
    name = "tc"
    hazard_task = "tc.track_intensity"
    metric_names_by_task = {
        "tc.track_intensity": [
            "track_error_km",
            "intensity_mae",
            "pressure_mae",
            "best_of_k_track_error_km",
            "best_of_k_intensity_mae",
            "best_of_k_pressure_mae",
        ],
        "tc.intensity": ["intensity_mae", "intensity_rmse", "intensity_mae_group_mean"],
    }

    def evaluate(self, model: nn.Module, data: DataBundle, config: ExperimentConfig) -> BenchmarkResult:
        split = data.get_split(config.benchmark.eval_split)
        task = config.benchmark.hazard_task
        meta = data.metadata
        lead_hours = list(meta.get("lead_hours") or [])
        params = config.benchmark.params or {}
        count = len(split.targets)
        # Sampling models (TropiCycloneNet) draw noise: score with a fixed seed without touching the
        # caller's random state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(params.get("seed", config.seed)))
            outputs = _batched_model_outputs(model, split.inputs, count, params.get("batch_size"), meta.get("batch_dims") or {})

        if task == "tc.intensity":
            if isinstance(outputs, Mapping):
                outputs = outputs["intensity"]
            predictions = _apply_transform(torch.as_tensor(outputs).detach().cpu().double(), meta.get("prediction_transform"))
            metrics = intensity_metrics(predictions, split.targets, lead_hours, (split.metadata or {}).get("groups"))
        elif task == "tc.track_intensity":
            variables = list(meta.get("target_variables") or ("lat", "lon", "wind"))
            if not lead_hours:
                raise ValueError("tc.track_intensity data needs metadata['lead_hours']")
            target = _as_dict(split.targets, variables)
            forecast = _as_dict(outputs, variables)
            metrics = track_intensity_metrics(forecast, target, lead_hours, params.get("track_distance", "great_circle"))
        else:
            raise ValueError(f"TropicalCycloneBenchmark does not score hazard task {task!r}")

        data_units = meta.get("units") or {}
        units = {"intensity_mae": data_units.get("wind"), "intensity_rmse": data_units.get("wind")}
        if task == "tc.track_intensity":
            units.update({"track_error_km": "km", "pressure_mae": data_units.get("pres")})
        units = {name: unit for name, unit in units.items() if unit}
        return BenchmarkResult(
            benchmark_name=self.name,
            hazard_task=task,
            metrics={key: value for key, value in metrics.items() if not math.isnan(value)},
            metadata={
                "split": config.benchmark.eval_split,
                "dataset_name": meta.get("dataset"),
                "source_dataset": meta.get("source_dataset", meta.get("dataset")),
                "lead_hours": lead_hours,
                "units": units,
                "track_distance": params.get("track_distance", "great_circle") if task == "tc.track_intensity" else None,
                "synthetic": bool(meta.get("synthetic", False)),
            },
        )


register_benchmark(TropicalCycloneBenchmark.name, TropicalCycloneBenchmark)

__all__ = [
    "EARTH_RADIUS_KM",
    "TropicalCycloneBenchmark",
    "great_circle_distance_km",
    "intensity_metrics",
    "tcn_equirectangular_distance_km",
    "track_intensity_metrics",
]
