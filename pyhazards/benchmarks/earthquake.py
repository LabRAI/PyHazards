"""Earthquake benchmark: seismic phase picking and dense-grid wavefield forecasting.

``earthquake.picking`` scores phase pickers the way the original papers do (see
:mod:`pyhazards.metrics.picking`):

1. waveforms ``(n, 3, samples)`` are reordered from the dataset's component order to the model's
   ``component_order`` (the model's ``sampling_rate`` must match the dataset's);
2. ``model.annotate(waveforms)`` (official per-window normalisation + forward) gives probabilities,
   ``model.extract_picks(probabilities)`` the official picks of each model (PhaseNet: peaks above 0.5,
   EQTransformer: its detection/picking post-processing, GPD: sliding-window triggers). Models without
   ``annotate`` are called directly: a ``(n, C, samples)`` output is read as per-sample probabilities
   with P in channel 1 and S in channel 2 (or ``model.phase_channels``) and picked with peaks above
   ``threshold`` at least ``min_distance_s`` apart; a ``(n, 2)`` output is read as one regressed
   (P, S) arrival sample per trace;
3. picks are matched to the dataset's arrivals (targets ``(n, 2)``, NaN = none) per phase:
   precision, recall and F1 with a tolerance window, and residual mean / standard deviation / MAE in
   seconds, under ``protocol="phasenet"`` (Zhu & Beroza 2019: TP if ``|dt| < 0.1 s``, residual
   statistics over ``|dt| < 0.5 s``) or ``protocol="eqtransformer"`` (Mousavi et al. 2020: TP if
   ``|dt| < 0.5 s``). ``tolerance_s`` / ``residual_window_s`` override the protocol;
4. models with ``extract_detections`` (EQTransformer) also get trace-level detection precision,
   recall and F1 (a trace is an event when it has a P or S arrival).

Benchmark ``params``: ``protocol``, ``tolerance_s``, ``residual_window_s``, ``batch_size`` (default
32), ``pick_params`` (passed to ``extract_picks``), ``detection_params`` (passed to
``extract_detections``), and ``threshold`` / ``min_distance_s`` for models without ``extract_picks``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

from ..configs import ExperimentConfig
from ..datasets.base import DataBundle
from ..metrics.picking import PICKING_PROTOCOLS, detection_scores, peak_picks, score_picks
from .base import Benchmark
from .registry import register_benchmark
from .schemas import BenchmarkResult

PICKING_METRICS = [
    "p_precision",
    "p_recall",
    "p_f1",
    "p_residual_mean",
    "p_residual_std",
    "p_pick_mae",
    "s_precision",
    "s_recall",
    "s_f1",
    "s_residual_mean",
    "s_residual_std",
    "s_pick_mae",
    "detection_precision",
    "detection_recall",
    "detection_f1",
]


def _reorder(waveforms: torch.Tensor, source: Any, target: Any) -> torch.Tensor:
    if not source or not target or str(source).upper() == str(target).upper():
        return waveforms
    from ..models._seismic import reorder_components

    return reorder_components(waveforms, str(source), str(target))


def _generic_picks(output: torch.Tensor, model: nn.Module, params: Dict[str, Any], sampling_rate: float):
    if output.ndim == 2 and output.shape[1] == 2:
        # Regressed (P, S) arrival samples: one pick of each phase per trace.
        return [{"P": [(float(p), math.nan)], "S": [(float(s), math.nan)]} for p, s in output.tolist()]
    if output.ndim != 3:
        raise ValueError(
            "earthquake.picking expects models to return per-sample probabilities (batch, classes, samples) "
            f"or regressed arrival samples (batch, 2); got shape {tuple(output.shape)}."
        )
    channels = getattr(model, "phase_channels", {"P": 1, "S": 2})
    threshold = float(params.get("threshold", 0.5))
    distance = int(round(float(params.get("min_distance_s", 0.5)) * sampling_rate))
    values = output.detach().cpu().double().numpy()
    return [
        {phase: peak_picks(trace[channel], threshold, distance) for phase, channel in channels.items()}
        for trace in values
    ]


class EarthquakeBenchmark(Benchmark):
    name = "earthquake"
    hazard_task = "earthquake.picking"
    metric_names_by_task = {
        "earthquake.picking": PICKING_METRICS,
        "earthquake.forecasting": ["mae", "mse"],
    }

    def evaluate(self, model: nn.Module, data: DataBundle, config: ExperimentConfig) -> BenchmarkResult:
        if config.benchmark.hazard_task == "earthquake.picking":
            return self._evaluate_picking(model, data, config)
        split = data.get_split(config.benchmark.eval_split)
        preds = model(split.inputs)
        y = split.targets
        metrics = {
            "mae": float(torch.mean(torch.abs(preds - y)).detach().cpu()),
            "mse": float(torch.mean((preds - y) ** 2).detach().cpu()),
        }
        return BenchmarkResult(
            benchmark_name=self.name,
            hazard_task=config.benchmark.hazard_task,
            metrics=metrics,
            metadata={
                "split": config.benchmark.eval_split,
                "dataset_name": data.metadata.get("dataset"),
                "source_dataset": data.metadata.get("source_dataset", data.metadata.get("dataset")),
            },
        )

    def _evaluate_picking(self, model: nn.Module, data: DataBundle, config: ExperimentConfig) -> BenchmarkResult:
        params = dict(config.benchmark.params)
        protocol = str(params.get("protocol", "phasenet")).lower()
        if protocol not in PICKING_PROTOCOLS:
            raise ValueError(f"Unknown picking protocol {protocol!r}; expected one of {sorted(PICKING_PROTOCOLS)}.")
        tolerance_s = float(params.get("tolerance_s", PICKING_PROTOCOLS[protocol]["tolerance_s"]))
        residual_window_s = float(params.get("residual_window_s", PICKING_PROTOCOLS[protocol]["residual_window_s"]))
        batch_size = int(params.get("batch_size", 32))
        pick_params = dict(params.get("pick_params", {}))
        detection_params = dict(params.get("detection_params", {}))

        split = data.get_split(config.benchmark.eval_split)
        waveforms, targets = split.inputs, split.targets
        if waveforms.ndim != 3 or targets.ndim != 2 or targets.shape[1] != 2 or len(waveforms) != len(targets):
            raise ValueError(
                "earthquake.picking needs inputs shaped (n, channels, samples) and targets shaped (n, 2) "
                f"[P, S arrival samples, NaN = none]; got {tuple(waveforms.shape)} and {tuple(targets.shape)}."
            )
        data_rate = data.metadata.get("sampling_rate") or data.feature_spec.extra.get("sampling_rate")
        model_rate = getattr(model, "sampling_rate", None)
        if data_rate and model_rate and not math.isclose(float(data_rate), float(model_rate)):
            raise ValueError(
                f"The model expects {model_rate} Hz waveforms but the dataset is sampled at {data_rate} Hz."
            )
        sampling_rate = float(data_rate or model_rate or 100.0)
        source_order = data.metadata.get("component_order") or data.feature_spec.extra.get("component_order")
        waveforms = _reorder(waveforms, source_order, getattr(model, "component_order", None))

        picks: List[Dict[str, list]] = []
        detections: List[list] = []
        has_detections = hasattr(model, "extract_detections")
        for start in range(0, len(waveforms), batch_size):
            batch = waveforms[start : start + batch_size]
            if hasattr(model, "annotate"):
                output = model.annotate(batch)
            else:
                output = model(batch)
            if hasattr(model, "extract_picks"):
                picks.extend(model.extract_picks(output, **pick_params))
            else:
                picks.extend(_generic_picks(output, model, params, sampling_rate))
            if has_detections:
                detections.extend(model.extract_detections(output, **detection_params))

        tolerance, window = tolerance_s * sampling_rate, residual_window_s * sampling_rate
        metrics: Dict[str, float] = {}
        counts: Dict[str, Dict[str, int]] = {}
        manual = targets.detach().cpu().double().tolist()
        for column, phase in enumerate(("P", "S")):
            predicted = [[sample for sample, _ in trace_picks.get(phase, [])] for trace_picks in picks]
            score = score_picks(predicted, [[row[column]] for row in manual], tolerance, window)
            values = score.metrics(sampling_rate)
            prefix = phase.lower()
            metrics.update(
                {
                    f"{prefix}_precision": values["precision"],
                    f"{prefix}_recall": values["recall"],
                    f"{prefix}_f1": values["f1"],
                    f"{prefix}_residual_mean": values["residual_mean"],
                    f"{prefix}_residual_std": values["residual_std"],
                    f"{prefix}_pick_mae": values["mae"],
                }
            )
            counts[phase] = {"manual": score.n_true, "predicted": score.n_pred, "true_positive": score.n_tp}
        if has_detections:
            is_event = [not (math.isnan(p) and math.isnan(s)) for p, s in manual]
            scores = detection_scores([len(windows) > 0 for windows in detections], is_event)
            metrics.update({f"detection_{key}": value for key, value in scores.items()})

        return BenchmarkResult(
            benchmark_name=self.name,
            hazard_task=config.benchmark.hazard_task,
            metrics=metrics,
            metadata={
                "split": config.benchmark.eval_split,
                "protocol": protocol,
                "tolerance_s": tolerance_s,
                "residual_window_s": residual_window_s,
                "sampling_rate": sampling_rate,
                "pick_counts": counts,
                "n_traces": len(waveforms),
                "dataset_name": data.metadata.get("dataset"),
                "source_dataset": data.metadata.get("source_dataset", data.metadata.get("dataset")),
                "synthetic_data": bool(data.metadata.get("synthetic", False)),
            },
        )

    def export_report(self, result: BenchmarkResult, output_dir: str, formats) -> Dict[str, str]:
        paths = super().export_report(result, output_dir=output_dir, formats=formats)
        if result.hazard_task == "earthquake.picking":
            target = Path(output_dir)
            target.mkdir(parents=True, exist_ok=True)
            path = target / "earthquake_pick_counts.json"
            path.write_text(json.dumps(result.metadata.get("pick_counts", {}), indent=2, sort_keys=True), encoding="utf-8")
            paths["pick_counts"] = str(path)
        return paths


register_benchmark(EarthquakeBenchmark.name, EarthquakeBenchmark)

__all__ = ["EarthquakeBenchmark", "PICKING_METRICS"]
