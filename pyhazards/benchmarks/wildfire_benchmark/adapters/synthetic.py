from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

from pyhazards.models import available_models

from ..artifacts import AdapterRunOutput
from .base import (
    WildfireSmokeAdapter,
    find_converged_step,
    normalized_consistency_score,
    stable_seed_offset,
)

MODEL_NAME_ALIASES = {
    "wrf_sfire_adapter": "wrf_sfire",
    "forefire_adapter": "forefire",
}


def resolve_local_model_name(model_name: str) -> str:
    return MODEL_NAME_ALIASES.get(model_name, model_name)


def summarize_metrics(best_val_loss: float, seed: int, model_name: str) -> Dict[str, float]:
    seed_offset = stable_seed_offset(model_name) // 17
    rng = np.random.default_rng(seed + seed_offset)

    quality = float(np.clip(1.0 / (1.0 + best_val_loss), 0.0, 1.0))
    auprc = float(np.clip(0.03 + 0.5 * quality + rng.normal(0.0, 0.015), 0.0, 1.0))
    auroc = float(np.clip(0.55 + 0.42 * quality + rng.normal(0.0, 0.01), 0.0, 1.0))
    brier = float(np.clip(0.42 - 0.28 * quality + rng.normal(0.0, 0.01), 0.0, 1.0))
    nll = float(np.clip(1.1 - 0.7 * quality + rng.normal(0.0, 0.03), 0.01, 5.0))
    ece = float(np.clip(0.22 - 0.14 * quality + rng.normal(0.0, 0.008), 0.0, 1.0))
    temporal_delta = float(np.clip(0.25 - 0.12 * quality + rng.normal(0.0, 0.01), 0.0, 1.0))

    return {
        "auprc": auprc,
        "auroc": auroc,
        "brier": brier,
        "nll": nll,
        "ece": ece,
        "mean_day_to_day_change": temporal_delta,
        "normalized_consistency_score": normalized_consistency_score(temporal_delta),
    }


class SyntheticWildfireModelAdapter(WildfireSmokeAdapter):
    """Smoke adapter for migrated my-copy wildfire benchmark models."""

    def _simulate_history(self, seed: int, num_steps: int) -> List[Dict[str, float]]:
        step_key = self.step_name
        defaults = self.model_spec.get("defaults", {})

        seed_offset = stable_seed_offset(self.model_name)
        rng = np.random.default_rng(seed + seed_offset)

        base_lr = float(defaults.get("lr", defaults.get("learning_rate", defaults.get("eta", 1e-3))))
        start_loss = float(rng.uniform(0.8, 1.6))
        floor_loss = float(rng.uniform(0.08, 0.25))
        speed = float(rng.uniform(0.02, 0.08))

        history: List[Dict[str, float]] = []
        for step in range(1, num_steps + 1):
            decay = floor_loss + (start_loss - floor_loss) * math.exp(-speed * step)
            noise = float(rng.normal(0.0, 0.01))
            train_loss = max(0.01, decay + noise)

            gap = float(rng.uniform(0.02, 0.09))
            val_noise = float(rng.normal(0.0, 0.008))
            val_loss = max(0.01, train_loss + gap + val_noise)

            cosine = 0.5 * (1.0 + math.cos(math.pi * (step - 1) / max(1, num_steps - 1)))
            learning_rate = base_lr * cosine

            history.append(
                {
                    step_key: float(step),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "learning_rate": float(learning_rate),
                }
            )
        return history

    def _resolve_model_metadata(self) -> Dict[str, object]:
        local_name = resolve_local_model_name(self.model_name)
        registered = local_name in set(available_models())
        source_path = None
        try:
            module = importlib.import_module(f"pyhazards.models.{local_name}")
            source_path = str(Path(module.__file__).resolve()) if getattr(module, "__file__", None) else None
        except Exception:
            source_path = None
        return {
            "canonical_model_name": local_name,
            "registered_in_my_copy": registered,
            "model_source": source_path,
        }

    def run(self, seed: int) -> AdapterRunOutput:
        num_steps = self.resolve_num_steps()
        history = self._simulate_history(seed=seed, num_steps=num_steps)

        val_loss = [float(item["val_loss"]) for item in history]
        best_idx = int(np.argmin(np.asarray(val_loss)))
        best_step = int(history[best_idx][self.step_name])

        conv_cfg = self.contract["shared_training"]["convergence_rule"]
        converged_step = find_converged_step(
            history=history,
            train_unit=self.train_unit,
            smooth_window=int(conv_cfg["smoothing_window"]),
            patience=int(conv_cfg["patience"]),
            min_improvement=float(conv_cfg["min_improvement"]),
        )

        metrics = summarize_metrics(best_val_loss=float(val_loss[best_idx]), seed=seed, model_name=self.model_name)
        model_meta = self._resolve_model_metadata()

        return AdapterRunOutput(
            history=history,
            metrics=metrics,
            best_step=best_step,
            converged_step=converged_step,
            train_unit=self.train_unit,
            notes={
                "adapter_kind": "synthetic_my_copy_benchmark",
                "status": "smoke_only",
                "message": "Synthetic smoke run executed inside my-copy wildfire benchmark skeleton.",
                **model_meta,
            },
        )
