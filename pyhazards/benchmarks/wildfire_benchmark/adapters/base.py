from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..artifacts import AdapterRunOutput, STEP_LABEL, build_experiment_setting_from_run_output


class WildfireSmokeAdapter(ABC):
    """Minimal adapter contract for my-copy wildfire benchmark smoke runs."""

    def __init__(self, model_spec: Dict[str, Any], contract: Dict[str, Any], step_limits: Dict[str, int]):
        self.model_spec = model_spec
        self.contract = contract
        self.step_limits = step_limits

    @property
    def model_name(self) -> str:
        return str(self.model_spec["name"])

    @property
    def train_unit(self) -> str:
        return str(self.model_spec["train_unit"])

    @property
    def step_name(self) -> str:
        return STEP_LABEL[self.train_unit]

    def resolve_num_steps(self) -> int:
        defaults = self.model_spec.get("defaults", {})
        if self.train_unit == "epoch":
            return max(5, min(int(defaults.get("max_epochs", self.step_limits["epoch"])), self.step_limits["epoch"]))
        if self.train_unit == "round":
            return max(10, min(int(defaults.get("num_boost_round", self.step_limits["round"])), self.step_limits["round"]))
        if self.train_unit == "iteration":
            return max(10, min(int(defaults.get("max_iter", self.step_limits["iteration"])), self.step_limits["iteration"]))
        if self.train_unit == "tree":
            return max(10, min(int(defaults.get("n_estimators", self.step_limits["tree"])), self.step_limits["tree"]))
        raise ValueError(f"Unsupported train_unit={self.train_unit}")

    @abstractmethod
    def run(self, seed: int) -> AdapterRunOutput:
        """Run one smoke seed and return standardized benchmark artifacts."""

    def build_experiment_setting(self, seed: int, run_output: AdapterRunOutput) -> Dict[str, Any]:
        return build_experiment_setting_from_run_output(
            contract=self.contract,
            model_spec=self.model_spec,
            seed=int(seed),
            run_output=run_output,
        )


def stable_seed_offset(model_name: str) -> int:
    digest = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return values[:]
    out: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        out.append(float(sum(chunk) / len(chunk)))
    return out


def find_converged_step(history: List[Dict[str, float]], train_unit: str, smooth_window: int, patience: int, min_improvement: float) -> int:
    step_key = STEP_LABEL[train_unit]
    val_loss = [float(row["val_loss"]) for row in history]
    smoothed = moving_average(val_loss, smooth_window)

    stable = 0
    for idx in range(1, len(smoothed)):
        improvement = smoothed[idx - 1] - smoothed[idx]
        if improvement < min_improvement:
            stable += 1
        else:
            stable = 0
        if stable >= patience:
            return int(history[idx][step_key])
    return int(history[-1][step_key])


def normalized_consistency_score(mean_day_to_day_change: float) -> float:
    return max(0.0, min(1.0, 1.0 - float(mean_day_to_day_change)))
