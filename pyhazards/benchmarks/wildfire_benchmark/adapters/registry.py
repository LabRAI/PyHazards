from __future__ import annotations

from typing import Any, Dict

from .base import WildfireSmokeAdapter
from .synthetic import SyntheticWildfireModelAdapter


SMOKE_ADAPTERS = {}


def create_adapter(model_spec: Dict[str, Any], contract: Dict[str, Any], step_limits: Dict[str, int]) -> WildfireSmokeAdapter:
    adapter_cls = SMOKE_ADAPTERS.get(str(model_spec["name"]), SyntheticWildfireModelAdapter)
    return adapter_cls(model_spec=model_spec, contract=contract, step_limits=step_limits)
