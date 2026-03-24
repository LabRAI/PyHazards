from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

WILDFIRE_BENCHMARK_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs" / "wildfire_benchmark"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_contract(path: str | Path | None = None) -> Dict[str, Any]:
    target = Path(path) if path else WILDFIRE_BENCHMARK_CONFIG_ROOT / "track_o_2024_v1.json"
    return load_json(target)


def load_model_catalog(kind: str = "main", path: str | Path | None = None) -> List[Dict[str, Any]]:
    if path is not None:
        return load_json(Path(path))
    filename = "model_catalog_22.json" if kind == "main" else "model_catalog_extensions_v1.json"
    return load_json(WILDFIRE_BENCHMARK_CONFIG_ROOT / filename)


def parse_seed_list(seed_text: str | List[int] | None) -> List[int]:
    if seed_text is None:
        return [42]
    if isinstance(seed_text, list):
        return [int(x) for x in seed_text] or [42]
    seeds = [int(s.strip()) for s in str(seed_text).split(",") if s.strip()]
    return seeds or [42]


def select_models(
    all_models: List[Dict[str, Any]],
    *,
    source_tier: str = "all",
    models: str | List[str] | None = None,
    limit_models: int = 0,
) -> List[Dict[str, Any]]:
    selected = list(all_models)
    if source_tier != "all":
        selected = [m for m in selected if m.get("source_tier") == source_tier]

    if models:
        allowed = set(models) if isinstance(models, list) else {x.strip() for x in str(models).split(",") if x.strip()}
        selected = [m for m in selected if m["name"] in allowed]

    selected = sorted(selected, key=lambda x: int(x.get("priority", 9999)))
    if limit_models > 0:
        selected = selected[:limit_models]
    return selected
