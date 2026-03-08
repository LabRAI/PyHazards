from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class HazardTask:
    """Canonical hazard task label used by benchmark and config layers."""

    name: str
    hazard: str
    target: str
    description: str


_HAZARD_TASKS: Dict[str, HazardTask] = {
    "earthquake.picking": HazardTask(
        name="earthquake.picking",
        hazard="earthquake",
        target="picking",
        description="Waveform-based earthquake phase detection and P/S picking.",
    ),
    "earthquake.forecasting": HazardTask(
        name="earthquake.forecasting",
        hazard="earthquake",
        target="forecasting",
        description="Earthquake forecasting over spatial or temporal forecast windows.",
    ),
    "wildfire.danger": HazardTask(
        name="wildfire.danger",
        hazard="wildfire",
        target="danger",
        description="Wildfire danger or risk prediction over a region and horizon.",
    ),
    "wildfire.spread": HazardTask(
        name="wildfire.spread",
        hazard="wildfire",
        target="spread",
        description="Wildfire spread forecasting over raster masks or burned-area grids.",
    ),
    "flood.streamflow": HazardTask(
        name="flood.streamflow",
        hazard="flood",
        target="streamflow",
        description="Riverine discharge or streamflow forecasting.",
    ),
    "flood.inundation": HazardTask(
        name="flood.inundation",
        hazard="flood",
        target="inundation",
        description="Flood inundation and water-extent forecasting over spatial grids.",
    ),
    "tc.track_intensity": HazardTask(
        name="tc.track_intensity",
        hazard="tc",
        target="track_intensity",
        description="Storm-track and intensity forecasting over lead-time horizons.",
    ),
}


def available_hazard_tasks() -> List[str]:
    return sorted(_HAZARD_TASKS.keys())


def get_hazard_task(name: str) -> HazardTask:
    key = name.strip().lower()
    if key not in _HAZARD_TASKS:
        raise KeyError(
            "Unknown hazard task '{name}'. Known: {known}".format(
                name=name,
                known=", ".join(available_hazard_tasks()),
            )
        )
    return _HAZARD_TASKS[key]


def has_hazard_task(name: str) -> bool:
    return name.strip().lower() in _HAZARD_TASKS


__all__ = [
    "HazardTask",
    "available_hazard_tasks",
    "get_hazard_task",
    "has_hazard_task",
]
