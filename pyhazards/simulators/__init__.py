"""Adapters to external, physics-based fire simulators.

These are not neural networks and are not in the model registry. Each adapter drives the official
software, which the user installs separately, or reads its outputs; nothing of the simulator is
reimplemented in PyHazards.

- :class:`ForeFireSimulator` runs ForeFire (GPL-3.0, optional ``pip install forefire``) on rasters.
- WRF-SFIRE outputs (``wrfout`` files) are read by :mod:`pyhazards.datasets.wrf_sfire`.
"""

from .forefire import (
    FOREFIRE_CHECKED_VERSION,
    ForeFireNotInstalledError,
    ForeFireResult,
    ForeFireSimulator,
    forefire_available,
    forefire_version,
)

__all__ = [
    "FOREFIRE_CHECKED_VERSION",
    "ForeFireNotInstalledError",
    "ForeFireResult",
    "ForeFireSimulator",
    "forefire_available",
    "forefire_version",
]
