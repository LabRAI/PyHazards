"""What PyHazards offers for each global ML weather model whose cyclone skill is cited in Appendix A.

These models are not reimplemented in PyHazards and are not in the model registry: their cyclone
results come from running a tracker on their forecast fields and scoring the tracks against best
tracks. Each entry lists the paper, the official code and weights with their licences, how the paper
evaluated cyclones, and the PyHazards routes (published forecasts or tracks that can be read, runners,
trackers). Parameter counts were measured on the official code or checkpoints during the PyHazards
audit (``notes/inventory_other/inventory_tc.json`` in the project repository).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

__all__ = ["FOUNDATION_MODELS", "FoundationModel"]


@dataclass(frozen=True)
class FoundationModel:
    name: str
    paper: str
    paper_url: str
    code_url: str
    code_license: str
    weights: str
    weights_license: str
    parameters: str
    paper_tc_evaluation: str
    sources: Tuple[str, ...]
    runner: str
    tracker: str
    notes: Tuple[str, ...] = field(default_factory=tuple)


FOUNDATION_MODELS: Dict[str, FoundationModel] = {
    "fourcastnet": FoundationModel(
        name="FourCastNet",
        paper="Pathak et al., FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators, arXiv:2202.11214 (2022)",
        paper_url="https://arxiv.org/abs/2202.11214",
        code_url="https://github.com/NVlabs/FourCastNet",
        code_license="BSD-3-Clause",
        weights="NERSC FCN_weights_v0 (20 variables); earth2studio FCN = HF nvidia/fourcastnet1 (26-variable retrain)",
        weights_license="v0: not stated (do not redistribute); nvidia/fourcastnet1: Apache-2.0 (model card)",
        parameters="74,691,840 (20-channel AFNO backbone, official networks/afnonet.py)",
        paper_tc_evaluation="Qualitative only: Hurricane Michael (2018) ensemble, eye = MSLP minimum (Sec. 3.1).",
        sources=(
            "TCBench 2023 tracks and fields of FourCastNet v2 (SFNO small), a different model: pyhazards.forecasts.tcbench (model='fourcastnet_v2')",
        ),
        runner="earth2studio FCN (26-variable AFNO) or SFNO: pyhazards.forecasts.earth2studio_runner",
        tracker="TempestExtremes rules (TCBench) or the ECMWF-style follower; the paper used no aggregate tracker.",
        notes=("No public archive of FourCastNet (v1) forecasts was found; WeatherBench 2 does not host it.",),
    ),
    "graphcast": FoundationModel(
        name="GraphCast",
        paper="Lam et al., Learning skillful medium-range global weather forecasting, Science 382(6677):1416-1421 (2023), doi:10.1126/science.adi2336",
        paper_url="https://doi.org/10.1126/science.adi2336",
        code_url="https://github.com/google-deepmind/weathernext",
        code_license="Apache-2.0",
        weights="gs://dm_graphcast (GraphCast 0.25 deg / 37 levels, GraphCast_operational, GraphCast_small)",
        weights_license="CC BY 4.0 per the repository README since 2026-08-06; the checkpoint metadata still says CC BY-NC-SA 4.0",
        parameters="36,348,131 used by the model code (36,464,582 arrays in the checkpoint); the paper states 36.7M",
        paper_tc_evaluation=(
            "Supplement 8.1: modified ECMWF tracker (not released) on 06/18 UTC forecasts, IBTrACS 2018-2021, "
            "homogeneous sample with HRES (TIGGE), median and mean geodesic track error to 5 days; numbers only in figures."
        ),
        sources=(
            "WeatherBench 2 GraphCast forecasts (2018 from the 1979-2017 model, 2020; 00/12 UTC, ERA5 initial conditions): pyhazards.forecasts.weatherbench2",
        ),
        runner="earth2studio GraphCastOperational / GraphCastSmall (JAX): pyhazards.forecasts.earth2studio_runner",
        tracker="pyhazards.forecasts.following.GRAPHCAST_TRACKER (re-implemented from the supplement's description)",
        notes=("GenCast is a different model (Price et al., Nature 2025) and is not covered here.",),
    ),
    "pangu_weather": FoundationModel(
        name="Pangu-Weather",
        paper="Bi et al., Accurate medium-range global weather forecasting with 3D neural networks, Nature 619:533-538 (2023), doi:10.1038/s41586-023-06185-3; technical report arXiv:2211.02556",
        paper_url="https://doi.org/10.1038/s41586-023-06185-3",
        code_url="https://github.com/198808xc/Pangu-Weather",
        code_license="none (no LICENSE file; only ONNX inference scripts and pseudocode are released)",
        weights="pangu_weather_1/3/6/24.onnx (about 1.1 GB each)",
        weights_license="CC BY-NC-SA 4.0: commercial use forbidden",
        parameters="about 64.2M learnable per lead-time model (derived from the 24 h ONNX graph; ~256M for the four models)",
        paper_tc_evaluation=(
            "arXiv:2211.02556 Sec. 4.2.2: ECMWF-style tracker (MSLP minimum, 445 / 278 km rules) on deterministic "
            "forecasts of 88 named 2018 cyclones (TC2018); mean direct position error 120.29 km at 3 days and 195.65 km at 5 days "
            "(ECMWF-HRES 162.28 / 272.10 km)."
        ),
        sources=(
            "TCBench 2023 Pangu-Weather tracks and raw fields: pyhazards.forecasts.tcbench (model='pangu')",
            "WeatherBench 2 Pangu-Weather forecasts 2018-2022 (ERA5 and HRES initial conditions): pyhazards.forecasts.weatherbench2",
        ),
        runner="earth2studio Pangu6 / Pangu24 (official ONNX graphs): pyhazards.forecasts.earth2studio_runner",
        tracker="pyhazards.forecasts.following.PANGU_TRACKER (paper's rules) or the TempestExtremes rules (TCBench)",
    ),
}
