"""Tropical cyclone tracks from global ML weather models: forecast sources, trackers and scoring.

FourCastNet, GraphCast and Pangu-Weather are global weather models (36-75M parameters, 0.25-degree
global fields). Their cyclone results come from running a tracker on their forecast fields and
comparing the tracks with best tracks; they are not cyclone models and are not reimplemented here.
This package is the honest PyHazards integration of that pipeline:

forecast fields
    :mod:`.tcbench` reads TCBench's released Pangu-Weather / FourCastNet v2 / AIFS fields (slices over
    HTTP) and their tracks; :mod:`.weatherbench2` reads WeatherBench 2's GraphCast and Pangu-Weather
    forecasts; :mod:`.earth2studio_runner` runs the models through NVIDIA earth2studio, which the user
    installs (weights downloaded by earth2studio, under their own licences).
trackers
    :mod:`.tempest` - TempestExtremes DetectNodes / StitchNodes rules (TCBench's tracker), checked
    against the official binaries and TCBench's released tracks; :mod:`.following` - the ECMWF-style
    tracker described in the Pangu-Weather and GraphCast papers (no official code exists).
matching and scoring
    :mod:`.matching` assigns IBTrACS storm IDs (HuracanPy's rule, as in TCBench); :mod:`.scoring`
    reports great-circle track error (km) and intensity errors (kt, hPa) per lead time against IBTrACS,
    optionally exactly as TCBench's evaluation computes them.

See :data:`.catalog.FOUNDATION_MODELS` for licences, parameter counts and each paper's protocol.
Nothing here is a ``torch.nn.Module`` or appears in the model registry.
"""

from .catalog import FOUNDATION_MODELS, FoundationModel
from .fields import EARTH_RADIUS_KM, TRACKER_VARIABLES, great_circle_km, relative_vorticity, standardize_fields
from .following import (
    ECMWF_TRACKER,
    GRAPHCAST_TRACKER,
    PANGU_TRACKER,
    FollowingTrackerConfig,
    follow_cyclone,
    required_variables,
)
from .matching import match_tracks, matched_forecast_tracks
from .pipeline import follow_forecast_tracks, storms_at, tempest_forecast_tracks
from .scoring import KT_PER_MS, TCBENCH_KT_PER_MS, forecast_track_errors, score_forecast_tracks, tcbench_position_error_km
from .tcbench import (
    TCBENCH_MODELS,
    TCBENCH_REVISION,
    download_tcbench_file,
    read_tcbench_fields,
    read_tcbench_matched_tracks,
    read_tcbench_unmatched_tracks,
    tcbench_ibtracs,
    tcbench_path,
)
from .tempest import (
    TCBENCH_DETECT_NODES,
    TCBENCH_STITCH_NODES,
    ClosedContourCriterion,
    DetectNodesConfig,
    NodeOutput,
    StitchNodesConfig,
    StitchThreshold,
    detect_nodes,
    read_stitchnodes_csv,
    stitch_nodes,
    tempest_tracks,
    write_stitchnodes_csv,
)
from .weatherbench2 import WEATHERBENCH2_FORECASTS, read_weatherbench2_forecast
from .earth2studio_runner import EARTH2STUDIO_MODELS, load_earth2studio_model, run_earth2studio_forecast

__all__ = [
    "ClosedContourCriterion",
    "DetectNodesConfig",
    "EARTH2STUDIO_MODELS",
    "EARTH_RADIUS_KM",
    "ECMWF_TRACKER",
    "FOUNDATION_MODELS",
    "FollowingTrackerConfig",
    "FoundationModel",
    "GRAPHCAST_TRACKER",
    "KT_PER_MS",
    "NodeOutput",
    "PANGU_TRACKER",
    "StitchNodesConfig",
    "StitchThreshold",
    "TCBENCH_DETECT_NODES",
    "TCBENCH_KT_PER_MS",
    "TCBENCH_MODELS",
    "TCBENCH_REVISION",
    "TCBENCH_STITCH_NODES",
    "TRACKER_VARIABLES",
    "WEATHERBENCH2_FORECASTS",
    "detect_nodes",
    "download_tcbench_file",
    "follow_cyclone",
    "follow_forecast_tracks",
    "forecast_track_errors",
    "great_circle_km",
    "load_earth2studio_model",
    "match_tracks",
    "matched_forecast_tracks",
    "read_stitchnodes_csv",
    "read_tcbench_fields",
    "read_tcbench_matched_tracks",
    "read_tcbench_unmatched_tracks",
    "read_weatherbench2_forecast",
    "relative_vorticity",
    "required_variables",
    "run_earth2studio_forecast",
    "score_forecast_tracks",
    "standardize_fields",
    "stitch_nodes",
    "storms_at",
    "tcbench_ibtracs",
    "tcbench_path",
    "tcbench_position_error_km",
    "tempest_forecast_tracks",
    "tempest_tracks",
    "write_stitchnodes_csv",
]
