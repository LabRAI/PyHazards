from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from dataloader.schema import Sample
from dataloader_v2.simple import load_data as load_data_v2


def _normalize_source_name(source: str) -> str:
    mapping = {
        "firms": "firms",
        "firm": "firms",
        "noaa": "noaa",
        "mtbs": "mtbs",
        "mtbs_fod": "mtbs",
        "fod": "mtbs",
        "era5": "era5",
        "merra2": "merra2",
        "landfire": "landfire",
    }
    return mapping.get(str(source).strip().lower(), str(source).strip().lower())


def _infer_label_source(data_sources: Sequence[str]) -> Optional[str]:
    normalized = {_normalize_source_name(d) for d in data_sources}
    for candidate in ("firms", "noaa", "mtbs"):
        if candidate in normalized:
            return candidate
    return None


def _default_binary_mapping(target_hazards: Optional[List[str]]) -> Optional[Dict[str, int]]:
    if not target_hazards:
        return None
    return {str(h).strip().lower(): 1 for h in target_hazards}


@dataclass(init=False)
class GeoLoadInput:
    r"""The request input of :func:`dataloader_v3.load_data`.

    Args:
        data_sources (Sequence[str]): Input datasets to load and fuse.
        temporal_window (Tuple[str, str]): Query time window as
            ``(start_date, end_date)``.
        area_of_interest_bbox (Tuple[float, float, float, float]): AOI bounds
            as ``(min_lon, min_lat, max_lon, max_lat)``.
        spatial_resolution_deg (float, optional): Target grid resolution in
            degrees. (default: ``0.1``)
        root_dir (str, optional): Root folder of local datasets.
            (default: ``"/home/yangshuang"``)
        synthetic_time (bool, optional): Enable temporal harmonization and
            gap-filling. (default: ``False``)
        temporal_cadence (str, optional): Target temporal cadence, such as
            ``"D"``, ``"H"``, ``"15min"``. (default: ``"D"``)
        target_hazards (List[str], optional): Hazards to encode in ``y``.
            If provided without ``label_mapping``, defaults to binary encoding
            (hazard -> ``1``). (default: ``None``)
        label_source (str, optional): Label source override
            (``"firms"``, ``"noaa"``, ``"mtbs"``). If omitted, inferred from
            ``data_sources``. (default: ``None``)
        label_mapping (Dict[str, int], optional): Explicit hazard-to-id mapping.
            (default: ``None``)
    """

    data_sources: Sequence[str]
    temporal_window: Tuple[str, str]
    area_of_interest_bbox: Tuple[float, float, float, float]
    spatial_resolution_deg: float
    root_dir: str
    synthetic_time: bool
    temporal_cadence: str
    target_hazards: Optional[List[str]]
    label_source: Optional[str]
    label_mapping: Optional[Dict[str, int]]

    def __init__(
        self,
        data_sources: Sequence[str],
        temporal_window: Tuple[str, str],
        area_of_interest_bbox: Tuple[float, float, float, float],
        spatial_resolution_deg: float = 0.1,
        root_dir: str = "/home/yangshuang",
        synthetic_time: bool = False,
        temporal_cadence: str = "D",
        target_hazards: Optional[List[str]] = None,
        label_source: Optional[str] = None,
        label_mapping: Optional[Dict[str, int]] = None,
    ) -> None:
        self.data_sources = data_sources
        self.temporal_window = temporal_window
        self.area_of_interest_bbox = area_of_interest_bbox
        self.spatial_resolution_deg = spatial_resolution_deg
        self.root_dir = root_dir
        self.synthetic_time = synthetic_time
        self.temporal_cadence = temporal_cadence
        self.target_hazards = target_hazards
        self.label_source = label_source
        self.label_mapping = label_mapping


def load_data(request: GeoLoadInput) -> Sample:
    """Load a unified geospatial sample from a structured request object."""
    resolved_label_source = (
        str(request.label_source).strip().lower()
        if request.label_source is not None
        else _infer_label_source(request.data_sources)
    )
    resolved_mapping = request.label_mapping or _default_binary_mapping(request.target_hazards)

    return load_data_v2(
        data=request.data_sources,
        date_range=request.temporal_window,
        bbox=request.area_of_interest_bbox,
        resolution=request.spatial_resolution_deg,
        root_dir=request.root_dir,
        synthetic_time=request.synthetic_time,
        target_freq=request.temporal_cadence,
        label_source=resolved_label_source,
        label_hazards=request.target_hazards,
        label_mapping=resolved_mapping,
    )


def load_data_legacy(
    data: Sequence[str],
    date_range: Tuple[str, str],
    bbox: Tuple[float, float, float, float],
    resolution: float = 0.1,
    root_dir: str = "/home/yangshuang",
    synthetic_time: bool = False,
    target_freq: str = "D",
    label_hazards: Optional[List[str]] = None,
    label_source: Optional[str] = None,
    label_mapping: Optional[Dict[str, int]] = None,
) -> Sample:
    """Backward-compatible wrapper for users not yet migrated to GeoLoadInput."""
    req = GeoLoadInput(
        data_sources=data,
        temporal_window=date_range,
        area_of_interest_bbox=bbox,
        spatial_resolution_deg=resolution,
        root_dir=root_dir,
        synthetic_time=synthetic_time,
        temporal_cadence=target_freq,
        target_hazards=label_hazards,
        label_source=label_source,
        label_mapping=label_mapping,
    )
    return load_data(req)
