# Dataloader V3

Minimal API with a structured request object.

## API
```python
from dataloader_v3 import (
    GeoLoadInput,
    load_data,
    save_sample_h5,
    load_sample_h5,
    to_torch_batch,
)
```

## `GeoLoadInput` (required format)
- `data_sources: list[str]`  
  Example: `["FIRMS", "ERA5"]`
- `temporal_window: tuple[str, str]`  
  Format: `("YYYY-MM-DD", "YYYY-MM-DD")` or `("YYYY-MM-DD HH:MM:SS", "...")`
- `area_of_interest_bbox: tuple[float, float, float, float]`  
  Order: `(min_lon, min_lat, max_lon, max_lat)`

## Optional
- `spatial_resolution_deg: float = 0.1`
- `root_dir: str = "/home/yangshuang"`
- `synthetic_time: bool = False`
- `temporal_cadence: str = "D"` (`"D"`, `"H"`, `"15min"`, ...)
- `target_hazards: list[str] | None = None` (if set and no mapping, default code is `1`)
- `label_source: str | None = None` (`"firms" | "noaa" | "mtbs"`, else auto-infer)
- `label_mapping: dict[str, int] | None = None`

## Example
```python
from dataloader_v3 import GeoLoadInput, load_data, save_sample_h5

req = GeoLoadInput(
    data_sources=["FIRMS"],
    temporal_window=("2023-01-01", "2023-01-02"),
    area_of_interest_bbox=(-87.8, 24.0, -79.8, 31.5),
    spatial_resolution_deg=0.25,
    synthetic_time=True,
    temporal_cadence="D",
    target_hazards=["wildfire"],
)
sample = load_data(req)
save_sample_h5(sample, "/home/yangshuang/output/sample_v3.h5")
sample2 = load_sample_h5("/home/yangshuang/output/sample_v3.h5")
x_t, y_t, meta = to_torch_batch(sample2)
```
