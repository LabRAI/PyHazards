#!/usr/bin/env python3

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataloader_v3 import GeoLoadInput, load_data
from dataloader.adapters import firms as firms_mod


def main() -> None:
    # Demo-speed mode: avoid loading very large FIRMS JSON archives.
    firms_mod.FIRMSAdapter.CSV_PATTERNS = ["firmsFL14-25/*.csv"]
    firms_mod.FIRMSAdapter.JSON_PATTERNS = []

    request = GeoLoadInput(
        root_dir="/home/yangshuang",
        data_sources=["FIRMS"],
        temporal_window=("2023-01-01", "2023-01-02"),
        area_of_interest_bbox=(-87.8, 24.0, -79.8, 31.5),
        spatial_resolution_deg=0.25,
        synthetic_time=True,
        temporal_cadence="D",
        target_hazards=["wildfire"],
    )
    sample = load_data(request)
    print("x shape:", sample.x.shape)
    print("y shape:", sample.y.shape)
    print("channels:", sample.meta.get("channels"))
    print("x synthetic ratio:", float(sample.meta["x_synthetic_mask"].mean()))
    print("y synthetic ratio:", float(sample.meta["y_synthetic_mask"].mean()))


if __name__ == "__main__":
    main()
