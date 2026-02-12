from __future__ import annotations

import importlib.util
import subprocess
import sys

import torch


DATASET_MODULES = [
    "pyhazards.datasets.merra2.inspection",
    "pyhazards.datasets.era5.inspection",
    "pyhazards.datasets.noaa_flood.inspection",
    "pyhazards.datasets.firms.inspection",
    "pyhazards.datasets.mtbs.inspection",
    "pyhazards.datasets.landfire.inspection",
    "pyhazards.datasets.wfigs.inspection",
    "pyhazards.datasets.goesr.inspection",
]


def verify_datasets() -> bool:
    ok = True
    print("=== Dataset Table Verification ===")
    for mod in DATASET_MODULES:
        spec = importlib.util.find_spec(mod)
        print(f"[import] {mod}: {'OK' if spec else 'MISSING'}")
        if spec is None:
            ok = False
            continue

        cmd = [sys.executable, "-m", mod, "--help"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(f"[cli]    {mod} --help: exit={res.returncode}")
        if res.returncode != 0:
            ok = False

    era5_cmd = [
        sys.executable,
        "-m",
        "pyhazards.datasets.era5.inspection",
        "--path",
        "pyhazards/data/era5_subset",
        "--max-vars",
        "5",
    ]
    era5_res = subprocess.run(era5_cmd, capture_output=True, text=True)
    print(
        "[run]    pyhazards.datasets.era5.inspection "
        f"--path pyhazards/data/era5_subset: exit={era5_res.returncode}"
    )
    if era5_res.returncode != 0:
        ok = False

    return ok


def verify_models() -> bool:
    ok = True
    print("\n=== Model Table Verification ===")
    from pyhazards.models import build_model

    wildfire = build_model(name="wildfire_aspp", task="segmentation", in_channels=12)
    x = torch.randn(2, 12, 16, 16)
    y = wildfire(x)
    print(f"[model] wildfire_aspp forward shape={tuple(y.shape)}")
    if tuple(y.shape) != (2, 1, 16, 16):
        ok = False

    hydrographnet = build_model(
        name="hydrographnet",
        task="regression",
        node_in_dim=2,
        edge_in_dim=3,
        out_dim=1,
    )
    batch = {
        "x": torch.randn(1, 3, 6, 2),
        "adj": torch.eye(6).unsqueeze(0),
        "coords": torch.randn(6, 2),
    }
    out = hydrographnet(batch)
    print(f"[model] hydrographnet forward shape={tuple(out.shape)}")
    if tuple(out.shape) != (1, 6, 1):
        ok = False

    return ok


def main() -> int:
    datasets_ok = verify_datasets()
    models_ok = verify_models()
    ok = datasets_ok and models_ok
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

