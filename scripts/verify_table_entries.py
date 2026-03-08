from __future__ import annotations

import importlib.util
import subprocess
import sys

from pyhazards.model_catalog import load_model_cards, run_smoke_test


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
    for card in load_model_cards():
        result = run_smoke_test(card)
        print(
            f"[model] {card.model_name}: expected={result['expected_shape']} "
            f"actual={result['actual_shape']}"
        )
        if not result["ok"]:
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
