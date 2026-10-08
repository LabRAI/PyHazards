"""Inspect (and optionally download) the SmokeBench images: counts, frame sizes, smoke-area bins.

usage: python -m pyhazards.datasets.figlib_smokebench.inspection --root /path/to/figlib [--download]
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import List, Optional

from . import FIgLibSmokeBench, NUM_NEGATIVES, download_smokebench


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, help="directory with smokeynet_metadata.pkl and the FIgLib sequences")
    parser.add_argument("--download", action="store_true", help="download missing boxes / sequences first")
    parser.add_argument("--sequences", nargs="*", default=None, help="restrict to these FIgLib sequences")
    parser.add_argument("--negatives", type=int, default=NUM_NEGATIVES)
    parser.add_argument("--negative-seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.download:
        download_smokebench(args.root, args.sequences)
    data = FIgLibSmokeBench(args.root, sequences=args.sequences, negatives=args.negatives, negative_seed=args.negative_seed)
    from pyhazards.prompted.metrics import AREA_BIN_EDGES, AREA_BIN_LABELS, assign_bin, quantile_edges, smoke_area  # noqa: PLC0415

    for key, value in data.describe().items():
        print(f"{key:>18}: {value}")
    areas = [smoke_area(sample.boxes) for sample in data.positives]
    bins = Counter(AREA_BIN_LABELS[assign_bin(area, AREA_BIN_EDGES)] for area in areas)
    print("smoke images per SmokeBench area bin:", ", ".join(f"{label} {bins[label]}" for label in AREA_BIN_LABELS))
    if areas:
        print("area quintiles of these images:", [round(edge, 1) for edge in quantile_edges(areas)])
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        print("(install Pillow to list frame sizes)")
        return 0
    sizes = Counter()
    for sample in data:
        with Image.open(sample.path) as image:
            sizes[image.size] += 1
    print("frame sizes (W x H):", ", ".join(f"{w}x{h}: {n}" for (w, h), n in sizes.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
