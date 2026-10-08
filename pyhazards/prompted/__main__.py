"""Command line: run one SmokeBench task with one backend.

usage: python -m pyhazards.prompted --backend qwen2_5_vl_7b --task classification \\
           --root data/figlib --output runs/qwen7b_cls [--download] [--sequences A B ...]
"""

from __future__ import annotations

import argparse
import json
from typing import List, Optional

from .backends import available_backends, build_backend
from .geometry import PROTOCOLS
from .prompts import TASKS
from .runner import evaluate_smokebench


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pyhazards.prompted", description=__doc__.splitlines()[0])
    parser.add_argument("--backend", required=True, choices=available_backends())
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--root", required=True, help="FIgLib SmokeBench directory (see figlib_smokebench)")
    parser.add_argument("--output", required=True, help="directory for records.jsonl and report.json")
    parser.add_argument("--protocol", default="paper", choices=PROTOCOLS)
    parser.add_argument("--download", action="store_true", help="download missing boxes and frames first")
    parser.add_argument("--sequences", nargs="*", default=None, help="restrict to these FIgLib sequences")
    parser.add_argument("--negatives", type=int, default=1000, help="smoke-free frames to draw (classification)")
    parser.add_argument("--negative-seed", type=int, default=0)
    parser.add_argument("--model-id", default=None, help="override the preset's model id")
    parser.add_argument("--greedy", action="store_true", help="greedy decoding instead of temperature 0.5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    from ..datasets.figlib_smokebench import FIgLibSmokeBench  # noqa: PLC0415

    dataset = FIgLibSmokeBench(
        args.root,
        sequences=args.sequences,
        negatives=args.negatives,
        negative_seed=args.negative_seed,
        download=args.download,
    )
    kwargs = {}
    if args.model_id:
        kwargs["model_id"] = args.model_id
    if args.greedy:
        if args.backend in {"gemini_2_5_pro", "gpt_4o"}:
            kwargs["temperature"] = 0.0
        else:
            kwargs["do_sample"] = False
    backend = build_backend(args.backend, **kwargs)
    report = evaluate_smokebench(
        backend,
        dataset,
        args.task,
        protocol=args.protocol,
        output_dir=args.output,
        resume=not args.no_resume,
        seed=args.seed,
        limit=args.limit,
        progress=True,
    )
    print(json.dumps({"metrics": report["metrics"], "paper_reported": report["paper_reported"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
