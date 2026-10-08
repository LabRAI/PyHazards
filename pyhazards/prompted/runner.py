"""Evaluate a prompted backend on SmokeBench and write a report.

One call runs one task: ``classification`` on every image (smoke and smoke-free); ``tile``,
``grid`` and ``detection`` on the smoke images only, as in the paper (their scores need a ground-truth
box, and the area / contrast bins are defined on boxes). Each image's answer(s), parsed prediction
and score are appended to ``records.jsonl`` as soon as they exist, so an interrupted run resumes
where it stopped (``resume=True``); ``report.json`` holds the aggregate metrics, the per-bin
breakdowns of the paper's tables, answer-format diagnostics, the exact prompt, the backend settings
and, when the backend is one of the SmokeBench models, the numbers the paper reports.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from .backends.base import PromptedBackend, check_image
from .geometry import PROTOCOLS, covered_cells, crop_tiles, draw_grid_overlay
from .metrics import (
    AREA_BIN_EDGES,
    AREA_BIN_LABELS,
    CONTRAST_BIN_EDGES,
    CONTRAST_BIN_LABELS,
    binned_means,
    detection_iou,
    quantile_edges,
    set_iou,
    smoke_area,
    weber_contrast,
)
from .parsers import EMPTY_BOX, answer_kind, parse_classification, parse_detection, parse_grid, parse_tile
from .prompts import GRID_LAYOUT, TASKS, TILE_LAYOUT, get_prompt

PathLike = Union[str, Path]
BENCHMARK = "SmokeBench (Qi, Li & Barnes, WACV 2026, arXiv:2512.11215)"


def _score_sample(backend: PromptedBackend, task: str, protocol: str, sample: Any, image: np.ndarray) -> Dict[str, Any]:
    height, width = image.shape[:2]
    label = bool(sample.label)
    boxes = [list(box) for box in getattr(sample, "boxes", ()) or ()]
    record: Dict[str, Any] = {"image_id": sample.image_id, "label": label, "height": height, "width": width}
    if label:
        if not boxes:
            raise ValueError(f"smoke image {sample.image_id} has no ground-truth box")
        record.update(boxes=boxes, area=smoke_area(boxes), contrast=weber_contrast(image, boxes))
    prompt = get_prompt(task)
    if task == "classification":
        answer = backend.generate(image, prompt)
        prediction = parse_classification(answer)
        record.update(answer=answer, kind=answer_kind(answer), prediction=prediction, correct=prediction == label)
    elif task == "tile":
        answers = backend.generate_batch(crop_tiles(image, TILE_LAYOUT, protocol=protocol), prompt)
        if len(answers) != TILE_LAYOUT[0] * TILE_LAYOUT[1]:
            raise RuntimeError(f"backend returned {len(answers)} answers for {TILE_LAYOUT[0] * TILE_LAYOUT[1]} tiles")
        predicted = [index + 1 for index, answer in enumerate(answers) if parse_tile(answer)]
        target = covered_cells(boxes, height, width, TILE_LAYOUT, protocol=protocol)
        record.update(answers=answers, predicted=predicted, target=target, iou=set_iou(predicted, target))
    elif task == "grid":
        answer = backend.generate(draw_grid_overlay(image, GRID_LAYOUT), prompt)
        predicted = sorted(parse_grid(answer))
        target = covered_cells(boxes, height, width, GRID_LAYOUT, protocol=protocol)
        record.update(answer=answer, predicted=predicted, target=target, iou=set_iou(predicted, target))
    elif task == "detection":
        answer = backend.generate(image, prompt)
        predicted = parse_detection(answer)
        score, problem = detection_iou(predicted, boxes)
        record.update(answer=answer, predicted=predicted, iou=score, problem=problem)
    record["served_model"] = backend.last_served_model
    return record


def _mean(values: Sequence[float]) -> Optional[float]:
    return float(np.mean(values)) if len(values) else None


def summarize(records: Sequence[Dict[str, Any]], task: str) -> Dict[str, Any]:
    """Aggregate per-image records of one task into the paper's metrics."""
    smoke = [record for record in records if record["label"]]
    areas = [record["area"] for record in smoke]
    contrasts = [record["contrast"] for record in smoke]
    metrics: Dict[str, Any] = {}
    if task == "classification":
        smoke_free = [record for record in records if not record["label"]]
        hits = [float(record["correct"]) for record in smoke]
        metrics.update(
            accuracy_smoke=_mean(hits),
            accuracy_smoke_free=_mean([float(record["correct"]) for record in smoke_free]),
            accuracy_all=_mean([float(record["correct"]) for record in records]),
            n_smoke=len(smoke),
            n_smoke_free=len(smoke_free),
            answers=dict(Counter(record["kind"] for record in records)),
        )
    else:
        hits = [float(record["iou"]) for record in smoke]
        metrics.update(miou=_mean(hits), n_smoke=len(smoke))
        if task == "tile":
            metrics["mean_predicted_tiles"] = _mean([len(record["predicted"]) for record in smoke])
        elif task == "grid":
            metrics["answers_without_ids"] = sum(1 for record in smoke if not record["predicted"])
            metrics["answers_with_out_of_range_ids"] = sum(
                1 for record in smoke if any(not 1 <= cell <= GRID_LAYOUT[0] * GRID_LAYOUT[1] for cell in record["predicted"])
            )
        elif task == "detection":
            metrics["empty_predictions"] = sum(1 for record in smoke if record["predicted"] == [EMPTY_BOX])
            metrics["malformed_predictions"] = dict(Counter(record["problem"] for record in smoke if record["problem"]))
    if smoke:
        metrics["by_area"] = binned_means(hits, areas, AREA_BIN_EDGES, AREA_BIN_LABELS)
        metrics["by_contrast"] = binned_means(hits, contrasts, CONTRAST_BIN_EDGES, CONTRAST_BIN_LABELS)
        metrics["quintile_edges"] = {"area": quantile_edges(areas), "contrast": quantile_edges(contrasts)}
    return metrics


def _read_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _paper_reported(name: str, task: str) -> Optional[Dict[str, Any]]:
    from .catalog import load_prompted_cards  # noqa: PLC0415 - avoid a cycle at import time

    for card in load_prompted_cards():
        if card.name == name:
            reported = card.smokebench_reported.model_dump(exclude_none=True)
            return reported.get(task)
    return None


def evaluate_smokebench(
    backend: PromptedBackend,
    dataset: Sequence[Any],
    task: str,
    *,
    protocol: str = "paper",
    output_dir: Optional[PathLike] = None,
    load_image: Optional[Callable[[Any], np.ndarray]] = None,
    resume: bool = True,
    seed: Optional[int] = 0,
    limit: Optional[int] = None,
    progress: bool = False,
) -> Dict[str, Any]:
    """Run one SmokeBench task with ``backend`` over ``dataset`` and return the report.

    Args:
        backend: a :class:`~pyhazards.prompted.backends.PromptedBackend`.
        dataset: samples with ``image_id``, ``label`` and ``boxes`` (e.g.
            :class:`~pyhazards.datasets.figlib_smokebench.FIgLibSmokeBench`).
        task: ``classification``, ``tile``, ``grid`` or ``detection``.
        protocol: ``"paper"`` or ``"reference_code"`` (tile crops and box-to-cell labels exactly as
            the authors' script, including its two geometry bugs; see :mod:`pyhazards.prompted.geometry`).
        output_dir: where ``records.jsonl`` and ``report.json`` are written (nothing is written if None).
        load_image: sample -> RGB uint8 array; defaults to ``dataset.load_image``.
        resume: reuse the records already in ``output_dir`` for the same task, protocol, prompt and model.
        seed: ``torch.manual_seed`` before the first query (sampling backends at temperature 0.5).
        limit: evaluate only the first ``limit`` eligible samples.
        progress: show a progress bar.
    """
    if task not in TASKS:
        raise ValueError(f"unknown SmokeBench task {task!r}; expected one of {TASKS}")
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")
    loader = load_image or getattr(dataset, "load_image", None)
    if loader is None:
        raise ValueError("pass load_image= or a dataset with a load_image(sample) method")
    samples: List[Any] = [sample for sample in dataset if task == "classification" or sample.label]
    if limit is not None:
        samples = samples[: int(limit)]
    if not samples:
        raise ValueError(f"no samples to evaluate for task {task!r}")

    prompt = get_prompt(task)
    run = {
        "benchmark": BENCHMARK,
        "task": task,
        "protocol": protocol,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "layout": list(TILE_LAYOUT if task == "tile" else GRID_LAYOUT) if task in {"tile", "grid"} else None,
        "backend": backend.describe(),
        "dataset": dataset.describe() if hasattr(dataset, "describe") else {"samples": len(samples)},
        "seed": seed,
    }

    records_path = run_path = None
    done: Dict[str, Dict[str, Any]] = {}
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        records_path, run_path = out / "records.jsonl", out / "run.json"
        if resume and run_path.exists():
            previous = json.loads(run_path.read_text(encoding="utf-8"))
            keys = ("task", "protocol", "prompt_sha256")
            if any(previous.get(key) != run[key] for key in keys) or previous["backend"].get("model_id") != run["backend"].get("model_id"):
                raise ValueError(f"{out} holds a different run ({previous.get('task')}, {previous['backend'].get('model_id')}); use another output_dir or resume=False")
            done = {record["image_id"]: record for record in _read_records(records_path)}
        elif records_path.exists():
            records_path.unlink()
        run_path.write_text(json.dumps(run, indent=2, sort_keys=True), encoding="utf-8")

    if seed is not None:
        import torch  # noqa: PLC0415

        torch.manual_seed(seed)

    iterator: Iterable[Any] = samples
    if progress:
        from tqdm import tqdm  # noqa: PLC0415

        iterator = tqdm(samples, desc=f"SmokeBench {task}")
    records: List[Dict[str, Any]] = []
    for sample in iterator:
        if sample.image_id in done:
            records.append(done[sample.image_id])
            continue
        image = check_image(loader(sample))
        record = _score_sample(backend, task, protocol, sample, image)
        records.append(record)
        if records_path is not None:
            with records_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

    report = dict(run)
    report["metrics"] = summarize(records, task)
    report["served_models"] = sorted({record["served_model"] for record in records if record.get("served_model")})
    report["paper_reported"] = _paper_reported(getattr(backend, "name", ""), task)
    if output_dir is not None:
        (Path(output_dir) / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


__all__ = ["BENCHMARK", "evaluate_smokebench", "summarize"]
