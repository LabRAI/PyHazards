"""Prompted multimodal-LLM baselines for wildfire smoke, following SmokeBench.

SmokeBench (Qi, Li & Barnes, "SmokeBench: Evaluating Multimodal Large Language Models for Wildfire
Smoke Detection", WACV 2026, arXiv:2512.11215) asks multimodal LLMs four questions about FIgLib camera
frames: whether there is smoke (classification), which of 3 x 4 tiles contain smoke (each tile asked
separately), which cells of a numbered 5 x 5 grid contain smoke, and for smoke bounding boxes
(detection). This package holds the paper's verbatim prompts (:mod:`.prompts`), the image partitions
(:mod:`.geometry`), answer parsing (:mod:`.parsers`), the metrics and smoke-area / contrast bins
(:mod:`.metrics`), backends for the evaluated models (:mod:`.backends`, optional dependencies) and a
resumable runner (:func:`evaluate_smokebench`). The images come from
:mod:`pyhazards.datasets.figlib_smokebench`.

These are not ``nn.Module`` models and are not in the model registry; see the "Prompted VLM
Baselines" documentation page.
"""

from __future__ import annotations

from .backends import (
    GeminiBackend,
    Idefics2Backend,
    InternVL3Backend,
    OpenAIBackend,
    PromptedBackend,
    Qwen25VLBackend,
    available_backends,
    build_backend,
)
from .geometry import PROTOCOLS, covered_cells, crop_tiles, draw_grid_overlay, tile_rects
from .metrics import accuracy, detection_iou, set_iou, weber_contrast
from .parsers import parse_classification, parse_detection, parse_grid, parse_tile
from .prompts import (
    CLASSIFICATION_PROMPT,
    DETECTION_PROMPT,
    GRID_LAYOUT,
    GRID_PROMPT,
    PROMPTS,
    TASKS,
    TEMPERATURE,
    TILE_LAYOUT,
    TILE_PROMPT,
    get_prompt,
)
from .runner import evaluate_smokebench, summarize

__all__ = [
    "CLASSIFICATION_PROMPT",
    "DETECTION_PROMPT",
    "GRID_LAYOUT",
    "GRID_PROMPT",
    "GeminiBackend",
    "Idefics2Backend",
    "InternVL3Backend",
    "OpenAIBackend",
    "PROMPTS",
    "PROTOCOLS",
    "PromptedBackend",
    "Qwen25VLBackend",
    "TASKS",
    "TEMPERATURE",
    "TILE_LAYOUT",
    "TILE_PROMPT",
    "accuracy",
    "available_backends",
    "build_backend",
    "covered_cells",
    "crop_tiles",
    "detection_iou",
    "draw_grid_overlay",
    "evaluate_smokebench",
    "get_prompt",
    "parse_classification",
    "parse_detection",
    "parse_grid",
    "parse_tile",
    "set_iou",
    "summarize",
    "tile_rects",
    "weber_contrast",
]
