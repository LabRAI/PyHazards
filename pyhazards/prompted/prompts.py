"""SmokeBench task definitions and the verbatim prompts of the paper.

Source: Qi, Li & Barnes, "SmokeBench: Evaluating Multimodal Large Language Models for Wildfire
Smoke Detection", WACV 2026, arXiv:2512.11215v1, Section 4.3 "Prompting Design". The strings below
are the prompt boxes of the paper's LaTeX source (``sec/4_experiment.tex``) as rendered text: source
line breaks become single spaces, ``\\textbf``/``\\textit`` markup is dropped, ``$x_1, y_1, ...$`` is
written ``x1, y1, ...`` and ``\\{ \\}`` are braces. The double quotes of the grid prompt are the ASCII
``"`` of the LaTeX source (pdflatex prints them as a closing curly quote, a font artefact); the
authors' released evaluation script also sends ASCII quotes.

The released script (SoraLink/MLLM, commit 183cdca) words three of the prompts slightly differently
from the paper (e.g. "Do MUST not return any other words"); PyHazards uses the paper's text, which is
the published protocol, and lists the difference in the docs.
"""

from __future__ import annotations

from typing import Dict, Tuple

TASKS: Tuple[str, ...] = ("classification", "tile", "grid", "detection")

CLASSIFICATION_PROMPT = (
    "Please look at this image. Detect if the image contains smoke. "
    "If you can find any smoke, return True. Otherwise, return False. "
    "Do not return any other words."
)

# Tile-based localization reuses the classification prompt on each tile (paper Sec. 4.3).
TILE_PROMPT = CLASSIFICATION_PROMPT

GRID_PROMPT = (
    "Please look at this image, which is divided into several numbered regions "
    "(from left to right, top to bottom). Please output the numbered regions that contain smoke "
    'in JSON format as a list of dicts like [{"region": 1}, {"region": 2}]. '
    "If you cannot find any smoke, return an empty list []. without other words"
)

DETECTION_PROMPT = (
    "Detect all smoke and output bounding boxes in the format "
    "[[x1, y1, x2, y2], [x1, y1, x2, y2]]. "
    "If you cannot find any smoke, return an empty list [] without other words. "
    "Do not return dictionaries."
)

PROMPTS: Dict[str, str] = {
    "classification": CLASSIFICATION_PROMPT,
    "tile": TILE_PROMPT,
    "grid": GRID_PROMPT,
    "detection": DETECTION_PROMPT,
}

# Fixed partitions of the paper: 3 x 4 tiles (Sec. 4.3, Figs. 2 and 4, Tables 4-5) and a 5 x 5
# numbered grid (Figs. 3 and 5, Tables 6-7), both as (rows, columns).
TILE_LAYOUT: Tuple[int, int] = (3, 4)
GRID_LAYOUT: Tuple[int, int] = (5, 5)

# "All inference is conducted with the decoding temperature fixed at 0.5" (Sec. 4.2).
TEMPERATURE = 0.5


def get_prompt(task: str) -> str:
    """Return the verbatim SmokeBench prompt for ``task``."""
    if task not in PROMPTS:
        raise ValueError(f"unknown SmokeBench task {task!r}; expected one of {TASKS}")
    return PROMPTS[task]


__all__ = [
    "CLASSIFICATION_PROMPT",
    "DETECTION_PROMPT",
    "GRID_LAYOUT",
    "GRID_PROMPT",
    "PROMPTS",
    "TASKS",
    "TEMPERATURE",
    "TILE_LAYOUT",
    "TILE_PROMPT",
    "get_prompt",
]
