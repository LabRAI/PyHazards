"""Turn SmokeBench model answers into predictions.

The paper only says that ``True`` maps to 1 and ``False`` to 0 (Sec. 4.4); how other answers are
read is not stated. The rules below follow the authors' evaluation script (github.com/SoraLink/MLLM,
commit 183cdca488e8c7afa5a254f86ac675cb98951c73, ``evaluations.py``; no licence, used only as a test
oracle, nothing is copied):

* classification: smoke is predicted iff the answer contains the substring ``True`` (case-sensitive).
  Every other answer -- ``False``, ``true``, refusals, empty text -- counts as "no smoke", so an
  unparseable answer is wrong on a smoke image and right on a smoke-free one;
* tile: a tile is predicted positive iff its answer contains ``True`` or ``true``;
* grid: every run of digits in the answer is a predicted cell id (``[{"region": 3}]`` -> ``{3}``);
  ids outside ``1..25`` are kept and only enlarge the union; an answer without digits is the
  empty prediction;
* detection: the first ``[[ ... ]]`` span (non-greedy) is decoded as JSON; empty entries are
  dropped; an answer without such a span, invalid JSON, or an empty list becomes the single box
  ``[0, 0, 0, 0]`` (IoU 0 with any real box).

:func:`answer_kind` additionally labels classification answers as ``"true"``, ``"false"`` or
``"other"`` for reporting how often a model ignored the requested format; it does not change scores.
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Set

_GRID_NUMBER = re.compile(r"\d+")
_BOX_LIST = re.compile(r"\[\s*\[.*?\]\s*\]", flags=re.DOTALL)
_JSON_FENCE = re.compile(r"```json\s*(.*?)\s*```", flags=re.DOTALL)

EMPTY_BOX = [0, 0, 0, 0]


def parse_classification(answer: str) -> bool:
    """SmokeBench classification answer -> smoke predicted (``"True" in answer``)."""
    return "True" in answer


def answer_kind(answer: str) -> str:
    """``"true"``/``"false"`` when the answer contains that word (``True`` wins), else ``"other"``."""
    if "True" in answer:
        return "true"
    if "False" in answer:
        return "false"
    return "other"


def parse_tile(answer: str) -> bool:
    """Tile answer -> tile predicted positive (``"True"`` or ``"true"`` in the answer)."""
    return "True" in answer or "true" in answer


def parse_grid(answer: str) -> Set[int]:
    """Grid answer -> set of predicted cell ids (every run of digits)."""
    return {int(token) for token in _GRID_NUMBER.findall(answer)}


def parse_detection(answer: str) -> List[Any]:
    """Detection answer -> list of predicted boxes (raw JSON values, ``[[0, 0, 0, 0]]`` if none)."""
    try:
        boxes = json.loads(_BOX_LIST.findall(answer)[0])
        boxes = [box for box in boxes if box]
        if len(boxes) == 0:
            return [list(EMPTY_BOX)]
        boxes = [list(EMPTY_BOX) if len(box) == 0 else box for box in boxes]
    except Exception:  # noqa: BLE001 - any malformed answer is the empty prediction
        return [list(EMPTY_BOX)]
    return boxes


def strip_json_fence(text: str) -> str:
    """Return the content of the first Markdown ```json fence, else the stripped text.

    The authors apply this to Qwen2.5-VL answers before parsing (``QwenVL.extract_json_from_markdown``).
    """
    match = _JSON_FENCE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


__all__ = [
    "EMPTY_BOX",
    "answer_kind",
    "parse_classification",
    "parse_detection",
    "parse_grid",
    "parse_tile",
    "strip_json_fence",
]
