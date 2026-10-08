"""Image partitions of SmokeBench: 3 x 4 tiles, the numbered 5 x 5 grid, and box-to-cell labels.

SmokeBench (Qi, Li & Barnes, WACV 2026) defines the partitions in Sec. 3.2/4.3 and Figs. 2-5:
the tile task crops the image into a fixed 3 x 4 set of non-overlapping tiles; the grid task draws a
fixed 5 x 5 grid of numbered cells (left to right, top to bottom) on the whole image; a tile or cell
is ground-truth positive when it overlaps an annotated smoke box.

What the paper leaves open is taken from the authors' evaluation script (github.com/SoraLink/MLLM,
commit 183cdca488e8c7afa5a254f86ac675cb98951c73, no licence; used only as a test oracle, nothing is
copied): cells are ``H // rows`` by ``W // cols`` pixels (remainder rows/columns at the bottom/right
belong to no tile), a box ``[x1, y1, x2, y2]`` covers the cells containing the pixel indices
``x1..x2`` and ``y1..y2`` (both ends inclusive), and the grid overlay is drawn with OpenCV: a white
1-pixel cell outline and the cell number in red ``FONT_HERSHEY_SIMPLEX`` (scale 1, thickness 2),
placed at ``(cx - text_width // 2, cy - text_height // 2)``.

Two behaviours of that script contradict the paper and are reproduced only with
``protocol="reference_code"``:

* tile crops use ``image[i*hb:(i+1)*wb, j*wb:(j+1)*hb]`` (the block height and width are swapped in
  the end coordinates), so tiles overlap or leave gaps whenever ``H // 3 != W // 4`` -- e.g. on the
  3072 x 2048 FIgLib frames (1,994 of the 5,046 smoke images); on the 2048 x 1536, 2560 x 1920 and
  1600 x 1200 frames both protocols agree;
* a box reaching the last ``H % rows`` / ``W % cols`` pixels (e.g. ``x2 == W``) is mapped to row or
  column index ``rows`` / ``cols``, i.e. to a wrong cell id (the first cell of the next row, or an id
  above ``rows * cols``); this concerns 72 of the 5,046 smoke boxes for the grid and 19 for the tiles.
  ``protocol="paper"`` clamps those indices to the last row/column.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .prompts import GRID_LAYOUT, TILE_LAYOUT

PROTOCOLS: Tuple[str, ...] = ("paper", "reference_code")

Box = Sequence[float]
Rect = Tuple[int, int, int, int]


def _check_protocol(protocol: str) -> None:
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")


def _check_layout(height: int, width: int, layout: Tuple[int, int]) -> Tuple[int, int]:
    rows, cols = (int(v) for v in layout)
    if rows < 1 or cols < 1:
        raise ValueError(f"layout must be (rows, cols) with positive entries, got {layout}")
    if height < rows or width < cols:
        raise ValueError(
            f"image of shape (H={height}, W={width}) is too small for a {rows} x {cols} partition"
        )
    return rows, cols


def _check_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an RGB image array of shape (H, W, 3), got shape {array.shape}")
    return array


def tile_rects(height: int, width: int, layout: Tuple[int, int] = TILE_LAYOUT, *, protocol: str = "paper") -> List[Rect]:
    """Pixel slices ``(x1, y1, x2, y2)`` of the tiles, row-major (tile id = index + 1).

    The slice ``image[y1:y2, x1:x2]`` is the tile; with ``protocol="reference_code"`` the
    coordinates may exceed the image and are clipped by the slicing, as in the authors' script.
    """
    _check_protocol(protocol)
    rows, cols = _check_layout(height, width, layout)
    hb, wb = height // rows, width // cols
    rects: List[Rect] = []
    for i in range(rows):
        for j in range(cols):
            if protocol == "paper":
                rects.append((j * wb, i * hb, (j + 1) * wb, (i + 1) * hb))
            else:
                rects.append((j * wb, i * hb, (j + 1) * hb, (i + 1) * wb))
    return rects


def crop_tiles(image: np.ndarray, layout: Tuple[int, int] = TILE_LAYOUT, *, protocol: str = "paper") -> List[np.ndarray]:
    """Crop an ``(H, W, 3)`` image into its tiles (row-major, contiguous copies)."""
    array = _check_image(image)
    height, width = array.shape[:2]
    return [
        np.ascontiguousarray(array[y1:y2, x1:x2])
        for x1, y1, x2, y2 in tile_rects(height, width, layout, protocol=protocol)
    ]


def covered_cells(
    boxes: Sequence[Box],
    height: int,
    width: int,
    layout: Tuple[int, int],
    *,
    protocol: str = "paper",
) -> List[int]:
    """1-based ids (row-major) of the cells that the boxes ``[x1, y1, x2, y2]`` overlap.

    Box coordinates are pixel indices, both ends inclusive (the FIgLib/SmokeyNet boxes are the
    rounded extremes of the annotated polygon). Returns the sorted unique ids.
    """
    _check_protocol(protocol)
    rows, cols = _check_layout(height, width, layout)
    cell_h, cell_w = height // rows, width // cols
    ids = set()
    for box in boxes:
        if len(box) != 4:
            raise ValueError(f"boxes must be [x1, y1, x2, y2], got {list(box)}")
        x1, y1, x2, y2 = box
        r0, r1 = int(y1 // cell_h), int(y2 // cell_h)
        c0, c1 = int(x1 // cell_w), int(x2 // cell_w)
        if protocol == "paper":
            r0, r1 = min(max(r0, 0), rows - 1), min(max(r1, 0), rows - 1)
            c0, c1 = min(max(c0, 0), cols - 1), min(max(c1, 0), cols - 1)
        for i in range(r0, r1 + 1):
            for j in range(c0, c1 + 1):
                ids.add(i * cols + j + 1)
    return sorted(ids)


def require_cv2():
    try:
        import cv2  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The SmokeBench grid overlay is drawn with OpenCV, as in the authors' script. "
            "Install it with `pip install pyhazards[prompted]` (opencv-python-headless)."
        ) from exc
    return cv2


def draw_grid_overlay(image: np.ndarray, layout: Tuple[int, int] = GRID_LAYOUT) -> np.ndarray:
    """Return a copy of an RGB ``(H, W, 3)`` uint8 image with the numbered SmokeBench grid drawn on it."""
    array = _check_image(image)
    if array.dtype != np.uint8:
        raise ValueError(f"expected a uint8 image, got dtype {array.dtype}")
    cv2 = require_cv2()
    height, width = array.shape[:2]
    rows, cols = _check_layout(height, width, layout)
    cell_h, cell_w = height // rows, width // cols
    canvas = np.ascontiguousarray(array[:, :, ::-1])  # draw in OpenCV's BGR order
    font = cv2.FONT_HERSHEY_SIMPLEX
    cell_id = 1
    for i in range(rows):
        for j in range(cols):
            left, top = j * cell_w, i * cell_h
            cv2.rectangle(canvas, (left, top), (left + cell_w, top + cell_h), (255, 255, 255), 1)
            label = str(cell_id)
            (text_w, text_h), _ = cv2.getTextSize(label, font, 1, 2)
            origin = (left + cell_w // 2 - text_w // 2, top + cell_h // 2 - text_h // 2)
            cv2.putText(canvas, label, origin, font, 1, (0, 0, 255), 2)
            cell_id += 1
    return np.ascontiguousarray(canvas[:, :, ::-1])


__all__ = [
    "PROTOCOLS",
    "covered_cells",
    "crop_tiles",
    "draw_grid_overlay",
    "require_cv2",
    "tile_rects",
]
