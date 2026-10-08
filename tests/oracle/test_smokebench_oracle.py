"""pyhazards.prompted checked against the SmokeBench authors' evaluation scripts.

Reference (pinned in repos.yaml): SoraLink/MLLM @ 183cdca, the evaluation code of Qi, Li & Barnes,
"SmokeBench" (WACV 2026), committed by the first author. It has no licence, so it is fetched at test
time and nothing is copied. Only its definitions are executed (``load_definitions``): from
``mllm_smoke_locate.py`` the grid overlay (``ImagePreprocess.add_grid``), the box-to-cell labels
(``get_annotation_grid_number``) and the cell IoU; from ``evaluations.py`` the ``Evaluator`` class with
a fake model in place of the MLLM (its module imports Unified-IO 2, Qwen utilities, etc.); from
``summary_evaluation.py`` the smoke area and Weber contrast. ``torchvision.ops.box_iou`` and OpenCV are
the libraries those scripts call.

Checked: tile crops and tile/grid ground truth (``protocol="reference_code"`` equals the script on
every case, ``protocol="paper"`` wherever the script's geometry bugs do not apply), the grid overlay
pixel for pixel, the parsing and scoring of classification, tile, grid and detection answers
(well-formed and malformed), the cell / box IoUs, smoke area and Weber contrast, OpenCV's grey
conversion on all 2**24 colours, and that SmokeyNet's ``metadata.pkl`` boxes are SmokeBench's 5,046
(their area quintiles are the paper's bins). With ``PYHAZARDS_SMOKEBENCH_ROOT`` pointing at the
downloaded images (``pyhazards.datasets.figlib_smokebench.download_smokebench``, ~11.4 GB) the contrast
quintiles of all 5,046 images are also compared with the paper's (skipped otherwise; mandatory with
``PYHAZARDS_ORACLE_LARGE=1``).
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from oracle_utils import load_definitions, missing, oracle_asset, oracle_package, oracle_repo
from pyhazards.datasets.figlib_smokebench import (
    FIgLibSmokeBench,
    NUM_SEQUENCES,
    NUM_SMOKE_IMAGES,
    SmokeSample,
    load_smokeynet_boxes,
)
from pyhazards.prompted import CLASSIFICATION_PROMPT, DETECTION_PROMPT, GRID_PROMPT
from pyhazards.prompted.geometry import covered_cells, crop_tiles, draw_grid_overlay
from pyhazards.prompted.metrics import (
    AREA_BIN_EDGES,
    CONTRAST_BIN_EDGES,
    CONTRAST_BIN_EDGES_PRINTED,
    box_iou,
    detection_iou,
    opencv_gray,
    quantile_edges,
    set_iou,
    smoke_area,
    weber_contrast,
)
from pyhazards.prompted.parsers import parse_classification, parse_detection, parse_grid, parse_tile

OPENCV_VERSION = "4.10.0"
SIZES = [(1536, 2048), (2048, 3072), (1200, 1600), (1920, 2560), (37, 53)]  # (H, W); FIgLib sizes + odd


@pytest.fixture(scope="module")
def ref():
    cv2 = oracle_package("cv2", OPENCV_VERSION, "requirements-smokebench.txt")
    try:
        from torchvision.ops import box_iou as tv_box_iou
    except ImportError:
        missing("torchvision is not installed (the authors' detection scoring calls torchvision.ops.box_iou)")
    root = oracle_repo("SoraLink-MLLM")
    locate = load_definitions(
        root / "mllm_smoke_locate.py",
        ["ImagePreprocess", "get_annotation_grid_number", "compute_grid_IoU", "add_bbox"],
        {"cv2": cv2, "json": json},
    )
    evaluations = load_definitions(
        root / "evaluations.py",
        ["Evaluator"],
        {"re": re, "json": json, "torch": torch, "box_iou": tv_box_iou, "cv2": cv2, "Path": Path, "os": os, **locate},
    )
    summary = load_definitions(
        root / "summary_evaluation.py", ["read_area", "compute_luminance_contrast"], {"json": json, "cv2": cv2, "np": np}
    )
    return SimpleNamespace(cv2=cv2, tv_box_iou=tv_box_iou, Evaluator=evaluations["Evaluator"], **locate, **summary)


class FakeMLLM:
    """Stands in for the authors' model wrappers: records inputs (BGR, as they receive them)."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.images = []

    def predict(self, image, prompt):
        self.images.append(image.copy())
        return self.answers.pop(0)

    def batch_predict(self, images, prompt):
        self.images.extend(image.copy() for image in images)
        return [self.answers.pop(0) for _ in images]


def _random_image(rng, height, width):
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _write_frame(ref, directory: Path, name: str, rgb: np.ndarray) -> Path:
    path = directory / name
    assert ref.cv2.imwrite(str(path), np.ascontiguousarray(rgb[:, :, ::-1]))  # PNG: lossless
    return path


def _write_boxes(directory: Path, boxes) -> Path:
    path = directory / "annotation.json"
    path.write_text(json.dumps({"det_boxes": [list(map(int, box)) for box in boxes]}))
    return path


def _boxes(rng, height, width, n=40):
    boxes = []
    for _ in range(n):
        x1, x2 = sorted(rng.integers(0, width + 1, size=2).tolist())
        y1, y2 = sorted(rng.integers(0, height + 1, size=2).tolist())
        boxes.append([x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)])
    # edge cases: full frame, right / bottom edges, cell boundaries, single pixels in corners
    boxes += [
        [0, 0, width, height],
        [width - 5, 3, width, 9],
        [2, height - 4, 7, height],
        [width // 5, height // 5, 2 * (width // 5), 2 * (height // 5)],
        [width // 4 - 1, height // 3 - 1, width // 4, height // 3],
        [0, 0, 1, 1],
        [width - 2, height - 2, width - 1, height - 1],
    ]
    return boxes


# --------------------------------------------------------------------------------------------------


def test_released_prompts_differ_from_the_papers_only_in_wording(ref):
    evaluator = ref.Evaluator()
    stem = "Please look at this image. Detect if the image contains smoke. If you can find any smoke, return True. Otherwise, return False."
    assert evaluator.prompt_classification.startswith(stem) and CLASSIFICATION_PROMPT.startswith(stem)
    assert evaluator.prompt_classification != CLASSIFICATION_PROMPT
    grid_stem = 'Please output the numbered regions that contain smoke in JSON format as a list of dicts like [{"region": 1}, {"region": 2}].'
    assert grid_stem in evaluator.prompt_grid and grid_stem in GRID_PROMPT and evaluator.prompt_grid != GRID_PROMPT
    assert "[[x1, y1, x2, y2], [x1, y1, x2, y2]]" in evaluator.prompt_bbox and "[[x1, y1, x2, y2], [x1, y1, x2, y2]]" in DETECTION_PROMPT


@pytest.mark.parametrize("height, width", SIZES)
@pytest.mark.parametrize("layout", [(5, 5), (3, 4)])
def test_cell_ground_truth_matches_the_script(ref, tmp_path, height, width, layout):
    rng = np.random.default_rng(height * 7 + width + layout[1])
    rows, cols = layout
    shape_only = SimpleNamespace(shape=(height, width, 3))
    for box in _boxes(rng, height, width):
        reference = set(ref.get_annotation_grid_number(_write_boxes(tmp_path, [box]), shape_only, rows=rows, cols=cols))
        assert set(covered_cells([box], height, width, layout, protocol="reference_code")) == reference, box
        if box[2] < cols * (width // cols) and box[3] < rows * (height // rows):
            assert set(covered_cells([box], height, width, layout)) == reference, box
        else:  # the script's out-of-range row/column ids; the paper protocol clamps them
            assert set(covered_cells([box], height, width, layout)) <= set(range(1, rows * cols + 1))


def test_cell_iou_matches_the_script(ref):
    rng = np.random.default_rng(0)
    for _ in range(200):
        pred = set(rng.integers(0, 30, size=rng.integers(0, 8)).tolist())
        target = set(rng.integers(1, 26, size=rng.integers(1, 8)).tolist())
        assert set_iou(pred, target) == ref.compute_grid_IoU(pred, target)


@pytest.mark.parametrize("height, width", SIZES)
def test_grid_overlay_is_pixel_identical(ref, tmp_path, height, width):
    rgb = _random_image(np.random.default_rng(width), height, width)
    reference_bgr = ref.ImagePreprocess.add_grid(str(_write_frame(ref, tmp_path, "frame.png", rgb)))
    ours = draw_grid_overlay(rgb)
    assert np.array_equal(ours[:, :, ::-1], reference_bgr)
    assert not np.array_equal(ours, rgb)


GRID_ANSWERS = ['[{"region": 1}, {"region": 7}]', "[]", "no smoke", '```json\n[{"region": 25}, {"region": 26}]\n```', "Regions 3, 3 and 12."]


@pytest.mark.parametrize("height, width", [(1536, 2048), (2048, 3072), (37, 53)])
def test_grid_task_matches_the_script(ref, tmp_path, height, width):
    rng = np.random.default_rng(height)
    rgb = _random_image(rng, height, width)
    frame = _write_frame(ref, tmp_path, "1469223181_+00060.png", rgb)
    for box in _boxes(rng, height, width, n=3):
        annotation = _write_boxes(tmp_path, [box])
        for answer in GRID_ANSWERS:
            evaluator = ref.Evaluator()
            evaluator.model = FakeMLLM([answer])
            result = evaluator._evaluate_grid(str(frame), str(annotation))
            assert np.array_equal(evaluator.model.images[0], draw_grid_overlay(rgb)[:, :, ::-1])
            assert set(result["predict_grids"]) == parse_grid(answer)
            target = covered_cells([box], height, width, (5, 5), protocol="reference_code")
            assert set(result["annotation_grids"]) == set(target)
            assert result["iou"] == set_iou(parse_grid(answer), target)


TILE_ANSWERS = ["True", "False", "true.", "TRUE", "", "Yes, True", "False.", "I think it is true", "No", "True", "False", "maybe"]


@pytest.mark.parametrize("height, width", [(1536, 2048), (2048, 3072), (1200, 1600), (1920, 2560)])
def test_tile_task_matches_the_script(ref, tmp_path, height, width):
    rng = np.random.default_rng(width)
    rgb = _random_image(rng, height, width)
    frame = _write_frame(ref, tmp_path, "1469223181_+00060.png", rgb)
    for box in _boxes(rng, height, width, n=3):
        evaluator = ref.Evaluator()
        evaluator.model = FakeMLLM(TILE_ANSWERS)
        result = evaluator._evaluate_subimage_classification(str(frame), str(_write_boxes(tmp_path, [box])))
        crops = crop_tiles(rgb, protocol="reference_code")
        assert len(evaluator.model.images) == 12
        for theirs, ours in zip(evaluator.model.images, crops):
            assert np.array_equal(theirs, ours[:, :, ::-1])
        predicted = [i + 1 for i, answer in enumerate(TILE_ANSWERS) if parse_tile(answer)]
        assert result["predict_grids"] == predicted
        target = covered_cells([box], height, width, (3, 4), protocol="reference_code")
        assert set(result["annotation_grids"]) == set(target)
        assert result["iou"] == set_iou(predicted, target)
    paper_crops = crop_tiles(rgb)
    same = all(np.array_equal(a, b) for a, b in zip(paper_crops, crops))
    assert same == (height // 3 == width // 4)  # the script's swapped block sizes matter on 3072 x 2048


CLASSIFICATION_ANSWERS = ["True", "False", "true", "True.", "False, not True", "", "There is smoke.", "FALSE", "**True**"]


def test_classification_scoring_matches_the_script(ref, tmp_path):
    rgb = _random_image(np.random.default_rng(1), 30, 40)
    for name in ("1469223181_+00060.png", "1469220001_-01200.png", "1465065600_+00000.png"):
        frame = _write_frame(ref, tmp_path, name, rgb)
        for answer in CLASSIFICATION_ANSWERS:
            evaluator = ref.Evaluator()
            evaluator.model = FakeMLLM([answer])
            result = evaluator._evaluate_classification(str(frame))
            assert np.array_equal(evaluator.model.images[0], rgb[:, :, ::-1])
            assert result["prediction"] == parse_classification(answer)
            sample = SmokeSample(f"seq/{Path(name).stem}", frame, label=result["label"])
            assert result["label"] == (sample.offset_seconds >= 0)


DETECTION_ANSWERS = [
    "[[100, 120, 400, 300]]",
    "[[0, 0, 0, 0]]",
    "[[10, 20, 30, 40], [150, 100, 380, 290]]",
    "```json\n[[110.5, 130.25, 390.75, 280]]\n```",
    "Boxes: [[50, 60, 70, 80]] and [[100, 120, 400, 300]]",
    "[]",
    "[[]]",
    "[[], [100, 120, 400, 300]]",
    "No smoke detected.",
    '[{"bbox": [100, 120, 400, 300]}]',
    "[[1, 2, 3, 4], [5, 6",
    "[[[100, 120, 400, 300]]]",
    "[[100, 120, 400]]",
    "[[400, 300, 100, 120]]",
    "[[100, 120, 100, 300]]",
]
REFERENCE_CRASHES = ['[["a", "b", "c", "d"]]', "[[1, 2, 3, 4], [5, 6]]", "[[1, 2, 3, 4, 5]]"]


def test_detection_parsing_and_scoring_match_the_script(ref, tmp_path):
    rgb = _random_image(np.random.default_rng(2), 400, 500)
    frame = _write_frame(ref, tmp_path, "1469223181_+00060.png", rgb)
    for gt in ([[100, 120, 400, 300]], [[100, 120, 400, 300], [10, 20, 60, 80]]):
        annotation = _write_boxes(tmp_path, gt)
        for answer in DETECTION_ANSWERS:
            evaluator = ref.Evaluator()
            assert evaluator._retireve_bbox(answer) == parse_detection(answer), answer
            evaluator.model = FakeMLLM([answer])
            result = evaluator._evaluate_coordinate(str(frame), str(annotation))
            score, problem = detection_iou(parse_detection(answer), gt)
            assert score == float(np.max(result["iou"])), answer
            assert problem in (None, "malformed")
        for answer in REFERENCE_CRASHES:  # the script crashes; PyHazards scores 0 and flags it
            evaluator = ref.Evaluator()
            evaluator.model = FakeMLLM([answer])
            with pytest.raises((TypeError, ValueError, RuntimeError)):
                evaluator._evaluate_coordinate(str(frame), str(annotation))
            assert detection_iou(parse_detection(answer), gt)[0] == 0.0
            assert detection_iou(parse_detection(answer), gt)[1] is not None


def test_grid_number_parsing_matches_the_script(ref):
    evaluator = ref.Evaluator()
    for answer in GRID_ANSWERS + DETECTION_ANSWERS + CLASSIFICATION_ANSWERS + ["region ١٢", "0007"]:
        assert parse_grid(answer) == set(evaluator._retrieve_grid_number(answer)), answer


def test_box_iou_matches_torchvision(ref):
    generator = torch.Generator().manual_seed(0)
    for dtype in (torch.float32, torch.int64):
        a = (torch.rand(17, 4, generator=generator) * 500).to(dtype)
        b = (torch.rand(5, 4, generator=generator) * 500).to(dtype)
        a[:, 2:] += a[:, :2]
        b[:, 2:] += b[:, :2]
        assert torch.equal(box_iou(a, b), ref.tv_box_iou(a, b))


@pytest.mark.parametrize("height, width", [(1536, 2048), (2048, 3072), (37, 53)])
def test_area_and_weber_contrast_match_the_script(ref, tmp_path, height, width):
    rng = np.random.default_rng(width + 1)
    rgb = _random_image(rng, height, width)
    rgb[: height // 4] = 0  # black pixels are excluded from the background frame
    frame = _write_frame(ref, tmp_path, "frame.png", rgb)
    image = FIgLibSmokeBench.load_image(SmokeSample("seq/1_+00000", frame, True))
    assert np.array_equal(image, rgb)
    for box in _boxes(rng, height, width, n=12):
        if box[2] - box[0] < 1 or box[3] - box[1] < 1 or box[2] > width or box[3] > height:
            continue
        annotation = _write_boxes(tmp_path, [box])
        assert ref.read_area(annotation) == smoke_area([box])
        _, weber = ref.compute_luminance_contrast(annotation, frame)
        assert weber_contrast(image, [box]) == pytest.approx(weber, rel=1e-12, abs=0), box
    two = [[5, 5, 30, 30], [width - 20, height - 20, width - 1, height - 1]]
    _, weber = ref.compute_luminance_contrast(_write_boxes(tmp_path, two), frame)
    assert weber_contrast(image, two) == pytest.approx(weber, rel=1e-12, abs=0)


def test_opencv_grey_conversion_on_every_colour(ref):
    codes = np.arange(1 << 24, dtype=np.uint32)
    for chunk in np.array_split(codes, 16):
        rgb = np.stack([(chunk >> 16) & 255, (chunk >> 8) & 255, chunk & 255], axis=-1).astype(np.uint8)[None]
        expected = ref.cv2.cvtColor(np.ascontiguousarray(rgb[..., ::-1]), ref.cv2.COLOR_BGR2GRAY)
        assert np.array_equal(opencv_gray(rgb), expected)


def test_smokeynet_boxes_are_the_smokebench_smoke_set():
    boxes = load_smokeynet_boxes(oracle_asset("smokeynet_metadata") / "smokeynet_metadata.pkl")
    assert len(boxes) == NUM_SMOKE_IMAGES
    assert len({key.split("/")[0] for key in boxes}) == NUM_SEQUENCES
    assert all(len(image_boxes) == 1 and len(image_boxes[0]) == 4 for image_boxes in boxes.values())
    assert all("_+" in key for key in boxes)  # every annotated frame is at or after the first plume
    areas = [smoke_area(image_boxes) for image_boxes in boxes.values()]
    # Paper Sec. 4.2: "(42, 4356], (4356, 11232], (11232, 28565], (28565, 83157], and (83157, 2232020]"
    assert quantile_edges(areas) == pytest.approx(list(AREA_BIN_EDGES), rel=1e-12)


def _contrast(sample):
    return weber_contrast(FIgLibSmokeBench.load_image(sample), sample.boxes)


def test_contrast_quintiles_of_all_smokebench_images(ref, tmp_path):
    root = os.environ.get("PYHAZARDS_SMOKEBENCH_ROOT")
    if not root:
        reason = "set PYHAZARDS_SMOKEBENCH_ROOT to the downloaded SmokeBench images (~11.4 GB) to run this check"
        if os.environ.get("PYHAZARDS_ORACLE_LARGE") == "1":
            pytest.fail(reason)
        pytest.skip(reason)
    data = FIgLibSmokeBench(root, negatives=0)
    assert len(data) == NUM_SMOKE_IMAGES
    with ProcessPoolExecutor(max_workers=8) as pool:
        contrasts = list(pool.map(_contrast, data.positives, chunksize=32))
    edges = quantile_edges(contrasts)
    # Paper Sec. 4.2: "(0, 0.0246], (0.0246, 0.0549], (0.0549, 0.0878], (0.0878, 0.13], and (0.13, 0.728]"
    assert [float(f"{edge:.3g}") for edge in edges[1:]] == list(CONTRAST_BIN_EDGES_PRINTED[1:])
    assert edges[1:] == pytest.approx(list(CONTRAST_BIN_EDGES[1:]), rel=1e-9)
    # The authors' function (OpenCV decoding) on real FIgLib JPEGs gives the same values.
    for index in range(0, NUM_SMOKE_IMAGES, 250):
        sample = data.positives[index]
        _, weber = ref.compute_luminance_contrast(_write_boxes(tmp_path, sample.boxes), sample.path)
        assert contrasts[index] == pytest.approx(weber, rel=1e-12, abs=0), sample.image_id
