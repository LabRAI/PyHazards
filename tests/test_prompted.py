"""Tests of the SmokeBench prompted-baseline module (no network, GPU, API key, Pillow or OpenCV needed)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
import torch

from pyhazards.prompted import (
    CLASSIFICATION_PROMPT,
    DETECTION_PROMPT,
    GRID_PROMPT,
    PROMPTS,
    TASKS,
    TEMPERATURE,
    TILE_PROMPT,
    PromptedBackend,
    available_backends,
    build_backend,
    evaluate_smokebench,
)
from pyhazards.prompted import runner as runner_module
from pyhazards.prompted.catalog import DOCS_PAGE_PATH, load_prompted_cards, render_prompted_page
from pyhazards.prompted.geometry import covered_cells, crop_tiles, tile_rects
from pyhazards.prompted.metrics import (
    AREA_BIN_EDGES,
    CONTRAST_BIN_EDGES,
    CONTRAST_BIN_EDGES_PRINTED,
    accuracy,
    assign_bin,
    binned_means,
    box_iou,
    detection_iou,
    opencv_gray,
    quantile_edges,
    set_iou,
    smoke_area,
    weber_contrast,
)
from pyhazards.prompted.parsers import (
    answer_kind,
    parse_classification,
    parse_detection,
    parse_grid,
    parse_tile,
    strip_json_fence,
)

# --------------------------------------------------------------------------------------------------
# Prompts: byte-identical to the boxes of arXiv:2512.11215v1, sec/4_experiment.tex, rendered as text.
# --------------------------------------------------------------------------------------------------

PAPER_CLASSIFICATION = (
    "Please look at this image. Detect if the image contains smoke. If you can find any smoke, return True. "
    "Otherwise, return False. Do not return any other words."
)
PAPER_GRID = (
    "Please look at this image, which is divided into several numbered regions (from left to right, top to "
    "bottom). Please output the numbered regions that contain smoke in JSON format as a list of dicts like "
    '[{"region": 1}, {"region": 2}]. If you cannot find any smoke, return an empty list []. without other words'
)
PAPER_DETECTION = (
    "Detect all smoke and output bounding boxes in the format [[x1, y1, x2, y2], [x1, y1, x2, y2]]. If you "
    "cannot find any smoke, return an empty list [] without other words. Do not return dictionaries."
)


def test_prompts_are_the_papers_verbatim():
    assert CLASSIFICATION_PROMPT == PAPER_CLASSIFICATION
    assert TILE_PROMPT == PAPER_CLASSIFICATION  # the tile task reuses the classification prompt
    assert GRID_PROMPT == PAPER_GRID
    assert DETECTION_PROMPT == PAPER_DETECTION
    assert set(PROMPTS) == set(TASKS) == {"classification", "tile", "grid", "detection"}
    assert TEMPERATURE == 0.5
    digests = {task: hashlib.sha256(PROMPTS[task].encode("utf-8")).hexdigest()[:16] for task in ("classification", "grid", "detection")}
    assert digests == {
        "classification": hashlib.sha256(PAPER_CLASSIFICATION.encode()).hexdigest()[:16],
        "grid": hashlib.sha256(PAPER_GRID.encode()).hexdigest()[:16],
        "detection": hashlib.sha256(PAPER_DETECTION.encode()).hexdigest()[:16],
    }
    for prompt in PROMPTS.values():
        assert prompt.isascii() and "  " not in prompt and prompt == prompt.strip()


# --------------------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------------------


def test_tiles_are_a_3x4_partition():
    rects = tile_rects(1536, 2048)
    assert len(rects) == 12
    assert rects[0] == (0, 0, 512, 512) and rects[3] == (1536, 0, 2048, 512) and rects[11] == (1536, 1024, 2048, 1536)
    covered = np.zeros((1536, 2048), dtype=int)
    for x1, y1, x2, y2 in rects:
        covered[y1:y2, x1:x2] += 1
    assert covered.min() == covered.max() == 1  # non-overlapping and complete
    # When H // 3 == W // 4 the authors' crop formula gives the same tiles.
    assert tile_rects(1536, 2048, protocol="reference_code") == rects
    assert tile_rects(1200, 1600, protocol="reference_code") == tile_rects(1200, 1600)


def test_tiles_of_3072x2048_frames_and_the_reference_quirk():
    paper = tile_rects(2048, 3072)
    assert paper[0] == (0, 0, 768, 682) and paper[11] == (2304, 1364, 3072, 2046)
    quirk = tile_rects(2048, 3072, protocol="reference_code")
    # image[i*hb:(i+1)*wb, j*wb:(j+1)*hb] with hb = 682, wb = 768
    assert quirk[0] == (0, 0, 682, 768) and quirk[5] == (768, 682, 1364, 1536) and quirk[11] == (2304, 1364, 2728, 2304)
    # The same on a small 20 x 30 image (hb = 6, wb = 7).
    image = (np.arange(20 * 30 * 3) % 251).reshape(20, 30, 3).astype(np.uint8)
    tiles = crop_tiles(image)
    assert [t.shape for t in tiles] == [(6, 7, 3)] * 12
    assert np.array_equal(tiles[5], image[6:12, 7:14])
    quirk_tiles = crop_tiles(image, protocol="reference_code")
    assert quirk_tiles[5].shape == (8, 5, 3) and np.array_equal(quirk_tiles[5], image[6:14, 7:12])
    assert quirk_tiles[11].shape == (8, 3, 3)  # rows 12:21 clipped to 20, columns 21:24


def test_partition_input_validation():
    with pytest.raises(ValueError, match="shape"):
        crop_tiles(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError, match="too small"):
        tile_rects(2, 3)
    with pytest.raises(ValueError, match="protocol"):
        tile_rects(30, 40, protocol="other")
    with pytest.raises(ValueError, match="x1, y1, x2, y2"):
        covered_cells([[1, 2, 3]], 100, 100, (5, 5))


def test_covered_cells_hand_cases():
    # 1536 x 2048 frame, 5 x 5 grid: cells are 307 x 409 pixels.
    assert covered_cells([[10, 10, 20, 20]], 1536, 2048, (5, 5)) == [1]
    assert covered_cells([[400, 300, 420, 320]], 1536, 2048, (5, 5)) == [1, 2, 6, 7]
    # both box ends are inclusive pixel indices: x2 == 409 touches column 1
    assert covered_cells([[0, 0, 408, 306]], 1536, 2048, (5, 5)) == [1]
    assert covered_cells([[0, 0, 409, 306]], 1536, 2048, (5, 5)) == [1, 2]
    # union over several boxes
    assert covered_cells([[0, 0, 5, 5], [2040, 1530, 2047, 1535]], 1536, 2048, (5, 5)) == [1, 25]
    # tile layout (3 x 4) on a 2048 x 3072 frame: 682 x 768 tiles
    assert covered_cells([[700, 600, 800, 700]], 2048, 3072, (3, 4)) == [1, 2, 5, 6]


def test_covered_cells_at_the_image_edge():
    # 2048 // 5 = 409 -> the last 3 columns (2045..2047) and x2 == W fall outside 5 * 409.
    box = [[1900, 100, 2048, 200]]
    assert covered_cells(box, 1536, 2048, (5, 5)) == [5]
    # the authors' formula maps column index 5 of row 0 to id 6 (first cell of row 1)
    assert covered_cells(box, 1536, 2048, (5, 5), protocol="reference_code") == [5, 6]
    bottom = [[100, 1500, 200, 1536]]
    assert covered_cells(bottom, 1536, 2048, (5, 5)) == [21]
    assert covered_cells(bottom, 1536, 2048, (5, 5), protocol="reference_code") == [21, 26]


# --------------------------------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer, smoke, kind",
    [
        ("True", True, "true"),
        ("False", False, "false"),
        ("True.", True, "true"),
        ("Answer: True", True, "true"),
        ("False, but maybe True", True, "true"),
        ("true", False, "other"),  # case-sensitive, as in the authors' script
        ("TRUE", False, "other"),
        ("", False, "other"),
        ("I cannot determine whether there is smoke.", False, "other"),
        ("No smoke.", False, "other"),
    ],
)
def test_classification_parsing(answer, smoke, kind):
    assert parse_classification(answer) is smoke
    assert answer_kind(answer) == kind


def test_tile_parsing_accepts_lowercase():
    assert parse_tile("True") and parse_tile("true") and parse_tile("It is true.")
    assert not parse_tile("False") and not parse_tile("TRUE") and not parse_tile("")


@pytest.mark.parametrize(
    "answer, cells",
    [
        ('[{"region": 1}, {"region": 7}]', {1, 7}),
        ('```json\n[{"region": 12}]\n```', {12}),
        ("[]", set()),
        ("No smoke is visible.", set()),
        ("Regions 3 and 3 and 26", {3, 26}),  # duplicates collapse, out-of-range ids are kept
        ('[{"region": 0}]', {0}),
        ('{"regions": [4, 5]}', {4, 5}),
    ],
)
def test_grid_parsing(answer, cells):
    assert parse_grid(answer) == cells


@pytest.mark.parametrize(
    "answer, boxes",
    [
        ("[[10, 20, 30, 40]]", [[10, 20, 30, 40]]),
        ("[[1, 2, 3, 4], [5, 6, 7, 8]]", [[1, 2, 3, 4], [5, 6, 7, 8]]),
        ("Boxes: [[1.5, 2, 3, 4]] done", [[1.5, 2, 3, 4]]),
        ("```json\n[[1, 2, 3, 4]]\n```", [[1, 2, 3, 4]]),
        ("[]", [[0, 0, 0, 0]]),
        ("[[]]", [[0, 0, 0, 0]]),
        ("[[], [1, 2, 3, 4]]", [[1, 2, 3, 4]]),
        ("no smoke", [[0, 0, 0, 0]]),
        ('[{"bbox": [1, 2, 3, 4]}]', [[0, 0, 0, 0]]),  # dictionaries are not boxes
        ("[[1, 2, 3, 4], [5, 6", [[0, 0, 0, 0]]),  # truncated JSON
        ("[[[1, 2, 3, 4]]]", [[0, 0, 0, 0]]),  # the non-greedy span is unbalanced
        ("[[1, 2, 3]]", [[1, 2, 3]]),  # kept; scored as malformed
        ('[["a", "b", "c", "d"]]', [["a", "b", "c", "d"]]),
        ("[[1, 2, 3, 4]] and [[5, 6, 7, 8]]", [[1, 2, 3, 4]]),  # first span only
    ],
)
def test_detection_parsing(answer, boxes):
    assert parse_detection(answer) == boxes


def test_json_fence_stripping():
    assert strip_json_fence('```json\n[{"region": 2}]\n```') == '[{"region": 2}]'
    assert strip_json_fence("  True \n") == "True"
    assert strip_json_fence("text ```json [1] ``` more") == "[1]"


# --------------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------------


def test_accuracy_and_set_iou_hand_cases():
    assert accuracy([True, False, True, False], [True, True, True, False]) == 0.75
    assert set_iou([1, 2, 3], [2, 3, 4, 5]) == pytest.approx(2 / 5)
    assert set_iou([], [7]) == 0.0
    assert set_iou([7], [7]) == 1.0
    with pytest.raises(ValueError):
        set_iou([], [])
    with pytest.raises(ValueError):
        accuracy([True], [])


def test_box_iou_matches_hand_computation():
    pred = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]])
    gt = torch.tensor([[5, 5, 15, 15]])
    iou = box_iou(pred, gt)
    assert iou.shape == (3, 1)
    assert iou[:, 0].tolist() == pytest.approx([25 / 175, 1.0, 0.0])
    with pytest.raises(ValueError, match="shape"):
        box_iou(torch.zeros(2, 3), gt)


def test_detection_iou_takes_the_best_pair_and_flags_malformed_boxes():
    gt = [(100, 100, 200, 200)]
    assert detection_iou([[0, 0, 0, 0]], gt) == (0.0, None)
    score, problem = detection_iou([[100, 100, 150, 200], [100, 100, 200, 200], [0, 0, 10, 10]], gt)
    assert score == pytest.approx(1.0) and problem is None
    assert detection_iou([[150, 100, 250, 200]], gt)[0] == pytest.approx(5000 / 15000)
    assert detection_iou([[1.5, 2, 3, 4]], gt) == (0.0, None)
    assert detection_iou([[1, 2, 3]], gt) == (0.0, "malformed")
    assert detection_iou([["a", "b", "c", "d"]], gt) == (0.0, "non_numeric")
    assert detection_iou([[1, 2, 3, 4], [1, 2]], gt) == (0.0, "non_numeric")  # ragged
    assert detection_iou([[5, 5, 5, 5]], [(5, 5, 5, 5)]) == (0.0, "nan")  # zero union


def test_opencv_grey_conversion_constants():
    pixels = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]]], dtype=np.uint8)
    # cv2.cvtColor(BGR2GRAY) of pure red, green, blue, white and black
    assert opencv_gray(pixels).tolist() == [[76, 150, 29, 255, 0]]


def test_weber_contrast_hand_case():
    image = np.full((100, 100, 3), 100, dtype=np.uint8)
    image[40:60, 40:60] = 120  # box pixels y1:y2, x1:x2
    assert weber_contrast(image, [(40, 40, 60, 60)]) == pytest.approx(0.2)
    # zero-valued frame pixels are ignored; the frame is clipped at the image border
    image[20:30, 20:80] = 0
    assert weber_contrast(image, [(40, 40, 60, 60)]) == pytest.approx(0.2)
    assert weber_contrast(image, [(0, 0, 100, 100)]) == 0.0  # no frame left
    assert smoke_area([(10, 20, 30, 60), (0, 0, 1000, 1000)]) == 800  # first box only


def test_bins_use_the_papers_quintiles():
    assert AREA_BIN_EDGES == (42, 4356, 11232, 28565, 83157, 2232020)
    assert [float(f"{edge:.3g}") for edge in CONTRAST_BIN_EDGES[1:]] == list(CONTRAST_BIN_EDGES_PRINTED[1:])
    assert [assign_bin(v, AREA_BIN_EDGES) for v in (1, 42, 4356, 4357, 11232, 83157, 83158, 10**8)] == [0, 0, 0, 1, 1, 3, 4, 4]
    means = binned_means([1.0, 0.0, 1.0], [100, 5000, 90000], AREA_BIN_EDGES, ["a", "b", "c", "d", "e"])
    assert means == {"a": {"mean": 1.0, "n": 1}, "b": {"mean": 0.0, "n": 1}, "c": {"mean": None, "n": 0}, "d": {"mean": None, "n": 0}, "e": {"mean": 1.0, "n": 1}}


def test_quantile_edges_match_pandas_qcut():
    pandas = pytest.importorskip("pandas")
    values = np.random.default_rng(0).integers(40, 2_000_000, size=503).tolist()
    edges = quantile_edges(values)
    categories = pandas.qcut(pandas.Series(values), 5).cat.categories
    assert edges[1:-1] == pytest.approx([interval.right for interval in categories[:-1]])
    codes = pandas.qcut(pandas.Series(values), 5).cat.codes.tolist()
    assert [assign_bin(v, edges) for v in values] == codes


# --------------------------------------------------------------------------------------------------
# Runner with a deterministic fake backend
# --------------------------------------------------------------------------------------------------


@dataclass
class FakeSample:
    image_id: str
    label: bool
    boxes: Tuple[Tuple[int, int, int, int], ...] = ()


class FakeDataset(list):
    """Synthetic 60 x 80 frames: smoke images carry a bright box, smoke-free ones are flat."""

    def load_image(self, sample):
        image = np.full((60, 80, 3), 50, dtype=np.uint8)
        for x1, y1, x2, y2 in sample.boxes:
            image[y1:y2, x1:x2] = 200
        return image


class FakeBackend(PromptedBackend):
    name = "fake"
    model_id = "fake/deterministic"

    def __init__(self, grid_answer='[{"region": 1}, {"region": 2}]', box_answer="[[0, 0, 40, 30]]"):
        super().__init__()
        self.calls: List[Tuple[Tuple[int, ...], str]] = []
        self.grid_answer, self.box_answer = grid_answer, box_answer

    def generate(self, image, prompt):
        self.calls.append((image.shape, prompt))
        if prompt == CLASSIFICATION_PROMPT:
            return "True" if image.max() > 150 else "False"
        if prompt == GRID_PROMPT:
            return self.grid_answer
        return self.box_answer


def _dataset():
    return FakeDataset(
        [
            FakeSample("seq/1_+00060", True, ((0, 0, 20, 15),)),  # top-left corner (tile 1)
            FakeSample("seq/2_+00120", True, ((30, 20, 50, 40),)),  # centre: tiles 6, 7, 10, 11 overlap
            FakeSample("seq/3_+00180", True, ((5, 5, 6, 6),)),  # tiny smoke, still bright
            FakeSample("seq/4_-00060", False),
            FakeSample("seq/5_-00120", False),
        ]
    )


def test_runner_classification_report(tmp_path):
    backend = FakeBackend()
    report = evaluate_smokebench(backend, _dataset(), "classification", output_dir=tmp_path)
    metrics = report["metrics"]
    assert metrics["accuracy_smoke"] == 1.0 and metrics["accuracy_smoke_free"] == 1.0 and metrics["accuracy_all"] == 1.0
    assert metrics["n_smoke"] == 3 and metrics["n_smoke_free"] == 2
    assert metrics["answers"] == {"true": 3, "false": 2}
    assert sum(item["n"] for item in metrics["by_area"].values()) == 3
    assert metrics["by_area"]["Very Small"] == {"mean": 1.0, "n": 3}  # areas 300, 400 and 1 pixels
    assert report["prompt"] == CLASSIFICATION_PROMPT and report["task"] == "classification"
    assert report["paper_reported"] is None  # not a SmokeBench model
    records = [json.loads(line) for line in (tmp_path / "records.jsonl").read_text().splitlines()]
    assert [r["image_id"] for r in records] == [s.image_id for s in _dataset()]
    assert records[0]["contrast"] > 0 and "contrast" not in records[3]
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["metrics"]["accuracy_all"] == 1.0
    assert len(backend.calls) == 5 and all(shape == (60, 80, 3) for shape, _ in backend.calls)


def test_runner_tile_task_scores_tile_sets():
    backend = FakeBackend()
    report = evaluate_smokebench(backend, _dataset(), "tile")
    # 60 x 80 frame -> 20 x 20 tiles; the fake model says True for tiles with bright pixels.
    # Box ends are inclusive, so (0, 0, 20, 15) also covers tile 2 and (30, 20, 50, 40) tiles 10-11:
    # IoUs {1} vs {1, 2} = 0.5, {6, 7} vs {6, 7, 10, 11} = 0.5, {1} vs {1} = 1.
    assert report["metrics"]["n_smoke"] == 3 and report["metrics"]["miou"] == pytest.approx(2 / 3)
    assert report["metrics"]["mean_predicted_tiles"] == pytest.approx(4 / 3)
    assert len(backend.calls) == 36 and all(shape == (20, 20, 3) for shape, _ in backend.calls)
    assert all(prompt == TILE_PROMPT for _, prompt in backend.calls)


def test_runner_tile_targets_include_boundary_tiles():
    # Box (30, 20, 50, 40) ends exactly on tile boundaries x = 40 / y = 40: inclusive ends add tiles.
    assert covered_cells([(30, 20, 50, 40)], 60, 80, (3, 4)) == [6, 7, 10, 11]


def test_runner_grid_task_with_a_stub_overlay(monkeypatch):
    monkeypatch.setattr(runner_module, "draw_grid_overlay", lambda image, layout: image)
    backend = FakeBackend(grid_answer='[{"region": 1}, {"region": 2}]')
    report = evaluate_smokebench(backend, _dataset(), "grid")
    # 60 x 80 frame, 5 x 5 grid of 12 x 16 cells; targets {1, 2, 6, 7}, {7, 8, 9, 12, 13, 14, 17, 18, 19}, {1}
    expected = np.mean([0.5, 0.0, 0.5])
    assert report["metrics"]["miou"] == pytest.approx(expected)
    assert report["metrics"]["answers_without_ids"] == 0


def test_runner_detection_task():
    backend = FakeBackend(box_answer="[[0, 0, 20, 15], [30, 20, 50, 40]]")
    report = evaluate_smokebench(backend, _dataset(), "detection")
    # third image: best pair is (0,0,20,15) vs (5,5,6,6) -> 1 / 300
    assert report["metrics"]["miou"] == pytest.approx(np.mean([1.0, 1.0, 1 / 300]))
    assert report["metrics"]["empty_predictions"] == 0
    bad = evaluate_smokebench(FakeBackend(box_answer="I see no smoke"), _dataset(), "detection")
    assert bad["metrics"]["miou"] == 0.0 and bad["metrics"]["empty_predictions"] == 3


def test_runner_resumes_and_refuses_a_different_run(tmp_path):
    evaluate_smokebench(FakeBackend(), _dataset(), "classification", output_dir=tmp_path, limit=2)
    backend = FakeBackend()
    report = evaluate_smokebench(backend, _dataset(), "classification", output_dir=tmp_path)
    assert len(backend.calls) == 3  # the first two images were reused
    assert report["metrics"]["n_smoke"] + report["metrics"]["n_smoke_free"] == 5
    with pytest.raises(ValueError, match="different run"):
        evaluate_smokebench(FakeBackend(), _dataset(), "detection", output_dir=tmp_path)
    fresh = FakeBackend()
    evaluate_smokebench(fresh, _dataset(), "classification", output_dir=tmp_path, resume=False)
    assert len(fresh.calls) == 5


def test_runner_attaches_the_papers_numbers_for_smokebench_models():
    backend = FakeBackend()
    backend.name = "qwen2_5_vl_7b"
    report = evaluate_smokebench(backend, _dataset(), "classification")
    assert report["paper_reported"]["area"]["Overall"] == 0.380
    assert report["paper_reported"]["smoke_free"] == 0.982


def test_runner_input_validation():
    with pytest.raises(ValueError, match="task"):
        evaluate_smokebench(FakeBackend(), _dataset(), "segmentation")
    with pytest.raises(ValueError, match="protocol"):
        evaluate_smokebench(FakeBackend(), _dataset(), "tile", protocol="loose")
    with pytest.raises(ValueError, match="no samples"):
        evaluate_smokebench(FakeBackend(), FakeDataset([FakeSample("seq/4_-00060", False)]), "tile")
    with pytest.raises(ValueError, match="load_image"):
        evaluate_smokebench(FakeBackend(), list(_dataset()), "classification")
    with pytest.raises(ValueError, match="RGB uint8"):
        evaluate_smokebench(FakeBackend(), _dataset(), "classification", load_image=lambda s: np.zeros((4, 4)))


# --------------------------------------------------------------------------------------------------
# Cards, docs and backend construction
# --------------------------------------------------------------------------------------------------


def test_cards_cover_the_backend_presets_and_render_the_docs_page():
    cards = load_prompted_cards()
    assert [card.name for card in cards] == ["idefics2_8b", "qwen2_5_vl_7b", "qwen2_5_vl_32b", "internvl3_14b", "gemini_2_5_pro", "gpt_4o"]
    assert sorted(card.name for card in cards) == available_backends()
    for card in cards:
        for task in card.smokebench_tasks:
            numbers = getattr(card.smokebench_reported, task)
            if numbers.area is not None:
                assert numbers.area["Overall"] == numbers.contrast["Overall"], card.name
    page = render_prompted_page(cards)
    assert DOCS_PAGE_PATH.read_text(encoding="utf-8") == page, "run scripts/render_prompted_docs.py"
    for text in (PAPER_CLASSIFICATION, PAPER_GRID, PAPER_DETECTION, "arxiv.org/abs/2502.13923", "arxiv.org/abs/2507.06261", "arxiv.org/abs/2504.10479", "arxiv.org/abs/2405.02246", "arxiv.org/abs/2410.21276", "arxiv.org/abs/2512.11215"):
        assert text in page


def test_models_page_and_index_link_the_prompted_page():
    root = Path(__file__).resolve().parents[1]
    assert ":doc:`pyhazards_prompted`" in (root / "docs" / "source" / "pyhazards_models.rst").read_text(encoding="utf-8")
    assert "pyhazards_prompted" in (root / "docs" / "source" / "index.rst").read_text(encoding="utf-8")


def test_build_backend_names_and_missing_optional_packages(monkeypatch):
    with pytest.raises(KeyError, match="unknown prompted backend"):
        build_backend("llama4")
    import importlib.util

    if importlib.util.find_spec("openai") is None:
        with pytest.raises(ImportError, match=r"pyhazards\[prompted-openai\]"):
            build_backend("gpt_4o")
    else:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            build_backend("gpt_4o")
    if importlib.util.find_spec("google") is None or importlib.util.find_spec("google.genai") is None:
        with pytest.raises(ImportError, match=r"pyhazards\[prompted-gemini\]"):
            build_backend("gemini_2_5_pro")
    else:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            build_backend("gemini_2_5_pro")
    if importlib.util.find_spec("transformers") is None:
        with pytest.raises(ImportError, match=r"pyhazards\[prompted-hf\]"):
            build_backend("qwen2_5_vl_7b")


def test_grid_overlay_draws_cell_lines_and_numbers():
    pytest.importorskip("cv2")
    from pyhazards.prompted.geometry import draw_grid_overlay

    image = np.full((120, 160, 3), 40, dtype=np.uint8)
    overlay = draw_grid_overlay(image)
    assert overlay.shape == image.shape and overlay.dtype == np.uint8
    assert np.array_equal(image, np.full((120, 160, 3), 40, dtype=np.uint8))  # input untouched
    # white 1-pixel outlines on the 24 x 32 cell borders
    assert (overlay[24, 5] == 255).all() and (overlay[5, 32] == 255).all() and (overlay[0, 7] == 255).all()
    # the numbers are red (R = 255, G = B = 0) near each cell centre
    red = (overlay[..., 0] == 255) & (overlay[..., 1] == 0) & (overlay[..., 2] == 0)
    assert red[0:24, 0:32].any() and red[96:120, 128:160].any()
    with pytest.raises(ValueError, match="uint8"):
        draw_grid_overlay(image.astype(np.float32))
