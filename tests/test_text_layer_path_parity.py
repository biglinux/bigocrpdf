"""The two OCR placement implementations must not drift apart.

``renderer.create_text_layer`` (path A, ReportLab canvas, image-only PDFs) and
``pdf_assembly._build_text_boxes`` (path B, raw content streams, mixed-content
and embedded-image PDFs) each compute the same pixel-to-point mapping with
their own copy of the arithmetic. Nothing links them, so an edit to one is
invisible to the other -- and a document routed through the other pipeline
would silently get differently-placed text.

Two kinds of assertion below:

* contract  -- for axis-aligned quads the two are algebraically identical, so
  any difference is a defect in one of them;
* characterisation -- for skewed quads they genuinely differ, by an amount
  derived in closed form here so the divergence is understood rather than
  merely recorded.
"""

import math
import random
from pathlib import Path

import pytest
from reportlab.pdfbase import pdfmetrics

from bigocrpdf.constants import FONT_SIZE_SCALE_FACTOR
from bigocrpdf.services.rapidocr_service import pdf_assembly
from bigocrpdf.services.rapidocr_service import renderer as renderer_module
from bigocrpdf.services.rapidocr_service.config import OCRConfig, OCRResult
from bigocrpdf.services.rapidocr_service.renderer import TextBox, TextLayerRenderer

SEED = 20260809

PAGE = (612.0, 792.0)
IMAGE = (2550, 3300)
SCALE_X = PAGE[0] / IMAGE[0]
SCALE_Y = PAGE[1] / IMAGE[1]


def _renderer() -> TextLayerRenderer:
    return TextLayerRenderer(OCRConfig(font_base_path=Path("/nonexistent-font-dir")))


def _place_a(box: list[list[float]], text: str = "WORD") -> tuple[float, ...]:
    layer = _renderer().create_text_layer(
        [OCRResult(text=text, box=box, confidence=0.9)], IMAGE[0], IMAGE[1], page_size_pts=PAGE
    )
    placed = layer.boxes[0]
    return (placed.x, placed.y, placed.width, placed.height, placed.font_size)


def _place_b(box: list[list[float]], text: str = "WORD") -> tuple[float, ...]:
    placed = pdf_assembly._build_text_boxes(
        [OCRResult(text=text, box=box, confidence=0.9)],
        img_x=0.0,
        img_y=0.0,
        img_height=PAGE[1],
        scale_x=SCALE_X,
        scale_y=SCALE_Y,
    )[0]
    return (
        placed["x"],
        placed["y"],
        placed["width"],
        placed["height"],
        placed["font_size"],
    )


def _axis_aligned(left, top, right, bottom) -> list[list[float]]:
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def _skewed(left, top, right, height, shear) -> list[list[float]]:
    """A quad sheared along the reading direction by ``shear`` pixels."""
    return [
        [left, top],
        [right, top + shear],
        [right, top + shear + height],
        [left, top + height],
    ]


class TestAxisAlignedParity:
    """For rectangles the two formulas reduce to the same expression."""

    @pytest.mark.parametrize(
        "box",
        [
            pytest.param(_axis_aligned(0, 0, 100, 40), id="origin"),
            pytest.param(_axis_aligned(100, 200, 900, 260), id="typical-line"),
            pytest.param(_axis_aligned(2400, 3200, 2550, 3300), id="bottom-right"),
            pytest.param(_axis_aligned(5, 5, 6, 6), id="tiny"),
        ],
    )
    def test_placements_agree_exactly(self, box):
        assert _place_a(box) == pytest.approx(_place_b(box), abs=1e-9)

    def test_placements_agree_over_a_random_sweep(self):
        rng = random.Random(SEED)

        for _ in range(200):
            left = rng.uniform(0, IMAGE[0] - 10)
            right = rng.uniform(left + 1, IMAGE[0])
            top = rng.uniform(0, IMAGE[1] - 10)
            bottom = rng.uniform(top + 1, IMAGE[1])
            box = _axis_aligned(left, top, right, bottom)

            assert _place_a(box) == pytest.approx(_place_b(box), abs=1e-9), f"seed={SEED}"


class TestSkewedQuadDivergence:
    """Path A uses the mean of the bottom edge; path B uses its maximum."""

    @pytest.mark.parametrize("shear", [10.0, 40.0, 120.0])
    def test_font_size_diverges_because_b_measures_the_bounding_box(self, shear):
        """B's glyph height absorbs the shear; A's does not.

        This is the more consequential of the two divergences: B inflates the
        font size by the full shear, which widens the selection rectangle and
        changes the horizontal stretch, whereas A measures the quad's own side
        edges and is unaffected.
        """
        box = _skewed(100.0, 200.0, 900.0, 60.0, shear)

        size_a = _place_a(box)[4]
        size_b = _place_b(box)[4]

        assert size_a == pytest.approx(60.0 * SCALE_Y * FONT_SIZE_SCALE_FACTOR, rel=1e-6)
        assert size_b == pytest.approx((60.0 + shear) * SCALE_Y * FONT_SIZE_SCALE_FACTOR, rel=1e-6)

    @pytest.mark.parametrize("shear", [10.0, 40.0, 120.0])
    def test_divergence_matches_the_closed_form(self, shear):
        """The gap has two sources, and both are derived rather than measured.

        A's baseline is the mean of the bottom corners; B's is the lowest one,
        which is shear/2 px below that. On top of that the descent term uses
        each path's own font size, and B's is inflated by the whole shear. So

            y_a - y_b = shear * scale_y * (0.5 - 0.207 * FONT_SIZE_SCALE_FACTOR)

        Deriving it proves the divergence is understood rather than merely
        recorded, and pins both causes at once.
        """
        box = _skewed(100.0, 200.0, 900.0, 60.0, shear)

        y_a = _place_a(box)[1]
        y_b = _place_b(box)[1]

        expected = shear * SCALE_Y * (0.5 - 0.207 * FONT_SIZE_SCALE_FACTOR)
        assert (y_a - y_b) == pytest.approx(expected, abs=1e-6)

    def test_path_a_sits_on_the_true_baseline_midpoint(self):
        """A is the more accurate of the two, which is why B should adopt it."""
        shear = 40.0
        box = _skewed(100.0, 200.0, 900.0, 60.0, shear)
        bottom_left_y = 200.0 + 60.0
        bottom_right_y = 200.0 + shear + 60.0
        true_mid_px = (bottom_left_y + bottom_right_y) / 2.0

        font_size = 60.0 * SCALE_Y * FONT_SIZE_SCALE_FACTOR
        expected = PAGE[1] - true_mid_px * SCALE_Y + 0.207 * font_size

        assert _place_a(box)[1] == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize("angle_deg", [0.5, 1.0, 2.0, 3.0])
    def test_realistic_residual_skew_keeps_both_paths_on_one_line(self, angle_deg):
        """Below the baseline-snapping threshold the two still group alike.

        Residual skew after deskew is a few degrees at most; within that range
        the divergence must stay under the 0.35 * height clustering threshold,
        or the two pipelines would split lines differently.
        """
        width_px, height_px = 800.0, 60.0
        shear = width_px * math.tan(math.radians(angle_deg))
        box = _skewed(100.0, 200.0, 100.0 + width_px, height_px, shear)

        divergence = abs(_place_a(box)[1] - _place_b(box)[1])
        height_pts = height_px * SCALE_Y

        assert divergence <= 0.35 * height_pts


class TestSnapBaselinesParity:
    """The two copies of the clustering code must behave identically."""

    @staticmethod
    def _inputs(ys: list[float], height: float = 12.0):
        boxes_a = [
            TextBox(text=f"W{i}", x=i * 50.0, y=y, width=40.0, height=height, font_size=height)
            for i, y in enumerate(ys)
        ]
        boxes_b = [
            {
                "text": f"W{i}",
                "x": i * 50.0,
                "y": y,
                "width": 40.0,
                "height": height,
                "font_size": height,
            }
            for i, y in enumerate(ys)
        ]
        return boxes_a, boxes_b

    @pytest.mark.parametrize(
        "ys",
        [
            pytest.param([700.0, 700.0, 700.0], id="identical"),
            pytest.param([700.0, 702.0, 660.0], id="one-cluster-plus-one"),
            pytest.param([700.0, 701.0, 702.0, 703.0], id="even-sized-cluster"),
            pytest.param([700.0, 688.0, 676.0], id="regular-lines"),
        ],
    )
    def test_both_implementations_snap_the_same(self, ys):
        boxes_a, boxes_b = self._inputs(ys)

        TextLayerRenderer._snap_baselines(boxes_a)
        pdf_assembly._snap_baselines(boxes_b)

        assert [box.y for box in boxes_a] == [box["y"] for box in boxes_b]

    def test_path_b_tolerates_an_empty_list(self):
        """Path A guards; path B indexes sorted_boxes[0] and would raise.

        Today only the caller's own emptiness check protects it, which is a
        guard one refactor away from disappearing.
        """
        pdf_assembly._snap_baselines([])


class TestSharedConstants:
    def test_both_paths_use_the_same_descent_fraction(self):
        """0.207 is a literal in pdf_assembly and a local in renderer."""
        source_a = Path(renderer_module.__file__).read_text(encoding="utf-8")
        source_b = Path(pdf_assembly.__file__).read_text(encoding="utf-8")

        assert "0.207" in source_a and "0.207" in source_b

    def test_both_paths_use_the_same_font_size_factor(self):
        placement_box = _axis_aligned(0, 0, 200, 100)

        assert _place_a(placement_box)[4] == pytest.approx(_place_b(placement_box)[4])
        assert FONT_SIZE_SCALE_FACTOR > 0


class TestHorizontalScaleClamp:
    def test_path_b_clamps_the_stretch(self):
        """pdf_assembly bounds Tz to [30, 300]."""
        boxes = [
            {
                "text": "W",
                "x": 0.0,
                "y": 100.0,
                "width": 5000.0,
                "height": 12.0,
                "font_size": 12.0,
            }
        ]

        commands = " ".join(pdf_assembly._emit_line_commands(boxes))
        scales = [float(value) for value in _tz_values(commands)]

        assert scales and all(scale <= 300.0 for scale in scales)

    def test_path_a_clamps_the_same_way(self):
        """Fixed: both paths now share MIN_HORIZ_SCALE / MAX_HORIZ_SCALE.

        An unclamped stretch made the invisible run cover a large part of the
        page, so text selection picked up words that were nowhere near the
        pointer.
        """
        natural = pdfmetrics.stringWidth("W", "Helvetica", 12.0)

        wide = renderer_module._line_horizontal_scale("W", natural * 50, "Helvetica", 12.0)
        narrow = renderer_module._line_horizontal_scale("W", natural / 50, "Helvetica", 12.0)

        assert wide == renderer_module.MAX_HORIZ_SCALE
        assert narrow == renderer_module.MIN_HORIZ_SCALE

    def test_both_paths_share_one_set_of_bounds(self):
        """pdf_assembly imports them, so the two cannot drift apart again."""
        assert pdf_assembly.MIN_HORIZ_SCALE is renderer_module.MIN_HORIZ_SCALE
        assert pdf_assembly.MAX_HORIZ_SCALE is renderer_module.MAX_HORIZ_SCALE


def _tz_values(text: str) -> list[str]:
    import re

    return re.findall(r"([\d.]+) Tz", text)
