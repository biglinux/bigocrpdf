"""Invariants of the OCR-pixel to PDF-point mapping.

These run in microseconds and need no PDF: they hold the algebra of
``TextLayerRenderer.create_text_layer`` and ``_snap_baselines`` in place.
Where test_text_layer_placement proves the mapping is right end to end, this
file pins *why*, so a failure says which property broke.
"""

import math
from pathlib import Path

import pytest

from bigocrpdf.constants import FONT_SIZE_SCALE_FACTOR, MAX_FONT_SIZE, MIN_FONT_SIZE
from bigocrpdf.services.rapidocr_service.config import OCRConfig, OCRResult
from bigocrpdf.services.rapidocr_service.renderer import (
    TextBox,
    TextLayerRenderer,
    _line_horizontal_scale,
    _line_text_with_spacing,
    _normalize_ocr_quadrilateral,
)

# Every randomised loop echoes this in its assertion message so a failure is
# reproducible without rerunning.
SEED = 20260809

PAGE = (612.0, 792.0)
IMAGE = (2550, 3300)


@pytest.fixture
def renderer() -> TextLayerRenderer:
    """A renderer pinned to Helvetica, so metrics are the built-in ones."""
    return TextLayerRenderer(OCRConfig(font_base_path=Path("/nonexistent-font-dir")))


def _rect(left: float, top: float, right: float, bottom: float) -> list[list[float]]:
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def _layer(renderer: TextLayerRenderer, boxes, page=PAGE, image=IMAGE, text="WORD"):
    results = [OCRResult(text=text, box=box, confidence=0.9) for box in boxes]
    return renderer.create_text_layer(results, image[0], image[1], page_size_pts=page)


class TestScaling:
    def test_left_edge_at_pixel_zero_maps_to_x_zero(self, renderer):
        layer = _layer(renderer, [_rect(0, 100, 200, 160)])

        assert layer.boxes[0].x == 0.0

    def test_bottom_edge_at_image_height_leaves_only_the_descent(self, renderer):
        """The Y-flip must be exact at the boundary, descent aside."""
        layer = _layer(renderer, [_rect(0, IMAGE[1] - 60, 200, IMAGE[1])])
        box = layer.boxes[0]

        assert box.y == pytest.approx(0.207 * box.font_size, abs=1e-9)

    def test_horizontal_scaling_is_independent_of_vertical(self, renderer):
        """Doubling page width doubles x and leaves y untouched.

        Catches the averaged px_to_pt (used for font size) leaking into
        placement, which square-ish real pages would hide.
        """
        box = [_rect(100, 200, 400, 260)]
        narrow = _layer(renderer, box, page=(300.0, 792.0)).boxes[0]
        wide = _layer(renderer, box, page=(600.0, 792.0)).boxes[0]

        assert wide.x == pytest.approx(2 * narrow.x)
        assert wide.width == pytest.approx(2 * narrow.width)
        assert wide.y == pytest.approx(narrow.y, abs=0.5)

    def test_scaling_a_quad_scales_the_mapped_offsets(self, renderer):
        small = _layer(renderer, [_rect(100, 200, 300, 260)]).boxes[0]
        large = _layer(renderer, [_rect(200, 400, 600, 520)]).boxes[0]

        assert large.x == pytest.approx(2 * small.x)
        assert large.width == pytest.approx(2 * small.width)

    def test_larger_pixel_y_maps_to_smaller_pdf_y(self, renderer):
        """PDF space is bottom-up; image space is top-down."""
        heights = [
            _layer(renderer, [_rect(0, top, 200, top + 60)]).boxes[0].y
            for top in (100, 500, 1500, 3000)
        ]

        assert heights == sorted(heights, reverse=True)

    def test_dpi_fallback_matches_an_explicit_page_size(self, renderer):
        """Without a page size the mapping comes from config.dpi."""
        box = [_rect(100, 200, 400, 260)]
        implied = (IMAGE[0] * 72.0 / renderer.config.dpi, IMAGE[1] * 72.0 / renderer.config.dpi)

        from_dpi = _layer(renderer, box, page=None).boxes[0]
        from_size = _layer(renderer, box, page=implied).boxes[0]

        assert from_dpi.x == pytest.approx(from_size.x)
        assert from_dpi.y == pytest.approx(from_size.y)


class TestFontSize:
    def test_font_size_follows_box_height(self, renderer):
        layer = _layer(renderer, [_rect(0, 0, 200, 100)])
        expected = 100 * (PAGE[1] / IMAGE[1]) * FONT_SIZE_SCALE_FACTOR

        assert layer.boxes[0].font_size == pytest.approx(expected)

    def test_font_size_is_clamped_at_both_ends(self, renderer):
        tiny = _layer(renderer, [_rect(0, 0, 200, 1)]).boxes[0]
        huge = _layer(renderer, [_rect(0, 0, 200, IMAGE[1])]).boxes[0]

        assert tiny.font_size == MIN_FONT_SIZE
        assert huge.font_size == MAX_FONT_SIZE

    def test_the_descent_uses_the_clamped_size(self, renderer):
        """Locks current behaviour: changing it must be a deliberate choice."""
        box = _layer(renderer, [_rect(0, IMAGE[1] - 1, 200, IMAGE[1])]).boxes[0]

        assert box.font_size == MIN_FONT_SIZE
        assert box.y == pytest.approx(0.207 * MIN_FONT_SIZE, abs=1e-9)

    def test_a_skewed_quad_uses_the_mean_side_height(self, renderer):
        """Height comes from the two sides, so a shear must not inflate it."""
        skewed = [[0.0, 100.0], [200.0, 110.0], [200.0, 170.0], [0.0, 160.0]]

        layer = _layer(renderer, [skewed])

        expected_height = 60 * (PAGE[1] / IMAGE[1])
        assert layer.boxes[0].height == pytest.approx(expected_height, rel=1e-6)


class TestQuadrilateralNormalisation:
    def test_legacy_flat_boxes_are_accepted(self):
        assert _normalize_ocr_quadrilateral([10, 20, 30, 40]) == (
            (10.0, 20.0),
            (30.0, 20.0),
            (30.0, 40.0),
            (10.0, 40.0),
        )

    @pytest.mark.parametrize(
        "box",
        [
            pytest.param(None, id="none"),
            pytest.param([], id="empty"),
            pytest.param([[0, 0], [1, 0]], id="too-short"),
            pytest.param([[0, 0], [1, 0], [1, 1], [0, float("nan")]], id="nan"),
            pytest.param([[0, 0], [1, 0], [1, 1], [0, float("inf")]], id="inf"),
            pytest.param([[0, 0], [1, 0], [1, 1], ["x", 1]], id="non-numeric"),
        ],
    )
    def test_degenerate_boxes_are_rejected(self, box):
        assert _normalize_ocr_quadrilateral(box) is None

    def test_a_zero_height_box_does_not_divide_by_zero(self, renderer):
        layer = _layer(renderer, [_rect(0, 100, 200, 100)])

        assert layer.boxes[0].font_size == MIN_FONT_SIZE

    def test_empty_text_produces_no_box(self, renderer):
        assert _layer(renderer, [_rect(0, 0, 100, 50)], text="   ").boxes == []


class TestSnapBaselines:
    @staticmethod
    def _boxes(ys: list[float], height: float = 12.0) -> list[TextBox]:
        return [
            TextBox(text=f"W{i}", x=float(i * 50), y=y, width=40.0, height=height, font_size=height)
            for i, y in enumerate(ys)
        ]

    def test_empty_input_is_tolerated(self):
        TextLayerRenderer._snap_baselines([])

    def test_snapping_is_idempotent(self):
        boxes = self._boxes([700.0, 701.5, 699.0, 640.0])
        TextLayerRenderer._snap_baselines(boxes)
        once = [box.y for box in boxes]
        TextLayerRenderer._snap_baselines(boxes)

        assert [box.y for box in boxes] == once

    def test_no_box_leapfrogs_another(self):
        boxes = self._boxes([700.0, 698.0, 660.0, 658.0, 600.0])
        before = sorted(range(len(boxes)), key=lambda i: -boxes[i].y)
        TextLayerRenderer._snap_baselines(boxes)
        after_ys = [boxes[i].y for i in before]

        assert after_ys == sorted(after_ys, reverse=True)

    def test_close_boxes_share_a_baseline(self):
        """Within 35% of the smaller height they are one visual line."""
        boxes = self._boxes([700.0, 700.0 + 12.0 * 0.3])
        TextLayerRenderer._snap_baselines(boxes)

        assert boxes[0].y == boxes[1].y

    def test_distant_boxes_keep_their_own(self):
        boxes = self._boxes([700.0, 700.0 - 12.0])
        TextLayerRenderer._snap_baselines(boxes)

        assert boxes[0].y != boxes[1].y

    def test_snapping_never_creates_new_baselines(self):
        import random

        rng = random.Random(SEED)
        ys = [rng.uniform(100.0, 700.0) for _ in range(40)]
        boxes = self._boxes(ys)
        before = len({box.y for box in boxes})

        TextLayerRenderer._snap_baselines(boxes)

        assert len({box.y for box in boxes}) <= before, f"seed={SEED}"


class TestLineAssembly:
    def test_spacing_grows_with_the_gap(self):
        def spaces(gap: float) -> int:
            boxes = [
                TextBox(text="A", x=0.0, y=0.0, width=20.0, height=12.0, font_size=12.0),
                TextBox(text="B", x=20.0 + gap, y=0.0, width=20.0, height=12.0, font_size=12.0),
            ]
            return _line_text_with_spacing(boxes, "Helvetica", 12.0).count(" ")

        assert spaces(0.0) <= spaces(20.0) <= spaces(100.0)

    def test_adjacent_boxes_still_get_one_space(self):
        boxes = [
            TextBox(text="A", x=0.0, y=0.0, width=20.0, height=12.0, font_size=12.0),
            TextBox(text="B", x=10.0, y=0.0, width=20.0, height=12.0, font_size=12.0),
        ]

        assert " " in _line_text_with_spacing(boxes, "Helvetica", 12.0)

    def test_horizontal_scale_is_finite_and_positive(self):
        for width in (0.0, 1.0, 50.0, 5000.0):
            scale = _line_horizontal_scale("WORD", width, "Helvetica", 12.0)
            assert math.isfinite(scale) and scale > 0.0

    def test_scale_is_unity_when_the_box_matches_the_string(self):
        from reportlab.pdfbase import pdfmetrics

        natural = pdfmetrics.stringWidth("WORD", "Helvetica", 12.0)

        assert _line_horizontal_scale("WORD", natural, "Helvetica", 12.0) == pytest.approx(100.0)
