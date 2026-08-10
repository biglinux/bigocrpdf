"""Behaviours of third-party libraries that this project's code depends on.

These are not tests of our code. They pin the handful of library behaviours
that the geometry and rendering paths silently assume, so that a dependency
upgrade which changes one of them fails here -- loudly, in one obvious place --
instead of degrading OCR quality in a way no other test can see.

Each test names the call site it protects. If one starts failing, the fix is in
that call site, not here.
"""

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas as rl_canvas

from bigocrpdf.services.rapidocr_service.dewarp_probmap import (
    _normalise_min_area_rect_angle,
)


def _rotated_bar(angle_deg: float, size: int = 400) -> np.ndarray:
    """A single horizontal bar rotated by ``angle_deg``, as text lines appear."""
    image = np.zeros((size, size), np.uint8)
    cv2.rectangle(image, (100, size // 2 - 10), (300, size // 2 + 10), 255, -1)
    matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle_deg, 1.0)
    return cv2.warpAffine(image, matrix, (size, size))


def _largest_contour(binary: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert contours, "fixture produced no contour"
    return max(contours, key=cv2.contourArea)


class TestOpenCVReturnShapes:
    """Protects every ``.reshape(-1, 4)`` and ``[0]`` indexing of cv2 output."""

    def test_hough_lines_p_is_reshapeable_to_four_columns(self):
        """OpenCV 5 returns (N, 4) where OpenCV 4 returned (N, 1, 4).

        Call sites: preprocess_orientation._hough_orientation_vote and
        preprocess_deskew._detect_skew_hough. Both reshape, so both survive the
        change; a future layout with a different element count would not.
        """
        edges = cv2.Canny(_rotated_bar(5.0), 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=10)

        assert lines is not None
        assert np.asarray(lines).size % 4 == 0
        assert np.asarray(lines).reshape(-1, 4).shape[1] == 4

    def test_hough_lines_p_len_counts_segments(self):
        """preprocess_orientation gates on ``len(lines) <= 20``.

        That gate only means "too few segments to trust" while the first axis
        is the segment axis, which holds for both (N, 4) and (N, 1, 4).
        """
        edges = cv2.Canny(_rotated_bar(5.0), 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=10)

        assert len(lines) == np.asarray(lines).reshape(-1, 4).shape[0]

    @pytest.mark.parametrize(
        "produce",
        [
            pytest.param(
                lambda img: cv2.HoughLines(cv2.Canny(img, 50, 150), 1, np.pi / 180, 100),
                id="HoughLines",
            ),
            pytest.param(
                lambda img: cv2.goodFeaturesToTrack(img, 10, 0.01, 10),
                id="goodFeaturesToTrack",
            ),
        ],
    )
    def test_other_detectors_keep_the_middle_axis(self, produce):
        """Unlike HoughLinesP, these still return (N, 1, K)."""
        result = produce(_rotated_bar(5.0))

        assert result is not None
        assert np.asarray(result).ndim == 3
        assert np.asarray(result).shape[1] == 1

    def test_find_contours_points_keep_the_middle_axis(self):
        """Contour points stay (N, 1, 2); contour_spans and the dewarp code index them so."""
        contour = _largest_contour((_rotated_bar(0.0) > 128).astype(np.uint8))

        assert contour.ndim == 3
        assert contour.shape[1:] == (1, 2)


class TestMinAreaRectAngle:
    """Protects the two angle normalisers that turn a RotatedRect into skew."""

    def test_angle_stays_in_the_documented_range(self):
        """OpenCV 5 documents [-90, 0). The normalisers below rely on nothing else."""
        for angle_deg in (0.0, 5.0, -5.0, 30.0, 85.0):
            _, _, angle = cv2.minAreaRect(
                _largest_contour((_rotated_bar(angle_deg) > 128).astype(np.uint8))
            )
            assert -90.0 <= angle < 0.0 or angle == pytest.approx(-90.0)

    @pytest.mark.parametrize("applied_deg", [0.0, 2.0, -2.0, 5.0, -5.0, 10.0])
    def test_probmap_normaliser_recovers_the_applied_angle(self, applied_deg):
        """dewarp_probmap._normalise_min_area_rect_angle must undo the convention.

        Asserting on the normaliser rather than on the raw convention keeps this
        test meaningful if a future OpenCV changes the range again: what matters
        is that our code still recovers the true skew.
        """
        (_, _), (rect_w, rect_h), raw_angle = cv2.minAreaRect(
            _largest_contour((_rotated_bar(applied_deg) > 128).astype(np.uint8))
        )
        _, _, angle = _normalise_min_area_rect_angle(rect_w, rect_h, raw_angle)

        assert angle == pytest.approx(-applied_deg, abs=0.3)

    @pytest.mark.parametrize("applied_deg", [0.0, 2.0, -2.0, 5.0, -5.0])
    def test_inline_deskew_normaliser_recovers_the_applied_angle(self, applied_deg):
        """The other copy of this logic, in preprocess_deskew._extract_text_line_angles."""
        _, _, raw_angle = cv2.minAreaRect(
            _largest_contour((_rotated_bar(applied_deg) > 128).astype(np.uint8))
        )
        angle = raw_angle
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90

        assert angle == pytest.approx(-applied_deg, abs=0.3)


class TestReportLabTextLayer:
    """Protects renderer.TextLayerRenderer._render_text_line."""

    def test_render_mode_and_horizontal_scale_live_on_the_text_object(self):
        """ReportLab 5 dropped Canvas.setTextRenderMode; the text object keeps it.

        The invisible OCR layer depends on setTextRenderMode(3). Losing it
        silently would make every OCR layer visible on top of the scan.
        """
        text_object = rl_canvas.Canvas("/dev/null").beginText(0, 0)

        assert hasattr(text_object, "setTextRenderMode")
        assert hasattr(text_object, "setHorizScale")

    def test_string_width_is_available_for_the_oracle(self):
        assert pdfmetrics.stringWidth("Hello", "Helvetica", 12) > 0


class TestPillowTransforms:
    """Protects the degradation generator and the synthetic fixtures."""

    @pytest.mark.parametrize("name", ["PERSPECTIVE", "MESH", "AFFINE", "QUAD"])
    def test_transform_constants_are_namespaced(self, name):
        assert hasattr(Image.Transform, name)

    @pytest.mark.parametrize("name", ["BICUBIC", "BILINEAR", "LANCZOS", "NEAREST"])
    def test_resampling_constants_are_namespaced(self, name):
        assert hasattr(Image.Resampling, name)

    def test_multiline_textbbox_reports_drawn_extent(self):
        """The generator uses this to prove no glyph leaves the canvas."""
        draw = ImageDraw.Draw(Image.new("RGB", (200, 100), "white"))

        left, top, right, bottom = draw.multiline_textbbox((10, 10), "ab\ncd")

        assert right > left and bottom > top

    def test_getexif_exists_for_the_privacy_check(self):
        assert Image.new("RGB", (4, 4)).getexif() is not None


def test_numpy_polyfit_is_available():
    """dewarp_probmap and contour_dewarp fit baselines with it."""
    assert np.polyfit([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], 1)[0] == pytest.approx(2.0)
