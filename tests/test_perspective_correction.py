"""Tests for perspective correction orchestrator."""

from unittest.mock import patch

import numpy as np

from bigocrpdf.services.perspective_correction import PerspectiveCorrector


class TestPerspectiveCorrector:
    """Tests for PerspectiveCorrector."""

    def test_init_default(self):
        pc = PerspectiveCorrector()
        assert pc.skew_threshold == 0.5
        assert pc.variance_threshold == 0.3
        assert pc.skip_skew is False

    def test_init_custom(self):
        pc = PerspectiveCorrector(skew_threshold=1.0, variance_threshold=0.5, skip_skew=True)
        assert pc.skew_threshold == 1.0
        assert pc.skip_skew is True

    def test_call_returns_ndarray(self):
        pc = PerspectiveCorrector()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = pc(img)
        assert isinstance(result, np.ndarray)

    def test_small_image_returns_input(self):
        pc = PerspectiveCorrector()
        # Very small image should pass through without errors
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        result = pc(img)
        assert result.shape[0] > 0

    def test_validate_correction_rejects_degraded(self):
        # Original with clear text-like horizontal structure
        original = np.full((100, 100, 3), 255, dtype=np.uint8)
        # Draw horizontal "text lines" (dark rows)
        for row in [20, 40, 60, 80]:
            original[row : row + 3, 10:90] = 0
        # Corrected: uniform gray (no text structure)
        corrected = np.full((100, 100, 3), 200, dtype=np.uint8)
        # Sharpness of original > corrected, so validation should reject
        assert PerspectiveCorrector._validate_correction(original, corrected, "test") is False

    def test_validate_correction_accepts_similar(self):
        original = np.full((100, 100, 3), 128, dtype=np.uint8)
        corrected = original.copy()
        assert PerspectiveCorrector._validate_correction(original, corrected, "test") is True

    def test_contour_correction_rejects_internal_document_frame(self):
        image = np.zeros((1000, 800, 3), dtype=np.uint8)
        contour = np.array(
            [[40, 200], [760, 200], [760, 940], [40, 940]],
            dtype=np.float32,
        )

        with (
            patch(
                "bigocrpdf.services.perspective_correction.detect_document_contour",
                return_value=contour,
            ),
            patch("bigocrpdf.services.perspective_correction.four_point_transform") as transform,
        ):
            result = PerspectiveCorrector()._try_contour_correction(image)

        assert result is image
        transform.assert_not_called()

    def test_contour_correction_uses_the_detected_contour_for_validation(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        contour = np.array([[5, 5], [95, 5], [90, 95], [10, 95]], dtype=np.float32)

        with (
            patch(
                "bigocrpdf.services.perspective_correction.detect_document_contour",
                return_value=contour,
            ) as detect,
            patch(
                "bigocrpdf.services.perspective_correction._contour_needs_perspective_correction",
                return_value=False,
            ) as needs_correction,
        ):
            result = PerspectiveCorrector()._try_contour_correction(image)

        assert result is image
        detect.assert_called_once_with(image)
        needs_correction.assert_called_once()
        np.testing.assert_array_equal(needs_correction.call_args.args[0], contour)


def test_validate_correction_rejects_validation_errors():
    original = np.zeros((20, 20, 3), dtype=np.uint8)
    invalid = np.empty((0, 0, 3), dtype=np.uint8)

    assert PerspectiveCorrector._validate_correction(original, invalid, "test") is False
