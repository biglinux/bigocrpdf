"""Tests for perspective correction orchestrator."""

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
