"""Tests for bilevel image analysis and binarization."""

import numpy as np
import pytest

from bigocrpdf.services.rapidocr_service.bilevel_analysis import (
    binarize,
    is_bilevel_candidate,
)


class TestIsBilevelCandidate:
    """Tests for is_bilevel_candidate()."""

    def test_pure_black_image(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        assert is_bilevel_candidate(img) is True

    def test_pure_white_image(self):
        img = np.full((100, 100), 255, dtype=np.uint8)
        assert is_bilevel_candidate(img) is True

    def test_black_and_white_image(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:50, :] = 255
        assert is_bilevel_candidate(img) is True

    def test_grayscale_gradient_fails(self):
        """A smooth gradient has many mid-gray pixels — not bilevel."""
        img = np.tile(np.arange(256, dtype=np.uint8), (100, 1))
        assert is_bilevel_candidate(img) is False

    def test_mostly_bilevel_with_noise(self):
        """95%+ black/white with some gray noise should still pass."""
        rng = np.random.default_rng(42)
        img = np.where(rng.random((200, 200)) > 0.5, np.uint8(255), np.uint8(0))
        # Add 3% gray noise
        n_gray = int(200 * 200 * 0.03)
        coords = rng.integers(0, 200, size=(n_gray, 2))
        img[coords[:, 0], coords[:, 1]] = 128
        assert is_bilevel_candidate(img) is True

    def test_high_gray_ratio_fails(self):
        """Image with >5% gray mid-tones should fail default threshold."""
        rng = np.random.default_rng(42)
        img = np.where(rng.random((200, 200)) > 0.5, np.uint8(255), np.uint8(0))
        # Add 10% gray
        n_gray = int(200 * 200 * 0.10)
        coords = rng.integers(0, 200, size=(n_gray, 2))
        img[coords[:, 0], coords[:, 1]] = 128
        assert is_bilevel_candidate(img) is False

    def test_bgr_color_image(self):
        """Pure B&W BGR image should be detected as bilevel."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[50:, :] = 255
        assert is_bilevel_candidate(img) is True

    def test_colorful_image_fails(self):
        """Colorful image should not be bilevel."""
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)
        assert is_bilevel_candidate(img) is False

    def test_custom_threshold(self):
        """Lower threshold should accept more gray content."""
        rng = np.random.default_rng(42)
        img = np.where(rng.random((200, 200)) > 0.5, np.uint8(255), np.uint8(0))
        n_gray = int(200 * 200 * 0.08)
        coords = rng.integers(0, 200, size=(n_gray, 2))
        img[coords[:, 0], coords[:, 1]] = 128
        # Fails at 0.95 but passes at 0.90
        assert is_bilevel_candidate(img, threshold=0.95) is False
        assert is_bilevel_candidate(img, threshold=0.90) is True

    def test_empty_image(self):
        img = np.zeros((0, 0), dtype=np.uint8)
        assert is_bilevel_candidate(img) is False


class TestBinarize:
    """Tests for binarize()."""

    def test_output_is_binary(self):
        """Result should contain only 0 and 255."""
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, size=(100, 100), dtype=np.uint8)
        result = binarize(img)
        unique = set(np.unique(result))
        assert unique <= {0, 255}

    def test_output_is_single_channel(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = binarize(img)
        assert len(result.shape) == 2

    def test_output_dtype(self):
        img = np.full((50, 50), 128, dtype=np.uint8)
        result = binarize(img)
        assert result.dtype == np.uint8

    def test_preserves_dimensions(self):
        img = np.zeros((123, 456), dtype=np.uint8)
        result = binarize(img)
        assert result.shape == (123, 456)

    def test_black_stays_black(self):
        img = np.zeros((50, 50), dtype=np.uint8)
        result = binarize(img)
        assert np.all(result == 0)

    def test_white_stays_white(self):
        img = np.full((50, 50), 255, dtype=np.uint8)
        result = binarize(img)
        assert np.all(result == 255)

    def test_bimodal_separates_cleanly(self):
        """Image with two clear peaks should be separated correctly."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:50, :] = 40   # dark region
        img[50:, :] = 220  # light region
        result = binarize(img)
        assert np.all(result[:50, :] == 0)
        assert np.all(result[50:, :] == 255)

    def test_bgr_input(self):
        """BGR color image should be handled."""
        img = np.full((50, 50, 3), 200, dtype=np.uint8)
        result = binarize(img)
        assert len(result.shape) == 2
        assert set(np.unique(result)) <= {0, 255}
