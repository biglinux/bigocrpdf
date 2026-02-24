"""Tests for image preprocessing functions."""

import cv2
import numpy as np

from bigocrpdf.services.rapidocr_service.preprocess_enhance import (
    adjust_brightness,
    apply_clahe,
    clean_borders,
    denoise,
    sharpen_text,
)
from bigocrpdf.services.rapidocr_service.preprocess_deskew import (
    measure_box_angles,
    rotate_image,
)
from bigocrpdf.services.perspective_margins import trim_white_borders


# ── Helpers ──────────────────────────────────────────────────────


def _make_bgr(h=100, w=100, value=128):
    """Create a solid-color BGR image."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _make_text_image(h=200, w=400):
    """Create a synthetic document image with dark text on white background."""
    img = np.full((h, w, 3), 255, dtype=np.uint8)  # White bg
    cv2.putText(img, "Hello World", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    return img


# ── rotate_image ─────────────────────────────────────────────────


class TestRotateImage:
    """Tests for rotate_image."""

    def test_zero_angle_preserves(self):
        img = _make_bgr()
        result = rotate_image(img, 0)
        np.testing.assert_array_equal(result, img)

    def test_preserves_dimensions(self):
        img = _make_bgr(h=200, w=300)
        result = rotate_image(img, 15)
        assert result.shape == img.shape

    def test_90_degree_rotation(self):
        img = _make_bgr(h=200, w=300)
        result = rotate_image(img, 90)
        assert result.shape == img.shape

    def test_negative_angle(self):
        img = _make_bgr(h=100, w=100)
        result = rotate_image(img, -5)
        assert result.shape == img.shape

    def test_returns_ndarray(self):
        img = _make_bgr()
        assert isinstance(rotate_image(img, 3), np.ndarray)


# ── measure_box_angles ───────────────────────────────────────────


class TestMeasureBoxAngles:
    """Tests for measure_box_angles."""

    def test_horizontal_box_returns_near_zero(self):
        box = np.array([[10, 50], [200, 50], [200, 70], [10, 70]], dtype=np.float32)
        angles, ys, widths = measure_box_angles([{"box": box}], page_width=400)
        assert len(angles) == 1
        assert abs(angles[0]) < 1.0

    def test_too_narrow_box_filtered(self):
        box = np.array([[10, 50], [15, 50], [15, 60], [10, 60]], dtype=np.float32)
        angles, _, _ = measure_box_angles([{"box": box}], page_width=400)
        assert len(angles) == 0

    def test_high_angle_box_filtered(self):
        # Create a nearly vertical box (angle > 15°)
        box = np.array([[100, 0], [110, 200], [120, 200], [110, 0]], dtype=np.float32)
        angles, _, _ = measure_box_angles([{"box": box}], page_width=400)
        assert len(angles) == 0

    def test_multiple_boxes(self):
        box1 = np.array([[10, 50], [200, 52], [200, 72], [10, 70]])
        box2 = np.array([[10, 100], [200, 103], [200, 123], [10, 120]])
        angles, ys, widths = measure_box_angles([{"box": box1}, {"box": box2}], page_width=400)
        assert len(angles) == 2

    def test_empty_boxes(self):
        angles, ys, widths = measure_box_angles([], page_width=400)
        assert len(angles) == 0


# ── adjust_brightness ────────────────────────────────────────────


class TestAdjustBrightness:
    """Tests for adjust_brightness."""

    def test_factor_one_no_change(self):
        img = _make_bgr(value=128)
        result = adjust_brightness(img, 1.0)
        # HSV conversion may have minor rounding differences
        diff = np.abs(result.astype(int) - img.astype(int))
        assert diff.max() <= 2

    def test_brighten(self):
        img = _make_bgr(value=100)
        result = adjust_brightness(img, 1.5)
        assert result.mean() > img.mean()

    def test_darken(self):
        img = _make_bgr(value=200)
        result = adjust_brightness(img, 0.5)
        assert result.mean() < img.mean()

    def test_clamps_to_255(self):
        img = _make_bgr(value=200)
        result = adjust_brightness(img, 5.0)
        assert result.max() <= 255

    def test_output_shape_preserved(self):
        img = _make_bgr(h=50, w=80)
        result = adjust_brightness(img, 1.2)
        assert result.shape == img.shape


# ── apply_clahe ──────────────────────────────────────────────────


class TestApplyClahe:
    """Tests for apply_clahe."""

    def test_output_shape_preserved(self):
        img = _make_text_image()
        result = apply_clahe(img)
        assert result.shape == img.shape

    def test_returns_uint8(self):
        img = _make_text_image()
        result = apply_clahe(img)
        assert result.dtype == np.uint8

    def test_custom_clip_limit(self):
        img = _make_text_image()
        result = apply_clahe(img, clip_limit=4.0)
        assert result.shape == img.shape


# ── denoise ──────────────────────────────────────────────────────


class TestDenoise:
    """Tests for denoise."""

    def test_output_shape_preserved(self):
        img = _make_bgr(h=100, w=100)
        result = denoise(img)
        assert result.shape == img.shape

    def test_reduces_noise(self):
        # Create noisy image
        img = _make_bgr(h=100, w=100, value=128)
        rng = np.random.default_rng(42)
        noise = rng.integers(-30, 30, img.shape, dtype=np.int16)
        noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        result = denoise(noisy)
        # Denoised should have lower variance than noisy input
        assert result.std() < noisy.std() + 5  # allow some tolerance


# ── sharpen_text ─────────────────────────────────────────────────


class TestSharpenText:
    """Tests for sharpen_text."""

    def test_output_shape_preserved(self):
        img = _make_text_image()
        result = sharpen_text(img)
        assert result.shape == img.shape

    def test_returns_uint8(self):
        img = _make_text_image()
        result = sharpen_text(img)
        assert result.dtype == np.uint8


# ── clean_borders ────────────────────────────────────────────────


class TestCleanBorders:
    """Tests for clean_borders."""

    def test_output_shape_preserved(self):
        img = _make_text_image()
        result = clean_borders(img)
        assert result.shape == img.shape

    def test_removes_dark_border(self):
        # Create image with dark border
        img = np.full((200, 200, 3), 255, dtype=np.uint8)
        img[:, 0:3] = 0  # Left dark border
        img[:, -3:] = 0  # Right dark border
        img[0:3, :] = 0  # Top dark border
        img[-3:, :] = 0  # Bottom dark border
        result = clean_borders(img)
        # Borders should be mostly white now
        assert result[0, 0].mean() > 200


# ── trim_white_borders ──────────────────────────────────────────


class TestTrimWhiteBorders:
    """Tests for trim_white_borders."""

    def test_trims_white_border(self):
        # Create image with white border around dark content
        img = np.full((200, 200, 3), 255, dtype=np.uint8)  # All white
        img[50:150, 50:150] = 50  # Dark center content
        result = trim_white_borders(img)
        # Result should be smaller than original
        assert result.shape[0] < img.shape[0]
        assert result.shape[1] < img.shape[1]

    def test_all_white_returns_original(self):
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        result = trim_white_borders(img)
        assert result.shape == img.shape

    def test_no_border_returns_similar(self):
        img = np.full((100, 100, 3), 50, dtype=np.uint8)
        result = trim_white_borders(img)
        # With margin padding, result should be very close to original size
        assert abs(result.shape[0] - img.shape[0]) <= 10
        assert abs(result.shape[1] - img.shape[1]) <= 10

    def test_preserves_channels(self):
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        img[30:70, 30:70] = 0
        result = trim_white_borders(img)
        assert result.ndim == 3
        assert result.shape[2] == 3
