"""
Tests for the unified rotation module.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from bigocrpdf.cli_ocr_commands import _load_pdf_page_rotations
from bigocrpdf.services.rapidocr_service import preprocess_orientation
from bigocrpdf.services.rapidocr_service.rotation import (
    PageRotation,
    apply_editor_modifications,
)


class TestPageRotation(unittest.TestCase):
    """Tests for PageRotation dataclass."""

    def test_effective_rotation_no_editor(self):
        """Original rotation only."""
        rot = PageRotation(page_number=1, original_pdf_rotation=90)
        self.assertEqual(rot.effective_rotation, 90)

    def test_effective_rotation_with_editor(self):
        """Combined original + editor rotation."""
        rot = PageRotation(page_number=1, original_pdf_rotation=90, editor_rotation=90)
        self.assertEqual(rot.effective_rotation, 180)

    def test_effective_rotation_wraps(self):
        """Rotation wraps at 360 degrees."""
        rot = PageRotation(page_number=1, original_pdf_rotation=270, editor_rotation=180)
        self.assertEqual(rot.effective_rotation, 90)  # 270 + 180 = 450 % 360 = 90

    def test_ocr_image_rotation_equals_original(self):
        """OCR rotation should match original PDF rotation."""
        rot = PageRotation(page_number=1, original_pdf_rotation=180)
        self.assertEqual(rot.ocr_image_rotation, 180)

    def test_pdf_dimensions_from_mediabox(self):
        """Dimensions from mediabox."""
        rot = PageRotation(page_number=1, mediabox=[0.0, 0.0, 612.0, 792.0])
        self.assertEqual(rot.pdf_dimensions, (612.0, 792.0))

    def test_pdf_dimensions_apply_user_unit_and_normalize_inverted_box(self):
        rot = PageRotation(
            page_number=1,
            mediabox=[612.0, 792.0, 0.0, 0.0],
            user_unit=2.0,
        )

        self.assertEqual(rot.pdf_dimensions, (1224.0, 1584.0))

    def test_pdf_dimensions_default_a4(self):
        """Default A4 when no mediabox."""
        rot = PageRotation(page_number=1)
        self.assertEqual(rot.pdf_dimensions, (595.0, 842.0))


def test_cli_rotation_loader_uses_unified_pdf_rotation_parser(monkeypatch, tmp_path) -> None:
    expected = [PageRotation(1, 90), PageRotation(2, 270)]
    monkeypatch.setattr(
        "bigocrpdf.services.rapidocr_service.rotation.extract_page_rotations",
        lambda path: expected if path == tmp_path / "input.pdf" else [],
    )

    assert _load_pdf_page_rotations(tmp_path / "input.pdf") == [90, 270]


def test_orientation_does_not_rotate_landscape_page_without_line_evidence(monkeypatch) -> None:
    image = np.full((800, 1200, 3), 255, dtype=np.uint8)
    monkeypatch.setattr(preprocess_orientation, "_hough_orientation_vote", lambda _gray: (0, None))
    monkeypatch.setattr(preprocess_orientation, "_edge_energy_vote", lambda _gray: 1)

    angle = preprocess_orientation.detect_orientation(
        image,
        SimpleNamespace(enable_orientation_detection=True),
    )

    assert angle == 0


def test_hough_orientation_accepts_opencv_4_and_5_line_shapes(monkeypatch) -> None:
    gray = np.full((200, 200), 255, dtype=np.uint8)
    segments = np.array([[10, 10, 10, 150]] * 24, dtype=np.int32)
    monkeypatch.setattr(preprocess_orientation.cv2, "Canny", lambda *_args, **_kwargs: gray)

    for lines in (segments[:, None, :], segments):
        monkeypatch.setattr(
            preprocess_orientation.cv2,
            "HoughLinesP",
            lambda *_args, lines=lines, **_kwargs: lines,
        )
        vote, angles = preprocess_orientation._hough_orientation_vote(gray)

        assert vote == 2
        assert angles is not None
        assert angles.shape == (24,)


def test_orientation_accepts_correlated_line_and_energy_evidence(monkeypatch) -> None:
    image = np.full((800, 1200, 3), 255, dtype=np.uint8)
    angles = np.array([90.0] * 12)
    monkeypatch.setattr(
        preprocess_orientation,
        "_hough_orientation_vote",
        lambda _gray: (1, angles),
    )
    monkeypatch.setattr(preprocess_orientation, "_edge_energy_vote", lambda _gray: 1)

    angle = preprocess_orientation.detect_orientation(
        image,
        SimpleNamespace(enable_orientation_detection=True),
    )

    assert angle == 90


class TestApplyEditorModifications(unittest.TestCase):
    """Tests for apply_editor_modifications function."""

    def test_no_modifications(self):
        """No editor modifications - returns unchanged."""
        rotations = [
            PageRotation(page_number=1, original_pdf_rotation=0),
            PageRotation(page_number=2, original_pdf_rotation=90),
        ]
        result = apply_editor_modifications(rotations, None)
        self.assertEqual(result[0].editor_rotation, 0)
        self.assertEqual(result[1].editor_rotation, 0)

    def test_apply_rotation(self):
        """Editor rotation is applied."""
        rotations = [
            PageRotation(page_number=1, original_pdf_rotation=0),
            PageRotation(page_number=2, original_pdf_rotation=0),
        ]
        mods = [{"page_number": 2, "rotation": 90, "deleted": False}]
        result = apply_editor_modifications(rotations, mods)
        self.assertEqual(result[0].editor_rotation, 0)
        self.assertEqual(result[1].editor_rotation, 90)

    def test_apply_deletion(self):
        """Deletion flag is applied."""
        rotations = [PageRotation(page_number=1), PageRotation(page_number=2)]
        mods = [{"page_number": 1, "deleted": True}]
        result = apply_editor_modifications(rotations, mods)
        self.assertTrue(result[0].deleted)
        self.assertFalse(result[1].deleted)

    def test_multiple_modifications(self):
        """Multiple modifications applied correctly."""
        rotations = [
            PageRotation(page_number=1),
            PageRotation(page_number=2),
            PageRotation(page_number=3),
        ]
        mods = [
            {"page_number": 1, "rotation": 180},
            {"page_number": 3, "rotation": 270, "deleted": True},
        ]
        result = apply_editor_modifications(rotations, mods)
        self.assertEqual(result[0].editor_rotation, 180)
        self.assertEqual(result[0].deleted, False)
        self.assertEqual(result[1].editor_rotation, 0)
        self.assertEqual(result[2].editor_rotation, 270)
        self.assertTrue(result[2].deleted)
