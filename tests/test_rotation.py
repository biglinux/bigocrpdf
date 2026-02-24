"""
Tests for the unified rotation module.
"""

import sys
import unittest
from unittest.mock import MagicMock

# Mock heavy dependencies before importing
sys.modules["cv2"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["pikepdf"] = MagicMock()
sys.modules["rapidocr"] = MagicMock()
sys.modules["shapely"] = MagicMock()
sys.modules["shapely.geometry"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["PIL.Image"] = MagicMock()

# Mock reportlab with proper submodule hierarchy
_reportlab = MagicMock()
sys.modules["reportlab"] = _reportlab
sys.modules["reportlab.pdfbase"] = _reportlab.pdfbase
sys.modules["reportlab.pdfbase.pdfmetrics"] = _reportlab.pdfbase.pdfmetrics
sys.modules["reportlab.pdfbase.ttfonts"] = _reportlab.pdfbase.ttfonts
sys.modules["reportlab.pdfgen"] = _reportlab.pdfgen
sys.modules["reportlab.pdfgen.canvas"] = _reportlab.pdfgen.canvas
sys.modules["reportlab.lib"] = _reportlab.lib
sys.modules["reportlab.lib.pagesizes"] = _reportlab.lib.pagesizes
sys.modules["reportlab.lib.colors"] = _reportlab.lib.colors
sys.modules["reportlab.lib.units"] = _reportlab.lib.units

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

    def test_pdf_dimensions_default_a4(self):
        """Default A4 when no mediabox."""
        rot = PageRotation(page_number=1)
        self.assertEqual(rot.pdf_dimensions, (595.0, 842.0))


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


if __name__ == "__main__":
    unittest.main()


class TestTransformOcrCoordsForRotation(unittest.TestCase):
    """Tests for transform_ocr_coords_for_rotation (pdf_extractor.py).

    This function transforms OCR coordinates from the upright image space
    to the PDF's native coordinate space, accounting for /Rotate metadata.
    """

    def setUp(self):
        """Import the function under test."""
        from bigocrpdf.services.rapidocr_service.config import OCRResult

        self.OCRResult = OCRResult

        from bigocrpdf.services.rapidocr_service.pdf_extractor import (
            transform_ocr_coords_for_rotation,
        )

        self.transform = transform_ocr_coords_for_rotation

    def _make_result(self, box):
        """Create an OCRResult with given box coordinates."""
        return self.OCRResult(text="test", box=box, confidence=0.95)

    def test_rotation_0_identity(self):
        """No rotation — coordinates are just scaled."""
        result = self._make_result([[0, 0], [100, 0], [100, 50], [0, 50]])
        transformed = self.transform(
            [result],
            ocr_img_size=(200, 100),  # OCR image W×H
            pdf_page_size=(400, 200),  # PDF W×H (2x scale)
            rotation=0,
        )
        self.assertEqual(len(transformed), 1)
        # 2x scale in both axes
        self.assertAlmostEqual(transformed[0].box[0][0], 0.0)
        self.assertAlmostEqual(transformed[0].box[0][1], 0.0)
        self.assertAlmostEqual(transformed[0].box[1][0], 200.0)
        self.assertAlmostEqual(transformed[0].box[2][1], 100.0)

    def test_rotation_90_contrato_like(self):
        """Rotation=90 (contrato.pdf case): landscape MediaBox, portrait OCR image.

        Original: MediaBox [0,0,3864,2814] (landscape), /Rotate=90
        Image extracted landscape → rotated to portrait for OCR.
        OCR coords are in portrait space and must map to landscape PDF space.
        """
        # Point at top-left of portrait image (after 90° CW rotation of landscape)
        result = self._make_result([[0, 0], [100, 0], [100, 100], [0, 100]])
        transformed = self.transform(
            [result],
            ocr_img_size=(2814, 3864),  # Portrait OCR image (W, H)
            pdf_page_size=(3864, 2814),  # Landscape PDF MediaBox (W, H)
            rotation=90,
        )
        self.assertEqual(len(transformed), 1)
        # scale_x = 3864/3864 = 1.0, scale_y = 2814/2814 = 1.0
        # new_x = p[1] * 1.0, new_y = (2814 - p[0]) * 1.0
        self.assertAlmostEqual(transformed[0].box[0][0], 0.0)
        self.assertAlmostEqual(transformed[0].box[0][1], 2814.0)

    def test_rotation_180(self):
        """Rotation=180 — coordinates are mirrored in both axes."""
        result = self._make_result([[10, 20], [110, 20], [110, 70], [10, 70]])
        transformed = self.transform(
            [result],
            ocr_img_size=(200, 100),
            pdf_page_size=(200, 100),  # Same size, no scaling
            rotation=180,
        )
        self.assertEqual(len(transformed), 1)
        # new_x = (200 - p[0]), new_y = (100 - p[1])
        self.assertAlmostEqual(transformed[0].box[0][0], 190.0)
        self.assertAlmostEqual(transformed[0].box[0][1], 80.0)

    def test_rotation_270(self):
        """Rotation=270 — coordinates transform correctly with correct scale_y."""
        # Simulate: landscape MediaBox, image rotated 270° to portrait for OCR
        result = self._make_result([[0, 0], [100, 0], [100, 50], [0, 50]])
        transformed = self.transform(
            [result],
            ocr_img_size=(2814, 3864),  # Portrait OCR (W, H)
            pdf_page_size=(3864, 2814),  # Landscape PDF (W, H)
            rotation=270,
        )
        self.assertEqual(len(transformed), 1)
        # scale_x = 3864/3864 = 1.0 (pdf_w / ocr_h)
        # scale_y = 2814/2814 = 1.0 (pdf_h / ocr_w)
        # new_x = (3864 - 0) * 1.0 = 3864
        # new_y = 0 * 1.0 = 0
        self.assertAlmostEqual(transformed[0].box[0][0], 3864.0)
        self.assertAlmostEqual(transformed[0].box[0][1], 0.0)


class TestGeometryChangeThreshold(unittest.TestCase):
    """Test that geometry change detection uses a significance threshold."""

    @staticmethod
    def _is_significant(orig_h, orig_w, proc_h, proc_w, orientation_angle=0):
        """Replicate the threshold logic from _create_text_layer_pdf."""
        total_size = orig_h + orig_w
        dim_change = abs(orig_h - proc_h) + abs(orig_w - proc_w)
        change_ratio = dim_change / total_size if total_size > 0 else 0
        return change_ratio > 0.05 or orientation_angle != 0

    def test_deskew_minor_change_not_significant(self):
        """Deskew adding small borders (~3%) should NOT trigger geometry mode."""
        # contrato.pdf: 2814x3864 → 2939x3954
        self.assertFalse(self._is_significant(3864, 2814, 3954, 2939))

    def test_perspective_correction_is_significant(self):
        """Perspective correction (~19%) SHOULD trigger geometry mode."""
        # example-need-fix-perspective1.pdf: 1920x2560 → 1495x2114
        self.assertTrue(self._is_significant(2560, 1920, 2114, 1495))

    def test_no_change_not_significant(self):
        """Identical dimensions should NOT trigger geometry mode."""
        self.assertFalse(self._is_significant(3864, 2814, 3864, 2814))

    def test_orientation_angle_is_significant(self):
        """Non-zero orientation angle should always trigger geometry mode."""
        self.assertTrue(self._is_significant(3864, 2814, 3864, 2814, orientation_angle=90))
