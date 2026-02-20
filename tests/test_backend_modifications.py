import sys

# Mock modules to avoid dependencies during import
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

reportlab = types.ModuleType("reportlab")
reportlab.__path__ = []  # Make it a package
reportlab.lib = types.ModuleType("reportlab.lib")
reportlab.lib.__path__ = []
reportlab.lib.pagesizes = types.ModuleType("reportlab.lib.pagesizes")
reportlab.pdfgen = types.ModuleType("reportlab.pdfgen")
reportlab.pdfgen = types.ModuleType("reportlab.pdfgen")
reportlab.pdfgen.canvas = MagicMock()
reportlab.pdfbase = types.ModuleType("reportlab.pdfbase")
reportlab.pdfbase.__path__ = []
reportlab.pdfbase.pdfmetrics = MagicMock()
reportlab.pdfbase.ttfonts = types.ModuleType("reportlab.pdfbase.ttfonts")
reportlab.pdfbase.ttfonts.TTFont = MagicMock()

# Attributes
reportlab.lib.pagesizes.A4 = (595.27, 841.89)  # A4 size in points

sys.modules["cv2"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["reportlab"] = reportlab
sys.modules["reportlab.lib"] = reportlab.lib
sys.modules["reportlab.lib.pagesizes"] = reportlab.lib.pagesizes
sys.modules["reportlab.pdfgen"] = reportlab.pdfgen
sys.modules["reportlab.pdfbase"] = reportlab.pdfbase
# We must inject the submodule into sys.modules explicitly
sys.modules["reportlab.pdfbase.ttfonts"] = reportlab.pdfbase.ttfonts
sys.modules["pikepdf"] = MagicMock()
sys.modules["rapidocr"] = MagicMock()

from bigocrpdf.services.rapidocr_service.backend import OCRConfig, ProfessionalPDFOCR


class TestBackendModifications(unittest.TestCase):
    def setUp(self):
        # Create dummy PDF file to pass exists() check
        self.test_pdf = Path("test.pdf")
        self.test_pdf.touch()

        self.config = OCRConfig()
        # Mock page modifications:
        # Page 1: Normal
        # Page 2: Deleted
        # Page 3: Rotated 90
        # Page 4: Rotated 180 (original was 0)
        self.config.page_modifications = [
            {"page_number": 2, "deleted": True, "rotation": 0},
            {"page_number": 3, "deleted": False, "rotation": 90},
            {"page_number": 4, "deleted": False, "rotation": 180},
        ]

        # Patch _init_engine to avoid loading real OCR engine/OpenVINO during test
        with patch.object(ProfessionalPDFOCR, "_init_engine"):
            self.backend = ProfessionalPDFOCR(self.config)

        # Mock dependencies
        self.backend.extractor = MagicMock()
        self.backend._overlay_text_on_original = MagicMock()
        self.backend._finalize_output = MagicMock()
        self.backend._calculate_final_stats = MagicMock()
        self.backend._process_page_result = MagicMock(return_value=(0.95, False))

    def tearDown(self):
        if self.test_pdf.exists():
            self.test_pdf.unlink()

    def test_process_image_only_pdf_modifications(self):
        """Test chunked pipeline applies editor modifications correctly.

        Verifies:
        - Extractor is called with page_range chunks
        - Deleted pages have None img_path in work items
        - apply_final_rotation_to_pdf is invoked
        - Overlay/merge and finalize are called
        """
        input_pdf = Path("test.pdf")
        output_pdf = Path("out.pdf")

        # Mock 4 pages extracted per chunk call
        dummy_images = [Path(f"img_{i}.png") for i in range(1, 5)]
        self.backend.extractor.extract.return_value = dummy_images

        from bigocrpdf.services.rapidocr_service.rotation import PageRotation

        original_rotations = [
            PageRotation(page_number=i, original_pdf_rotation=0, mediabox=(0, 0, 595, 842))
            for i in range(1, 5)
        ]

        with (
            patch(
                "bigocrpdf.services.rapidocr_service.backend_pipeline.extract_page_rotations",
                return_value=original_rotations,
            ),
            patch(
                "bigocrpdf.services.rapidocr_service.backend_pipeline.apply_final_rotation_to_pdf",
            ) as mock_apply_final,
        ):
            self.backend._process_image_only_pdf(input_pdf, output_pdf)

        # Verify extractor was called (chunked extraction)
        self.assertTrue(self.backend.extractor.extract.called)

        # Verify _process_page_result was called for each page
        # (4 pages total — page 2 deleted gets None img_path)
        self.assertEqual(self.backend._process_page_result.call_count, 4)

        # Check that page 2 was passed with None img_path (deleted)
        all_work_items = []
        for call in self.backend._process_page_result.call_args_list:
            work_item = call[0][2]  # 3rd positional arg is work_item
            all_work_items.append(work_item)

        # Page 2 should have img_path=None (deleted)
        page2_item = next(w for w in all_work_items if w["page_num"] == 2)
        self.assertIsNone(page2_item["img_path"])

        # Pages 1, 3, 4 should have valid img_path
        for page_num in [1, 3, 4]:
            item = next(w for w in all_work_items if w["page_num"] == page_num)
            self.assertIsNotNone(item["img_path"])

        # Verify overlay was used (no standalone pages in mock)
        self.backend._overlay_text_on_original.assert_called_once()

        # Verify apply_final_rotation_to_pdf was called
        mock_apply_final.assert_called_once()

        # Verify finalize was called
        self.backend._finalize_output.assert_called_once()


if __name__ == "__main__":
    unittest.main()
