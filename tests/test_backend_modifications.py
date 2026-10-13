import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import MagicMock, patch

import pikepdf
import pytest

from bigocrpdf.services.ocr.backend import OCRConfig, ProfessionalPDFOCR
from bigocrpdf.services.ocr.config import ProcessingStats


class TestBackendModifications(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        self.test_pdf = self.test_dir / "test.pdf"
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

        # Create backend instance (no heavy OCR engine to init)
        self.backend: Any = ProfessionalPDFOCR(self.config)

        # Mock dependencies
        self.backend.extractor = MagicMock()
        self.backend._finalize_output = MagicMock()
        self.backend._calculate_final_stats = MagicMock()
        self.backend._process_page_result = MagicMock(return_value=(0.95, False))
        worker = MagicMock(threads=4)
        worker.wait_ready.return_value = 3
        worker.recognize.return_value = {"boxes": None}
        self.backend._start_ocr_worker = MagicMock(return_value=worker)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_renderer_receives_full_ocr_config(self):
        assert self.backend.renderer.config is self.config

    def test_process_image_only_pdf_modifications(self):
        """Test chunked pipeline applies editor modifications correctly.

        Verifies:
        - Extractor is called with page_range chunks
        - Deleted pages have None img_path in work items
        - apply_final_rotation_to_pdf is invoked
        - Overlay/merge and finalize are called
        """
        input_pdf = self.test_pdf
        output_pdf = self.test_dir / "out.pdf"

        dummy_images = [self.test_dir / f"img_{i}.png" for i in range(1, 5)]

        def extract_chunk(*_args, page_range, **_kwargs):
            start, end = page_range
            return dummy_images[start - 1 : end]

        self.backend.extractor.extract.side_effect = extract_chunk

        from bigocrpdf.services.ocr.rotation import PageRotation

        original_rotations = [
            PageRotation(page_number=i, original_pdf_rotation=0, mediabox=[0, 0, 595, 842])
            for i in range(1, 5)
        ]

        with (
            patch(
                "bigocrpdf.services.ocr.backend_pipeline.extract_page_rotations",
                return_value=original_rotations,
            ),
            patch(
                "bigocrpdf.services.ocr.resource_manager.adjust_chunk_size",
                return_value=4,
            ),
            patch(
                "bigocrpdf.services.ocr.backend_pipeline.apply_final_rotation_to_pdf",
            ) as mock_apply_final,
            patch("bigocrpdf.services.ocr.backend_pipeline.overlay_text_on_original") as overlay,
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
        overlay.assert_called_once()

        # Verify apply_final_rotation_to_pdf was called
        mock_apply_final.assert_called_once()

        # Verify finalize was called
        self.backend._finalize_output.assert_called_once()


def test_unread_pages_fail_the_job(tmp_path: Path) -> None:
    source = tmp_path / "in.pdf"
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page()
    pdf.save(source)
    backend = ProfessionalPDFOCR(OCRConfig())
    backend._process_image_only_pdf = MagicMock(
        return_value=ProcessingStats(
            failed_pages={4: "worker crashed", 2: "plan truncated", 7: "plan truncated"}
        )
    )

    with pytest.raises(RuntimeError) as raised:
        backend.process(source, tmp_path / "out.pdf")

    assert str(raised.value).splitlines() == [
        "Pages 2, 7 could not be read: plan truncated",
        "Page 4 could not be read: worker crashed",
    ]


def test_processing_temp_paths_do_not_follow_predictable_symlinks(tmp_path: Path) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("KEEP", encoding="utf-8")
    predictable_paths = [
        tmp_path / ".out_processing.pdf",
        tmp_path / ".out_textlayer.pdf",
    ]
    for predictable_path in predictable_paths:
        predictable_path.symlink_to(victim)

    backend = ProfessionalPDFOCR(OCRConfig())
    backend._analyze_pdf_metadata = MagicMock(return_value={"total_pages": 1})
    backend._run_chunked_ocr_pipeline = MagicMock()
    backend._post_process_pdf = MagicMock()
    backend._calculate_final_stats = MagicMock()

    backend._process_image_only_pdf(tmp_path / "input.pdf", tmp_path / "out.pdf")

    text_layer_path = backend._run_chunked_ocr_pipeline.call_args.args[1]
    worker_scratch_path = backend._run_chunked_ocr_pipeline.call_args.args[3]
    post_process_args = backend._post_process_pdf.call_args.args
    merged_path = post_process_args[2]
    assert text_layer_path not in predictable_paths
    assert merged_path not in predictable_paths
    assert text_layer_path != merged_path
    assert not text_layer_path.exists()
    assert not merged_path.exists()
    assert not worker_scratch_path.exists()
    assert victim.read_text(encoding="utf-8") == "KEEP"
    assert all(path.is_symlink() for path in predictable_paths)
