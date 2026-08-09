"""Tests for bilevel_optimizer module."""

import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import pikepdf
from pikepdf import Name
from PIL import Image
from reportlab.pdfgen import canvas

from bigocrpdf.services.rapidocr_service.bilevel_optimizer import (
    _embed_ccitt,
    _embed_jbig2,
    _get_page_xobjects,
    optimize_bilevel_images,
)
from bigocrpdf.services.rapidocr_service.jbig2_encoder import jbig2enc_available


def _create_test_pdf(
    tmpdir: Path,
    num_pages: int = 2,
    width: int = 200,
    height: int = 100,
    bilevel: bool = True,
) -> Path:
    """Create a test PDF with standalone image pages."""
    pdf_path = tmpdir / "test.pdf"

    c = canvas.Canvas(str(pdf_path))
    for i in range(num_pages):
        if bilevel:
            # Black text on white background
            img = np.full((height, width, 3), 255, dtype=np.uint8)
            img[10:20, 10:50] = 0
            img[30:40, 20 + i * 10 : 80 + i * 10] = 0
        else:
            # Colorful gradient
            img = np.zeros((height, width, 3), dtype=np.uint8)
            img[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)
            img[:, :, 1] = np.linspace(255, 0, width, dtype=np.uint8)
            img[:, :, 2] = 128

        pil_img = Image.fromarray(img)
        img_path = tmpdir / f"page_{i}.png"
        pil_img.save(img_path, "PNG")

        w_pt = width / 150 * 72
        h_pt = height / 150 * 72
        c.setPageSize((w_pt, h_pt))
        c.drawImage(str(img_path), 0, 0, w_pt, h_pt)
        c.showPage()

    c.save()
    return pdf_path


class _PdfTestCase(unittest.TestCase):
    def create_test_pdf(self, **kwargs) -> Path:
        tmpdir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        return _create_test_pdf(tmpdir, **kwargs)


class TestOptimizeBilevelImages(_PdfTestCase):
    def test_nonexistent_pdf(self):
        result = optimize_bilevel_images(Path("/nonexistent.pdf"), {1: "jbig2"})
        self.assertEqual(result, 0)

    @unittest.skipUnless(jbig2enc_available(), "jbig2enc not installed")
    def test_optimizes_bilevel_pages(self):
        pdf_path = self.create_test_pdf(num_pages=2, bilevel=True)
        encodings = {1: "jbig2", 2: "jbig2"}

        before_size = pdf_path.stat().st_size
        count = optimize_bilevel_images(pdf_path, encodings)
        after_size = pdf_path.stat().st_size

        self.assertEqual(count, 2)
        self.assertLess(after_size, before_size)

    @unittest.skipUnless(jbig2enc_available(), "jbig2enc not installed")
    def test_jbig2_filter_applied(self):
        pdf_path = self.create_test_pdf(num_pages=1, bilevel=True)
        optimize_bilevel_images(pdf_path, {1: "jbig2"})

        with pikepdf.open(pdf_path) as pdf:
            page = pdf.pages[0]
            xobjects = cast(pikepdf.Dictionary, page.Resources.XObject)
            for _, obj in xobjects.items():
                if obj.get("/Subtype") == Name.Image:
                    self.assertEqual(obj.get("/Filter"), Name.JBIG2Decode)
                    self.assertEqual(int(obj.get("/BitsPerComponent", 0)), 1)

    def test_optimizes_all_pages_regardless_of_flags(self):
        """Optimizer processes all pages, not just standalone ones."""
        pdf_path = self.create_test_pdf(num_pages=2, bilevel=True)
        encodings = {1: "jbig2", 2: "jbig2"}

        count = optimize_bilevel_images(pdf_path, encodings)
        # Both pages should be optimized
        self.assertEqual(count, 2)

    def test_falls_back_to_ccitt_without_jbig2enc(self):
        pdf_path = self.create_test_pdf(num_pages=1, bilevel=True)

        with patch(
            "bigocrpdf.services.rapidocr_service.bilevel_optimizer.jbig2enc_available",
            return_value=False,
        ):
            count = optimize_bilevel_images(pdf_path, {1: "jbig2"})

        self.assertEqual(count, 1)
        with pikepdf.open(pdf_path) as pdf:
            page = pdf.pages[0]
            xobjects = cast(pikepdf.Dictionary, page.Resources.XObject)
            for _, obj in xobjects.items():
                if obj.get("/Subtype") == Name.Image:
                    self.assertEqual(obj.get("/Filter"), Name.CCITTFaxDecode)
                    self.assertEqual(int(obj.get("/BitsPerComponent", 0)), 1)

    def test_skips_non_bilevel_without_force(self):
        pdf_path = self.create_test_pdf(num_pages=1, bilevel=False)
        encodings = {}  # No bilevel encoding

        count = optimize_bilevel_images(pdf_path, encodings)
        self.assertEqual(count, 0)

    @unittest.skipUnless(jbig2enc_available(), "jbig2enc not installed")
    def test_force_bilevel_converts_color(self):
        pdf_path = self.create_test_pdf(num_pages=1, bilevel=False)
        encodings = {}

        count = optimize_bilevel_images(pdf_path, encodings, force_bilevel=True)
        self.assertEqual(count, 1)

    def test_save_failure_reports_no_optimized_images(self):
        pdf_path = self.create_test_pdf(num_pages=1, bilevel=True)

        with (
            patch(
                "bigocrpdf.services.rapidocr_service.bilevel_optimizer.jbig2enc_available",
                return_value=False,
            ),
            patch.object(pikepdf.Pdf, "save", side_effect=OSError("disk full")),
        ):
            count = optimize_bilevel_images(pdf_path, {1: "jbig2"})

        self.assertEqual(count, 0)


class TestGetPageXobjects(_PdfTestCase):
    def test_returns_xobjects_for_pdf_page(self):
        pdf_path = self.create_test_pdf(num_pages=1)
        with pikepdf.open(pdf_path) as pdf:
            xobjects = _get_page_xobjects(pdf.pages[0])
            assert xobjects is not None
            self.assertGreater(len(xobjects), 0)


class TestEmbedFunctions(unittest.TestCase):
    def test_embed_jbig2(self):
        pdf = pikepdf.new()
        page = pdf.add_blank_page(page_size=(100, 100))
        page["/Resources"] = pikepdf.Dictionary({"/XObject": pikepdf.Dictionary()})
        xobjects = cast(pikepdf.Dictionary, page.Resources.XObject)

        # Create a dummy image first
        dummy_stream = pikepdf.Stream(pdf, b"\xff" * 100)
        dummy_stream["/Subtype"] = Name.Image
        dummy_stream["/Width"] = 10
        dummy_stream["/Height"] = 10
        xobjects["/Im0"] = dummy_stream

        _embed_jbig2(pdf, xobjects, "/Im0", b"\x00\x01\x02", b"\x03\x04", 10, 10)

        obj = xobjects["/Im0"]
        self.assertEqual(obj.get("/Filter"), Name.JBIG2Decode)
        self.assertEqual(int(obj.get("/Width", 0)), 10)
        self.assertEqual(int(obj.get("/BitsPerComponent", 0)), 1)

    def test_embed_ccitt(self):
        pdf = pikepdf.new()
        page = pdf.add_blank_page(page_size=(100, 100))
        page["/Resources"] = pikepdf.Dictionary({"/XObject": pikepdf.Dictionary()})
        xobjects = cast(pikepdf.Dictionary, page.Resources.XObject)

        dummy_stream = pikepdf.Stream(pdf, b"\xff" * 100)
        dummy_stream["/Subtype"] = Name.Image
        xobjects["/Im0"] = dummy_stream

        _embed_ccitt(pdf, xobjects, "/Im0", b"\x00\x01\x02", 200, 100)

        obj = xobjects["/Im0"]
        self.assertEqual(obj.get("/Filter"), Name.CCITTFaxDecode)
        self.assertEqual(int(obj.get("/Width", 0)), 200)
        self.assertEqual(int(obj.get("/Height", 0)), 100)
        decode_parms = obj.get("/DecodeParms")
        assert decode_parms is not None
        self.assertEqual(int(decode_parms.get("/K", 0)), -1)
        self.assertEqual(int(decode_parms.get("/Columns", 0)), 200)
