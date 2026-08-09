"""Tests for PDF text escaping and assembly utilities."""

import subprocess
from pathlib import Path

import pikepdf
from reportlab.pdfgen import canvas as reportlab_canvas

from bigocrpdf.services.rapidocr_service.config import OCRConfig, OCRResult
from bigocrpdf.services.rapidocr_service.pdf_assembly import (
    _filter_invisible_text_ops,
    _pdf_text_operand,
    append_text_to_page,
    create_text_layer_commands,
    escape_pdf_text,
    strip_invisible_text,
)
from bigocrpdf.services.rapidocr_service.pdf_extractor import (
    _scan_page_text_ops,
    has_native_text,
    page_has_ocr_text,
)
from bigocrpdf.services.rapidocr_service.renderer import TextLayerRenderer
from bigocrpdf.utils.pdf_utils import set_root_page_layout


class TestEscapePdfText:
    """Tests for escape_pdf_text."""

    def test_plain_ascii(self):
        assert escape_pdf_text("Hello World") == "Hello World"

    def test_escapes_backslash(self):
        assert escape_pdf_text("a\\b") == "a\\\\b"

    def test_escapes_parentheses(self):
        assert escape_pdf_text("(test)") == "\\(test\\)"

    def test_em_dash_replaced(self):
        assert escape_pdf_text("word\u2014word") == "word-word"

    def test_en_dash_replaced(self):
        assert escape_pdf_text("1\u20135") == "1-5"

    def test_smart_quotes_replaced(self):
        text = "\u201cHello\u201d \u2018world\u2019"
        assert escape_pdf_text(text) == "\"Hello\" 'world'"

    def test_ellipsis_replaced(self):
        assert escape_pdf_text("wait\u2026") == "wait..."

    def test_bullet_replaced(self):
        assert escape_pdf_text("\u2022 item") == "* item"

    def test_fi_ligature(self):
        assert escape_pdf_text("\ufb01nd") == "find"

    def test_fl_ligature(self):
        assert escape_pdf_text("\ufb02ow") == "flow"

    def test_zero_width_chars_removed(self):
        assert escape_pdf_text("a\u200bb\u200cc\ufeffd") == "abcd"

    def test_non_latin1_is_preserved(self):
        assert escape_pdf_text("\u4e16") == "\u4e16"

    def test_pdf_text_operand_encodes_unicode_as_utf16_hex(self):
        operand = _pdf_text_operand("ação 中文")
        assert operand.startswith("<FEFF")
        assert "?" not in operand

    def test_combined_escaping(self):
        text = "(test\\) \u2014 \u201cquote\u201d"
        result = escape_pdf_text(text)
        assert "\\\\" in result
        assert "\\(" in result
        assert "\\)" in result
        assert "-" in result
        assert '"' in result

    def test_empty_string(self):
        assert escape_pdf_text("") == ""

    def test_only_latin1_passes(self):
        text = "\xe9\xe0\xfc"  # é à ü
        assert escape_pdf_text(text) == text


def test_text_scanner_tracks_render_mode_changes_within_one_block() -> None:
    operations = [
        ([], "BT"),
        ([3], "Tr"),
        (["hidden"], "Tj"),
        ([0], "Tr"),
        (["visible"], "Tj"),
        ([], "ET"),
    ]

    assert _scan_page_text_ops(operations) == (True, True)


def test_invisible_text_filter_preserves_mixed_render_mode_block() -> None:
    operations = [
        ([], "BT"),
        ([3], "Tr"),
        (["hidden"], "Tj"),
        ([0], "Tr"),
        (["visible"], "Tj"),
        ([], "ET"),
    ]

    filtered, removed = _filter_invisible_text_ops(operations)

    assert filtered == operations
    assert removed == 0


def test_invisible_text_filter_preserves_mixed_render_mode_q_group() -> None:
    operations = [
        ([], "q"),
        ([], "BT"),
        ([3], "Tr"),
        (["hidden"], "Tj"),
        ([0], "Tr"),
        (["visible"], "Tj"),
        ([], "ET"),
        ([], "Q"),
    ]

    filtered, removed = _filter_invisible_text_ops(operations)

    assert filtered == operations
    assert removed == 0


def test_reportlab_text_layer_uses_render_mode_without_alpha_states(tmp_path: Path) -> None:
    pdf_path = tmp_path / "reportlab_text_layer.pdf"
    pdf_canvas = reportlab_canvas.Canvas(
        str(pdf_path),
        pagesize=(200, 100),
    )
    renderer = TextLayerRenderer(OCRConfig(font_base_path=tmp_path))
    renderer.render(
        pdf_canvas,
        [OCRResult("searchable", [[10, 10], [150, 10], [150, 30], [10, 30]], 1.0)],
        (200, 100),
        page_size_pts=(200, 100),
    )
    pdf_canvas.showPage()
    pdf_canvas.save()

    with pikepdf.open(pdf_path) as pdf:
        page = pdf.pages[0]
        contents = page.Contents.read_bytes()
        assert b"3 Tr" in contents
        assert b" gs" not in contents
        assert "/ExtGState" not in page.Resources


def test_appended_text_layer_extracts_unicode_with_pdftotext(tmp_path: Path) -> None:
    sample = "ação João nº 中文 العربية Ελληνικά हिन्दी ภาษาไทย"
    pdf_path = tmp_path / "unicode_text_layer.pdf"
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(500, 200))
    ocr_result = OCRResult(sample, [[20, 20], [480, 20], [480, 60], [20, 60]], 1.0)

    commands = create_text_layer_commands(
        [ocr_result],
        img_x=0,
        img_y=0,
        img_width=500,
        img_height=200,
        scale_x=1,
        scale_y=1,
    )
    append_text_to_page(pdf, page, commands)
    pdf.save(pdf_path)

    result = subprocess.run(
        ["pdftotext", str(pdf_path), "-"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    extracted = result.stdout
    assert "?" not in extracted
    assert "�" not in extracted
    assert "□" not in extracted
    for expected in ["ação", "João", "nº", "中文", "Ελληνικά", "हिन्दी", "ภาษาไทย"]:
        assert expected in extracted


def test_appended_text_layer_preserves_existing_page_content(tmp_path: Path) -> None:
    pdf_path = tmp_path / "existing_content.pdf"
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(500, 200))
    page.Resources = pikepdf.Dictionary(
        Font=pikepdf.Dictionary(
            F1=pikepdf.Dictionary(
                Type=pikepdf.Name("/Font"),
                Subtype=pikepdf.Name("/Type1"),
                BaseFont=pikepdf.Name("/Helvetica"),
                Encoding=pikepdf.Name("/WinAnsiEncoding"),
            )
        )
    )
    page.Contents = pikepdf.Stream(pdf, b"BT /F1 10 Tf 1 0 0 1 20 20 Tm (visible) Tj ET")
    ocr_result = OCRResult("ação 中文", [[20, 20], [240, 20], [240, 60], [20, 60]], 1.0)

    commands = create_text_layer_commands(
        [ocr_result],
        img_x=0,
        img_y=0,
        img_width=500,
        img_height=200,
        scale_x=1,
        scale_y=1,
    )
    append_text_to_page(pdf, page, commands)
    pdf.save(pdf_path)

    result = subprocess.run(
        ["pdftotext", str(pdf_path), "-"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "visible" in result.stdout
    assert "ação" in result.stdout
    assert "中文" in result.stdout


def test_appended_actual_text_layer_is_detected_and_stripped(tmp_path: Path) -> None:
    pdf_path = tmp_path / "ocr_only.pdf"
    stripped_path = tmp_path / "stripped.pdf"
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(500, 200))
    ocr_result = OCRResult("ação 中文", [[20, 20], [240, 20], [240, 60], [20, 60]], 1.0)
    commands = create_text_layer_commands(
        [ocr_result],
        img_x=0,
        img_y=0,
        img_width=500,
        img_height=200,
        scale_x=1,
        scale_y=1,
    )
    append_text_to_page(pdf, page, commands)
    pdf.save(pdf_path)

    assert has_native_text(pdf_path) is False
    with pikepdf.open(pdf_path) as loaded_pdf:
        loaded_page = loaded_pdf.pages[0]
        assert page_has_ocr_text(loaded_page) is True
        assert strip_invisible_text(loaded_page, loaded_pdf) == 1
        assert page_has_ocr_text(loaded_page) is False
        loaded_pdf.save(stripped_path)

    result = subprocess.run(
        ["pdftotext", str(stripped_path), "-"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert "ação" not in result.stdout
    assert "中文" not in result.stdout


class TestSetRootPageLayout:
    """Tests for the viewer /PageLayout catalog setting."""

    @staticmethod
    def _blank_pdf() -> pikepdf.Pdf:
        pdf = pikepdf.Pdf.new()
        pdf.add_blank_page(page_size=(200, 300))
        return pdf

    def test_default_omits_page_layout(self):
        pdf = self._blank_pdf()
        assert set_root_page_layout(pdf, "default") is False
        assert "/PageLayout" not in pdf.Root

    def test_unknown_value_omits_page_layout(self):
        pdf = self._blank_pdf()
        assert set_root_page_layout(pdf, "garbage") is False
        assert "/PageLayout" not in pdf.Root

    def test_single_maps_to_singlepage(self):
        pdf = self._blank_pdf()
        assert set_root_page_layout(pdf, "single") is True
        assert str(pdf.Root.PageLayout) == "/SinglePage"

    def test_continuous_maps_to_onecolumn(self):
        pdf = self._blank_pdf()
        assert set_root_page_layout(pdf, "continuous") is True
        assert str(pdf.Root.PageLayout) == "/OneColumn"

    def test_two_page_maps_to_twocolumnleft(self):
        pdf = self._blank_pdf()
        assert set_root_page_layout(pdf, "two_page") is True
        assert str(pdf.Root.PageLayout) == "/TwoColumnLeft"
