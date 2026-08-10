"""Tests for PDF text escaping and assembly utilities."""

import subprocess
from pathlib import Path

import pikepdf
import pytest
from reportlab.pdfgen import canvas as reportlab_canvas

from bigocrpdf.services.rapidocr_service.config import OCRConfig, OCRResult
from bigocrpdf.services.rapidocr_service.pdf_assembly import (
    _filter_invisible_text_ops,
    _pdf_text_operand,
    append_text_to_page,
    create_text_layer_commands,
    escape_pdf_text,
    merge_single_page,
    strip_invisible_text,
)
from bigocrpdf.services.rapidocr_service.pdf_extractor import (
    _scan_page_text_ops,
    has_native_text,
    page_has_ocr_text,
)
from bigocrpdf.services.rapidocr_service.pdf_page_geometry import merge_page_fonts
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


class TestTextLayerFontNamesDoNotCollide:
    """A page that already names a font must not decide what our text says.

    ReportLab writes an embedded subset whose codes are assigned in order of
    first use, so code 1 means whatever character that document met first, and
    only its own ToUnicode says which. The merge used to keep any font name the
    page already had and drop ours -- our codes were then read through a
    stranger's table. Measured on a six-page scan re-run through OCR, every
    accent changed identity: ``Jürgen`` became ``Jçrgen``, ``CATALOGAÇÃO``
    became ``CATALOGAõãO``. ``/F1`` and ``/F2+0`` are generic enough that this
    is not an exotic collision, and re-OCRing our own output triggers it every
    time.
    """

    # Distinct accents in a different order on each side, so the two documents
    # cannot agree on subset codes by accident.
    PAGE_TEXT = "çõãêé"
    LAYER_TEXT = "üÇÃáí"

    def _draw(self, path: Path, text: str) -> Path:
        results = [
            OCRResult(text=text, box=[[40, 40], [560, 40], [560, 70], [40, 70]], confidence=0.99)
        ]
        pdf = reportlab_canvas.Canvas(str(path), pagesize=(612, 792))
        TextLayerRenderer(OCRConfig()).render(pdf, results, (612, 792))
        pdf.save()
        return path

    @pytest.fixture
    def merged(self, tmp_path: Path) -> Path:
        page = pikepdf.open(self._draw(tmp_path / "page.pdf", self.PAGE_TEXT))
        layer = pikepdf.open(self._draw(tmp_path / "layer.pdf", self.LAYER_TEXT))

        assert merge_single_page(page.pages[0], layer.pages[0], page, 0)

        out = tmp_path / "merged.pdf"
        page.save(out)
        return out

    def test_the_two_sides_really_do_collide(self, tmp_path: Path):
        """Without a shared font name there is nothing for this to guard."""
        page = pikepdf.open(self._draw(tmp_path / "a.pdf", self.PAGE_TEXT))
        layer = pikepdf.open(self._draw(tmp_path / "b.pdf", self.LAYER_TEXT))

        shared = set(page.pages[0].resources["/Font"].keys()) & set(
            layer.pages[0].resources["/Font"].keys()
        )

        assert shared, "the fixture no longer reproduces the collision"

    def test_our_text_survives_the_merge(self, merged: Path):
        extracted = subprocess.run(
            ["pdftotext", str(merged), "-"], capture_output=True, text=True, check=True
        ).stdout

        assert self.LAYER_TEXT in extracted

    def test_no_accent_is_swapped_for_another(self, merged: Path):
        """The failure was silent: same length, every accent a different one."""
        extracted = subprocess.run(
            ["pdftotext", str(merged), "-"], capture_output=True, text=True, check=True
        ).stdout

        stale = set(self.PAGE_TEXT) - set(self.LAYER_TEXT)
        assert not (stale & set(extracted)), "text decoded through the page's own font"

    def test_the_page_keeps_its_own_font(self, merged: Path):
        """Renaming ours must not evict a font the page's own content uses."""
        with pikepdf.open(merged) as pdf:
            fonts = pdf.pages[0].resources["/Font"]

            assert "/F1" in fonts

    def test_a_font_that_fails_to_copy_records_no_rename(self, tmp_path: Path, monkeypatch):
        """A rename is a promise that the font is there under the new name.

        Recording it before the copy meant a failed copy sent the stream to a
        resource the page does not have -- worse than the wrong font, which at
        least renders something.
        """
        page = pikepdf.open(self._draw(tmp_path / "page.pdf", self.PAGE_TEXT))
        layer = pikepdf.open(self._draw(tmp_path / "layer.pdf", self.LAYER_TEXT))
        monkeypatch.setattr(
            type(page), "copy_foreign", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no"))
        )

        renames = merge_page_fonts(page.pages[0], layer.pages[0].resources, page)

        assert renames == {}
