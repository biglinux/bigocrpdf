import argparse
import hashlib
import json
import logging
import re
import shutil
import subprocess
import threading
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from bigocrpdf.cli import _cmd_export_odf, _cmd_export_txt  # type: ignore[import-untyped]
from bigocrpdf.services.rapidocr_service.config import (  # type: ignore[import-untyped]
    OcrDocument,
    OcrLayoutBlock,
    OcrLine,
    OcrPage,
    OCRResult,
    OcrWord,
)
from bigocrpdf.services.rapidocr_service.ocr_document_export import (  # type: ignore[import-untyped]
    convert_ocr_document_to_markdown,
    convert_ocr_document_to_odf,
    convert_ocr_document_to_text,
    ocr_document_to_pages_elements,
)
from bigocrpdf.services.rapidocr_service.ocr_document_io import (  # type: ignore[import-untyped]
    complete_ocr_document,
    load_ocr_document_json,
    ocr_document_json_path,
    write_ocr_document_json,
)
from bigocrpdf.utils.odf_builder import (  # type: ignore[import-untyped]
    ExportCancelled,
    create_odf,
    create_positioned_text_odf,
)
from bigocrpdf.utils.tsv_odf_converter import (  # type: ignore[import-untyped]
    convert_pdf_to_odf,
    create_text,
    fix_cross_page_breaks,
)
from bigocrpdf.utils.tsv_parser import DocElement, Word  # type: ignore[import-untyped]


def _box(left: float, top: float, width: float, height: float = 10.0) -> list[list[float]]:
    return [
        [left, top],
        [left + width, top],
        [left + width, top + height],
        [left, top + height],
    ]


def test_cross_page_merge_preserves_raw_lines_in_plain_text() -> None:
    pages = [
        [DocElement("paragraph", "First part,", raw_lines=["First part,"])],
        [DocElement("paragraph", "continued here", raw_lines=["continued here"])],
    ]

    text = create_text(fix_cross_page_breaks(pages))

    assert "First part,\ncontinued here" in text
    assert text.index("continued here") < text.index("--- Page 2 ---")


def test_ocr_document_text_preserves_page_order() -> None:
    document = OcrDocument(
        pages=[
            OcrPage(
                page_index=2,
                width_px=600,
                height_px=800,
                dpi=300,
                text_results=[OCRResult("Second page", _box(20, 20, 100), 0.95)],
            ),
            OcrPage(
                page_index=1,
                width_px=600,
                height_px=800,
                dpi=300,
                text_results=[OCRResult("First page", _box(20, 20, 100), 0.98)],
            ),
        ]
    )

    text = convert_ocr_document_to_text(document)

    assert text.startswith("--- Page 1 ---")
    assert text.index("First page") < text.index("--- Page 2 ---") < text.index("Second page")


def _write_one_page_pdf(path: Path) -> None:
    """Write a real single-page PDF.

    The JSON writer refuses a document that does not cover the PDF's pages, so
    these tests need a PDF whose page count is real rather than a byte prefix.
    """
    import pikepdf

    with pikepdf.Pdf.new() as pdf:
        pdf.add_blank_page(page_size=(200, 200))
        pdf.save(path)


def _write_json(document: OcrDocument, pdf_path: Path) -> Path:
    """Write structured OCR to the default name beside *pdf_path*."""
    return write_ocr_document_json(document, pdf_path, ocr_document_json_path(pdf_path))


def _load_json(pdf_path: Path, **kwargs) -> OcrDocument | None:
    """Load structured OCR from the default name beside *pdf_path*."""
    return load_ocr_document_json(ocr_document_json_path(pdf_path), pdf_path, **kwargs)


def test_cli_positioned_odt_ignores_corrupt_structured_json(tmp_path: Path) -> None:
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    ocr_document_json_path(pdf_path).write_text("{not-json", encoding="utf-8")
    output_path = tmp_path / "positioned.odt"
    args = argparse.Namespace(
        input=pdf_path,
        output=output_path,
        preserve_text_layout=True,
    )

    with patch(
        "bigocrpdf.utils.tsv_odf_converter.convert_pdf_to_odf",
        return_value=str(output_path),
    ) as convert_pdf:
        exit_code = _cmd_export_odf(args, logging.getLogger("test"))

    assert exit_code == 0
    convert_pdf.assert_called_once_with(str(pdf_path), str(output_path), include_images=True)


def test_ocr_document_markdown_uses_structured_table_rows(tmp_path: Path) -> None:
    document = OcrDocument(
        pages=[
            OcrPage(
                page_index=1,
                width_px=800,
                height_px=1000,
                dpi=300,
                text_results=[
                    OCRResult("Name", _box(20, 20, 40), 0.99),
                    OCRResult("Age", _box(180, 20, 30), 0.99),
                    OCRResult("Alice", _box(20, 42, 40), 0.99),
                    OCRResult("30", _box(180, 42, 20), 0.99),
                ],
            )
        ]
    )

    markdown = convert_ocr_document_to_markdown(
        document,
        source_path=str(tmp_path / "people.pdf"),
        include_front_matter=True,
    )

    assert 'title: "people"' in markdown
    assert "| Name | Age |" in markdown
    assert "| Alice | 30 |" in markdown


def test_ocr_document_splits_form_separators_into_table_columns() -> None:
    document = OcrDocument(
        pages=[
            OcrPage(
                page_index=1,
                width_px=800,
                height_px=1000,
                dpi=300,
                text_results=[
                    OCRResult("Name:Alice", _box(20, 20, 260), 0.99),
                    OCRResult("Date:2026", _box(20, 42, 260), 0.99),
                ],
            )
        ]
    )

    markdown = convert_ocr_document_to_markdown(document)

    assert "| Name: | Alice |" in markdown
    assert "| Date: | 2026 |" in markdown


def test_ocr_document_export_prefers_structured_lines() -> None:
    document = OcrDocument(
        pages=[
            OcrPage(
                page_index=1,
                width_px=800,
                height_px=1000,
                dpi=300,
                text_results=[OCRResult("Stale fallback text", _box(20, 20, 200), 0.5)],
                lines=[
                    OcrLine(
                        text="Structured text",
                        bbox=[20, 20, 180, 32],
                        reading_order=0,
                        words=[
                            OcrWord("Structured", [20, 20, 95, 32], 0.99),
                            OcrWord("text", [105, 20, 140, 32], 0.99),
                        ],
                    )
                ],
            )
        ]
    )

    text = convert_ocr_document_to_text(document)

    assert "Structured text" in text
    assert "Stale fallback text" not in text


def test_ocr_document_export_prefers_persisted_layout_blocks() -> None:
    document = OcrDocument(
        pages=[
            OcrPage(
                page_index=1,
                width_px=800,
                height_px=1000,
                dpi=300,
                text_results=[OCRResult("Stale fallback text", _box(20, 20, 200), 0.5)],
                layout_blocks=[
                    OcrLayoutBlock(
                        kind="table",
                        rows=[["Field", "Value"], ["Name", "Alice"]],
                        reading_order=0,
                    )
                ],
            )
        ]
    )

    markdown = convert_ocr_document_to_markdown(document)

    assert "| Field | Value |" in markdown
    assert "| Name | Alice |" in markdown
    assert "Stale fallback text" not in markdown


def test_ocr_document_odt_export_writes_content_xml(tmp_path: Path) -> None:
    document = OcrDocument(
        pages=[
            OcrPage(
                page_index=1,
                width_px=600,
                height_px=800,
                dpi=300,
                text_results=[OCRResult("Contract heading", _box(20, 20, 120), 0.97)],
            )
        ]
    )
    output_path = tmp_path / "out.odt"

    result = convert_ocr_document_to_odf(document, str(output_path))

    assert result == str(output_path)
    with zipfile.ZipFile(output_path) as odt:
        content = odt.read("content.xml").decode("utf-8")
        styles = odt.read("styles.xml").decode("utf-8")
    assert "Contract heading" in content
    assert 'fo:page-width="22.27cm"' in styles
    assert 'fo:page-height="29.70cm"' in styles


def test_structured_odt_reflows_paragraph_without_ocr_line_breaks(tmp_path: Path) -> None:
    document = OcrDocument(
        pages=[
            OcrPage(
                page_index=1,
                width_px=800,
                height_px=1000,
                dpi=300,
                layout_blocks=[
                    OcrLayoutBlock(
                        kind="paragraph",
                        text="This paragraph continues across OCR lines.",
                        raw_lines=["This paragraph continues", "across OCR lines."],
                    )
                ],
            )
        ]
    )
    output_path = tmp_path / "reflow.odt"

    convert_ocr_document_to_odf(document, str(output_path))

    with zipfile.ZipFile(output_path) as odt:
        content = odt.read("content.xml").decode("utf-8")
    assert "This paragraph continues across OCR lines." in content
    assert "text:line-break" not in content


def test_structured_odt_preserves_preformatted_line_breaks(tmp_path: Path) -> None:
    document = OcrDocument(
        pages=[
            OcrPage(
                page_index=1,
                width_px=800,
                height_px=1000,
                dpi=300,
                layout_blocks=[
                    OcrLayoutBlock(
                        kind="preformatted",
                        text="┌───┐ │ A │ └───┘",
                        raw_lines=["┌───┐", "│ A │", "└───┘"],
                    )
                ],
            )
        ]
    )
    output_path = tmp_path / "preformatted.odt"

    convert_ocr_document_to_odf(document, str(output_path))

    with zipfile.ZipFile(output_path) as odt:
        content = odt.read("content.xml").decode("utf-8")
    assert 'text:style-name="Preformatted"' in content
    assert content.count("text:line-break") == 2


@pytest.mark.skipif(
    shutil.which("libreoffice") is None or shutil.which("pdfinfo") is None,
    reason="LibreOffice and Poppler are required for rendered ODT pagination validation",
)
def test_structured_odt_preserves_intermediate_empty_page(tmp_path: Path) -> None:
    output_path = tmp_path / "structured-empty-page.odt"
    create_odf(
        [
            [DocElement(kind="paragraph", text="First page")],
            [],
            [DocElement(kind="paragraph", text="Third page")],
        ],
        str(output_path),
    )

    profile_uri = (tmp_path / "libreoffice-structured-profile").as_uri()
    subprocess.run(
        [
            "libreoffice",
            "--headless",
            f"-env:UserInstallation={profile_uri}",
            "--convert-to",
            "pdf",
            "--outdir",
            str(tmp_path),
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    page_info = subprocess.run(
        ["pdfinfo", str(output_path.with_suffix(".pdf"))],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    assert "Pages:           3" in page_info


def test_cancelled_empty_structured_odt_is_not_written(tmp_path: Path) -> None:
    output_path = tmp_path / "cancelled.odt"
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(ExportCancelled):
        create_odf([], str(output_path), cancel_event=cancel_event)

    assert not output_path.exists()


def test_cancelled_empty_positioned_odt_is_not_written(tmp_path: Path) -> None:
    output_path = tmp_path / "cancelled-positioned.odt"
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(ExportCancelled):
        create_positioned_text_odf({}, str(output_path), [], cancel_event)

    assert not output_path.exists()


def test_structured_odt_detects_compact_three_column_table(tmp_path: Path) -> None:
    from bigocrpdf.utils.tsv_odf_converter import process_page

    elements = process_page(
        [
            Word("Produto", 18, 30, 35, 9),
            Word("Quantidade", 92, 30, 52, 9),
            Word("Valor", 163, 30, 24, 9),
            Word("Livro", 18, 49, 22, 10),
            Word("2", 90, 49, 6, 10),
            Word("R$", 163, 49, 12, 10),
            Word("80,00", 178, 49, 25, 10),
        ],
        1,
    )

    assert len(elements) == 1
    assert elements[0].kind == "table"
    assert elements[0].rows == [["Produto", "Quantidade", "Valor"], ["Livro", "2", "R$ 80,00"]]


def test_positioned_odt_anchors_editable_text_per_page(tmp_path: Path) -> None:
    output_path = tmp_path / "positioned.odt"
    create_positioned_text_odf(
        {
            1: [Word("Page one", 72, 72, 90, 12)],
            2: [Word("Page two", 36, 48, 80, 10)],
        },
        str(output_path),
        [(612, 792, 22.95, 29.7), (792, 612, 29.7, 22.95)],
    )

    with zipfile.ZipFile(output_path) as odt:
        content = odt.read("content.xml").decode("utf-8")
        styles = odt.read("styles.xml").decode("utf-8")
        members = odt.namelist()
    assert content.count('text:anchor-type="page"') == 2
    assert 'text:anchor-page-number="1"' in content
    assert 'text:anchor-page-number="2"' in content
    assert "Page one" in content
    assert "Page two" in content
    # Distinct and ascending: a shared z-index leaves the stacking order
    # undefined, and with it which frame a click selects, the Tab order, and
    # the order LibreOffice emits text in when exporting.
    z_indices = [int(m) for m in re.findall(r'draw:z-index="(\d+)"', content)]
    assert z_indices == sorted(z_indices)
    assert len(set(z_indices)) == len(z_indices) == 2
    assert 'style:master-page-name="PositionedMaster2"' in content
    assert 'fo:margin="0cm"' in styles
    assert content.index("<draw:frame") < content.index("<text:p")
    assert not any(member.startswith("Pictures/") for member in members)


@pytest.mark.skipif(
    shutil.which("libreoffice") is None
    or shutil.which("pdftotext") is None
    or shutil.which("pdfinfo") is None,
    reason="LibreOffice and Poppler are required for rendered ODT text validation",
)
def test_positioned_odt_renders_extractable_text(tmp_path: Path) -> None:
    output_path = tmp_path / "positioned.odt"
    create_positioned_text_odf(
        {
            1: [Word("Editable contract text", 72, 72, 140, 12)],
            3: [Word("Third page text", 72, 72, 120, 12)],
        },
        str(output_path),
        [(612, 792, 22.95, 29.7)] * 3,
    )

    profile_uri = (tmp_path / "libreoffice-profile").as_uri()
    subprocess.run(
        [
            "libreoffice",
            "--headless",
            f"-env:UserInstallation={profile_uri}",
            "--convert-to",
            "pdf",
            "--outdir",
            str(tmp_path),
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    rendered_pdf = output_path.with_suffix(".pdf")
    rendered_text = subprocess.run(
        ["pdftotext", str(rendered_pdf), "-"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    assert "Editable contract text" in rendered_text
    assert "Third page text" in rendered_text
    page_info = subprocess.run(
        ["pdfinfo", str(rendered_pdf)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    assert "Pages:           3" in page_info
    empty_page_text = subprocess.run(
        ["pdftotext", "-f", "2", "-l", "2", str(rendered_pdf), "-"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    assert not empty_page_text.strip()


def test_positioned_odt_does_not_silently_fall_back_without_geometry(tmp_path: Path) -> None:
    with (
        patch("bigocrpdf.utils.tsv_odf_converter.parse_tsv_pages", return_value={}),
        patch("bigocrpdf.utils.tsv_odf_converter._pdf_page_geometries", return_value=[]),
        pytest.raises(ValueError, match="page geometry for positioned ODT"),
    ):
        convert_pdf_to_odf("source.pdf", str(tmp_path / "out.odt"), include_images=True)


def test_positioned_odt_splits_columns_at_large_horizontal_gap(tmp_path: Path) -> None:
    output_path = tmp_path / "columns.odt"
    create_positioned_text_odf(
        {
            1: [
                Word("Left", 40, 100, 30, 10),
                Word("column", 75, 100, 45, 10),
                Word("Right", 360, 100, 35, 10),
                Word("column", 400, 100, 45, 10),
            ]
        },
        str(output_path),
        [(612, 792, 22.95, 29.7)],
    )

    with zipfile.ZipFile(output_path) as odt:
        content = odt.read("content.xml").decode("utf-8")
    assert content.count("<draw:text-box") == 2
    assert ">Left column<" in content
    assert ">Right column<" in content


def test_positioned_odt_preserves_intermediate_page_without_text(tmp_path: Path) -> None:
    output_path = tmp_path / "empty-page.odt"
    create_positioned_text_odf(
        {1: [Word("First", 40, 40, 30, 10)], 3: [Word("Third", 40, 40, 30, 10)]},
        str(output_path),
        [(612, 792, 22.95, 29.7)] * 3,
    )

    with zipfile.ZipFile(output_path) as odt:
        content = odt.read("content.xml").decode("utf-8")
    assert 'text:anchor-page-number="1"' in content
    assert 'text:anchor-page-number="2"' not in content
    assert 'text:anchor-page-number="3"' in content
    assert 'style:master-page-name="PositionedMaster2"' in content
    assert 'style:master-page-name="PositionedMaster3"' in content


def test_odt_preserves_centered_heading_geometry(tmp_path: Path) -> None:
    document = OcrDocument(
        pages=[
            OcrPage(
                page_index=1,
                width_px=800,
                height_px=1000,
                dpi=300,
                lines=[
                    OcrLine(
                        "Centered title",
                        bbox=[250, 100, 550, 125],
                        reading_order=0,
                    )
                ],
                layout_blocks=[
                    OcrLayoutBlock(
                        kind="heading2",
                        text="Centered title",
                        y_top=100,
                        reading_order=0,
                    )
                ],
            )
        ]
    )
    output_path = tmp_path / "centered.odt"

    convert_ocr_document_to_odf(document, str(output_path))

    with zipfile.ZipFile(output_path) as odt:
        content = odt.read("content.xml").decode("utf-8")
    assert 'text:style-name="H2C"' in content
    assert 'style:name="H2C"' in content
    assert 'fo:text-align="center"' in content


def test_odt_preserves_detected_column_flow(tmp_path: Path) -> None:
    document = OcrDocument(
        pages=[
            OcrPage(
                page_index=1,
                width_px=1200,
                height_px=1600,
                dpi=150,
                layout_blocks=[
                    OcrLayoutBlock(
                        kind="paragraph", text="Left column", y_top=700, reading_order=0
                    ),
                    OcrLayoutBlock(
                        kind="paragraph", text="Right column", y_top=100, reading_order=1
                    ),
                ],
            )
        ]
    )
    output_path = tmp_path / "columns.odt"

    convert_ocr_document_to_odf(document, str(output_path))

    with zipfile.ZipFile(output_path) as odt:
        content = odt.read("content.xml").decode("utf-8")
    assert 'fo:column-count="2"' in content
    assert 'fo:break-before="column"' in content
    assert "Left column" in content
    assert "Right column" in content


def test_odt_scales_typography_from_source_line_height(tmp_path: Path) -> None:
    document = OcrDocument(
        pages=[
            OcrPage(
                page_index=1,
                width_px=2550,
                height_px=3300,
                dpi=300,
                lines=[OcrLine("Dense text", bbox=[100, 100, 500, 125], reading_order=0)],
                layout_blocks=[
                    OcrLayoutBlock(kind="paragraph", text="Dense text", y_top=100, reading_order=0)
                ],
            )
        ]
    )
    output_path = tmp_path / "dense.odt"

    convert_ocr_document_to_odf(document, str(output_path))

    with zipfile.ZipFile(output_path) as odt:
        content = odt.read("content.xml").decode("utf-8")
    assert 'fo:font-size="6.50pt"' in content


def test_odt_scales_typography_from_legacy_result_geometry(tmp_path: Path) -> None:
    document = OcrDocument(
        pages=[
            OcrPage(
                page_index=1,
                width_px=2550,
                height_px=3300,
                dpi=300,
                text_results=[OCRResult("Dense text", _box(100, 100, 400, 25), 0.97)],
            )
        ]
    )
    output_path = tmp_path / "legacy-dense.odt"

    convert_ocr_document_to_odf(document, str(output_path))

    with zipfile.ZipFile(output_path) as odt:
        content = odt.read("content.xml").decode("utf-8")
    assert 'fo:font-size="6.50pt"' in content


def test_ocr_document_odt_export_replaces_symlink_without_touching_target(tmp_path: Path) -> None:
    document = OcrDocument(
        pages=[
            OcrPage(
                page_index=1,
                width_px=600,
                height_px=800,
                dpi=300,
                text_results=[OCRResult("Safe export", _box(20, 20, 100), 0.97)],
            )
        ]
    )
    protected_path = tmp_path / "protected.odt"
    protected_path.write_text("do not overwrite", encoding="utf-8")
    output_path = tmp_path / "out.odt"
    output_path.symlink_to(protected_path)

    convert_ocr_document_to_odf(document, str(output_path))

    assert protected_path.read_text(encoding="utf-8") == "do not overwrite"
    assert output_path.is_file()
    assert not output_path.is_symlink()


def test_ocr_document_sidecar_roundtrip(tmp_path: Path) -> None:
    pdf_path = tmp_path / "out.pdf"
    _write_one_page_pdf(pdf_path)
    document = OcrDocument(
        diagnostics={"engine": "rapidocr"},
        pages=[
            OcrPage(
                page_index=1,
                width_px=600,
                height_px=800,
                dpi=300,
                retry_level=1,
                diagnostics={"mode": "ocr"},
                text_results=[OCRResult("Saved text", _box(20, 20, 80), 0.91)],
                layout_blocks=[
                    OcrLayoutBlock(
                        kind="paragraph",
                        text="Saved text",
                        raw_lines=["Saved text"],
                        reading_order=0,
                    )
                ],
                lines=[
                    OcrLine(
                        text="Saved text",
                        bbox=[20, 20, 100, 30],
                        reading_order=0,
                        words=[
                            OcrWord("Saved", [20, 20, 58, 30], 0.91),
                            OcrWord("text", [64, 20, 100, 30], 0.91),
                        ],
                    )
                ],
            )
        ],
    )

    sidecar_path = _write_json(document, pdf_path)
    loaded = _load_json(pdf_path)

    assert sidecar_path == ocr_document_json_path(pdf_path)
    assert loaded is not None
    assert loaded.diagnostics == {"engine": "rapidocr"}
    assert loaded.pages[0].retry_level == 1
    assert loaded.pages[0].text_results[0].text == "Saved text"
    assert loaded.pages[0].text_results[0].box == _box(20, 20, 80)
    assert loaded.pages[0].lines[0].text == "Saved text"
    assert loaded.pages[0].lines[0].words[1].bbox == [64, 20, 100, 30]
    assert loaded.pages[0].layout_blocks[0].text == "Saved text"
    assert loaded.pages[0].layout_blocks[0].raw_lines == ["Saved text"]


def test_legacy_sidecar_requires_explicit_unverified_opt_in(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "legacy.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\nlegacy")
    ocr_document_json_path(pdf_path).write_text(
        '{"version": 1, "document": {"diagnostics": {}, "pages": []}}',
        encoding="utf-8",
    )

    assert _load_json(pdf_path) is None
    assert _load_json(pdf_path, allow_unverified_legacy=True) is not None


def test_sidecar_temp_symlink_cannot_overwrite_target(tmp_path: Path) -> None:
    pdf_path = tmp_path / "out.pdf"
    _write_one_page_pdf(pdf_path)
    victim = tmp_path / "victim.txt"
    victim.write_text("KEEP", encoding="utf-8")
    predictable_temp = tmp_path / "out.bigocr.json.tmp"
    predictable_temp.symlink_to(victim)

    _write_json(OcrDocument(pages=[OcrPage(1, 100, 100, 300)]), pdf_path)

    assert victim.read_text(encoding="utf-8") == "KEEP"
    assert predictable_temp.is_symlink()


def test_write_ocr_document_json_enriches_layout_blocks(tmp_path: Path) -> None:
    pdf_path = tmp_path / "out.pdf"
    _write_one_page_pdf(pdf_path)
    document = OcrDocument(
        pages=[
            OcrPage(
                page_index=1,
                width_px=800,
                height_px=1000,
                dpi=300,
                text_results=[
                    OCRResult("Name", _box(20, 20, 40), 0.99),
                    OCRResult("Age", _box(180, 20, 30), 0.99),
                    OCRResult("Alice", _box(20, 42, 40), 0.99),
                    OCRResult("30", _box(180, 42, 20), 0.99),
                ],
            )
        ]
    )

    _write_json(document, pdf_path)
    loaded = _load_json(pdf_path)

    assert loaded is not None
    assert loaded.pages[0].layout_blocks[0].kind == "table"
    assert loaded.pages[0].layout_blocks[0].rows == [["Name", "Age"], ["Alice", "30"]]


def test_stale_sidecar_is_ignored_after_pdf_content_changes(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "out.pdf"
    _write_one_page_pdf(pdf_path)
    _write_json(
        OcrDocument(
            pages=[
                OcrPage(
                    page_index=1,
                    width_px=100,
                    height_px=100,
                    dpi=300,
                    native_text="stale text",
                )
            ]
        ),
        pdf_path,
    )

    pdf_path.write_bytes(pdf_path.read_bytes() + b"\n% edited elsewhere\n")

    assert _load_json(pdf_path) is None


def test_json_is_refused_when_it_does_not_cover_the_pdf_it_names(tmp_path: Path) -> None:
    """Page counts from the pipeline are a claim; the PDF's own pages are the fact.

    ``complete_ocr_document`` can only compare the document against the counts
    it was handed, so a document covering one page passes it while the published
    PDF has two. A file that names a PDF must describe all of it.
    """
    import pikepdf

    pdf_path = tmp_path / "two.pdf"
    with pikepdf.Pdf.new() as pdf:
        pdf.add_blank_page(page_size=(200, 200))
        pdf.add_blank_page(page_size=(200, 200))
        pdf.save(pdf_path)
    one_page = OcrDocument(pages=[OcrPage(1, 100, 100, 300, native_text="only the first")])
    assert complete_ocr_document(one_page, pages_total=1, pages_processed=1) is not None

    with pytest.raises(ValueError, match="covers 1 of the 2 pages"):
        _write_json(one_page, pdf_path)

    assert not ocr_document_json_path(pdf_path).exists()


def test_invalidation_marker_written_by_older_versions_still_loads_as_nothing(
    tmp_path: Path,
) -> None:
    """Older versions dropped a marker beside every PDF they could not describe.

    Nothing writes one now, but the ones already on disk must keep reading as
    'no structured OCR' rather than as an error.
    """
    pdf_path = tmp_path / "out.pdf"
    pdf_path.write_bytes(b"PDF without structured OCR")
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    ocr_document_json_path(pdf_path).write_text(
        json.dumps(
            {
                "version": 2,
                "state": "unavailable",
                "pdf": {"sha256": digest, "size_bytes": pdf_path.stat().st_size},
                "reason": "structured-data-not-produced",
            }
        ),
        encoding="utf-8",
    )

    assert _load_json(pdf_path) is None


def test_ocr_document_sidecar_corrupt_json_fails_clearly(tmp_path: Path) -> None:
    pdf_path = tmp_path / "out.pdf"
    ocr_document_json_path(pdf_path).write_text("{not-json", encoding="utf-8")

    try:
        _load_json(pdf_path)
    except ValueError as exc:
        assert "Invalid OCR JSON" in str(exc)
    else:
        raise AssertionError("Corrupt OCR sidecar was accepted")


def test_ocr_document_sidecar_symlink_is_rejected(tmp_path: Path) -> None:
    pdf_path = tmp_path / "out.pdf"
    target = tmp_path / "target.json"
    target.write_text(
        '{"version": 1, "document": {"diagnostics": {}, "pages": []}}',
        encoding="utf-8",
    )
    ocr_document_json_path(pdf_path).symlink_to(target)

    with pytest.raises(ValueError, match="Invalid OCR JSON"):
        _load_json(pdf_path, allow_unverified_legacy=True)


def test_ocr_document_sidecar_invalid_utf8_fails_clearly(tmp_path: Path) -> None:
    pdf_path = tmp_path / "out.pdf"
    ocr_document_json_path(pdf_path).write_bytes(b"\xff")

    with pytest.raises(ValueError, match="Invalid OCR JSON"):
        _load_json(pdf_path)


def test_ocr_document_sidecar_missing_document_fails_clearly(tmp_path: Path) -> None:
    pdf_path = tmp_path / "out.pdf"
    ocr_document_json_path(pdf_path).write_text('{"version": 1}', encoding="utf-8")

    try:
        _load_json(pdf_path)
    except ValueError as exc:
        assert "missing document payload" in str(exc)
    else:
        raise AssertionError("Incomplete OCR sidecar was accepted")


def test_ocr_document_sidecar_invalid_document_shape_fails_clearly(tmp_path: Path) -> None:
    pdf_path = tmp_path / "out.pdf"
    ocr_document_json_path(pdf_path).write_text(
        '{"version": 1, "document": {"pages": [null]}}',
        encoding="utf-8",
    )

    try:
        _load_json(pdf_path)
    except ValueError as exc:
        assert "Invalid OCR JSON" in str(exc)
    else:
        raise AssertionError("Structurally invalid OCR sidecar was accepted")


def test_ocr_document_sidecar_invalid_version_fails_clearly(tmp_path: Path) -> None:
    pdf_path = tmp_path / "out.pdf"
    ocr_document_json_path(pdf_path).write_text(
        '{"version": "banana", "document": {"pages": []}}',
        encoding="utf-8",
    )

    try:
        _load_json(pdf_path)
    except ValueError as exc:
        assert "Unsupported OCR JSON version" in str(exc)
    else:
        raise AssertionError("Invalid OCR JSON version was accepted")


def test_native_text_page_becomes_document_paragraph() -> None:
    document = OcrDocument(
        pages=[
            OcrPage(
                page_index=1,
                width_px=600,
                height_px=800,
                dpi=300,
                native_text="Line one\nLine two",
            )
        ]
    )

    pages = ocr_document_to_pages_elements(document)

    assert pages[0][0].text == "Line one Line two"
    assert pages[0][0].raw_lines == ["Line one", "Line two"]


def _structured_json_for_export(tmp_path: Path) -> tuple[Path, Path]:
    pdf_path = tmp_path / "input.pdf"
    _write_one_page_pdf(pdf_path)
    json_path = _write_json(
        OcrDocument(
            pages=[
                OcrPage(
                    page_index=1,
                    width_px=600,
                    height_px=800,
                    dpi=300,
                    text_results=[OCRResult("Structured text", _box(20, 20, 90), 0.95)],
                )
            ]
        ),
        pdf_path,
    )
    return pdf_path, json_path


def test_cli_export_txt_uses_structured_json_when_asked(tmp_path: Path) -> None:
    pdf_path, json_path = _structured_json_for_export(tmp_path)
    output_path = tmp_path / "out.txt"
    args = argparse.Namespace(input=pdf_path, output=output_path, from_json=json_path)

    with patch(
        "bigocrpdf.utils.tsv_odf_converter.convert_pdf_to_text",
        side_effect=AssertionError("pdftotext fallback should not run"),
    ):
        exit_code = _cmd_export_txt(args, logging.getLogger("test"))

    assert exit_code == 0
    assert "Structured text" in output_path.read_text(encoding="utf-8")


def test_cli_export_txt_reads_the_pdf_when_no_json_is_named(tmp_path: Path) -> None:
    """A JSON sitting beside the PDF is not a reason to read it."""
    pdf_path, _json_path = _structured_json_for_export(tmp_path)
    output_path = tmp_path / "out.txt"
    args = argparse.Namespace(input=pdf_path, output=output_path, from_json=None)

    with patch(
        "bigocrpdf.utils.tsv_odf_converter.convert_pdf_to_text",
        return_value="text from the PDF layer",
    ) as from_pdf:
        exit_code = _cmd_export_txt(args, logging.getLogger("test"))

    assert exit_code == 0
    assert from_pdf.called
    assert output_path.read_text(encoding="utf-8") == "text from the PDF layer"


def test_cli_export_txt_falls_back_when_the_named_json_describes_another_pdf(
    tmp_path: Path,
) -> None:
    pdf_path, json_path = _structured_json_for_export(tmp_path)
    pdf_path.write_bytes(b"%PDF-1.7\nchanged after the export was written")
    output_path = tmp_path / "out.txt"
    args = argparse.Namespace(input=pdf_path, output=output_path, from_json=json_path)

    with patch(
        "bigocrpdf.utils.tsv_odf_converter.convert_pdf_to_text",
        return_value="text from the PDF layer",
    ):
        exit_code = _cmd_export_txt(args, logging.getLogger("test"))

    assert exit_code == 0
    assert output_path.read_text(encoding="utf-8") == "text from the PDF layer"
