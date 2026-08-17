"""OCR CLI page-selection contracts."""

import argparse
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pikepdf
import pytest

from bigocrpdf.cli_ocr_commands import _prepare_selected_pages, _run_full_ocr
from bigocrpdf.services.rapidocr_service.config import OCRConfig, ProcessingStats

SIZES = [(100, 200), (300, 400), (500, 600)]


def _write_pdf(path: Path, sizes: list[tuple[int, int]] = SIZES) -> None:
    with pikepdf.Pdf.new() as pdf:
        for width, height in sizes:
            pdf.add_blank_page(page_size=(width, height))
        pdf.save(path)


def _page_sizes(path: Path) -> list[tuple[int, int]]:
    with pikepdf.open(path) as pdf:
        return [
            (int(page.mediabox[2] - page.mediabox[0]), int(page.mediabox[3] - page.mediabox[1]))
            for page in pdf.pages
        ]


def _args(source: Path, output: Path, pages: str) -> argparse.Namespace:
    return argparse.Namespace(
        input=source,
        output=output,
        pages=pages,
        save_preprocessed=None,
        sidecar_json=None,
    )


def _logger() -> logging.Logger:
    logger = logging.Logger("test-page-selection")
    logger.addHandler(logging.NullHandler())
    return logger


def test_prepare_selected_pages_preserves_order_and_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    selected = tmp_path / "selected.pdf"
    _write_pdf(source)
    with pikepdf.open(source, allow_overwriting_input=True) as pdf:
        pdf.docinfo["/Title"] = "Scanned forms"
        pdf.save(source)

    assert _prepare_selected_pages(source, selected, [3, 1]) == selected
    assert _page_sizes(selected) == [SIZES[2], SIZES[0]]
    with pikepdf.open(selected) as pdf:
        assert str(pdf.docinfo["/Title"]) == "Scanned forms"


def test_prepare_selected_pages_reuses_source_for_all_pages(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    selected = tmp_path / "selected.pdf"
    _write_pdf(source, SIZES[:2])

    assert _prepare_selected_pages(source, selected, [1, 2]) == source
    assert not selected.exists()


def test_prepare_selected_pages_rejects_out_of_range_page(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    selected = tmp_path / "selected.pdf"
    _write_pdf(source, SIZES[:2])

    with pytest.raises(ValueError, match="Page 3 is out of range; document has 2 pages"):
        _prepare_selected_pages(source, selected, [1, 3])

    assert not selected.exists()


@pytest.mark.parametrize(
    ("pages", "page_range", "page_set", "expected_sizes"),
    [
        ("2", (2, 2), None, [SIZES[1]]),
        ("1,3", (1, 3), {1, 3}, [SIZES[0], SIZES[2]]),
    ],
)
def test_full_ocr_processes_only_selected_pages(
    tmp_path: Path,
    pages: str,
    page_range: tuple[int, int],
    page_set: set[int] | None,
    expected_sizes: list[tuple[int, int]],
) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    _write_pdf(source)
    processed_inputs: list[Path] = []

    def process(input_pdf, staged_output, progress_callback):
        input_pdf = Path(input_pdf)
        processed_inputs.append(input_pdf)
        assert _page_sizes(input_pdf) == expected_sizes
        shutil.copy2(input_pdf, staged_output)
        return ProcessingStats(pages_total=len(expected_sizes), pages_processed=len(expected_sizes))

    config = OCRConfig()
    with patch(
        "bigocrpdf.services.rapidocr_service.backend.ProfessionalPDFOCR",
        return_value=SimpleNamespace(process=process),
    ):
        result = _run_full_ocr(
            _args(source, output, pages),
            config,
            page_range,
            _logger(),
            page_set=page_set,
        )

    assert result == 0
    assert _page_sizes(output) == expected_sizes
    assert config.page_range is None
    assert processed_inputs and not processed_inputs[0].parent.exists()


def test_full_ocr_rejects_page_outside_document(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    _write_pdf(source, SIZES[:2])
    process = MagicMock()

    with patch(
        "bigocrpdf.services.rapidocr_service.backend.ProfessionalPDFOCR",
        return_value=SimpleNamespace(process=process),
    ):
        result = _run_full_ocr(
            _args(source, output, "3"),
            OCRConfig(),
            (3, 3),
            _logger(),
        )

    assert result == 1
    assert not output.exists()
    process.assert_not_called()
