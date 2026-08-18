"""Crash-safe publication contracts for the OCR CLI."""

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pikepdf

from bigocrpdf.cli_ocr_commands import _run_full_ocr
from bigocrpdf.services.rapidocr_service.config import (
    OCRConfig,
    OcrDocument,
    OcrPage,
    ProcessingStats,
)
from bigocrpdf.services.rapidocr_service.ocr_document_io import (
    _read_split_family,
    load_ocr_document_json,
    ocr_document_json_path,
)


def _write_pdf(path: Path, pages: int = 1) -> None:
    with pikepdf.Pdf.new() as pdf:
        for _index in range(pages):
            pdf.add_blank_page()
        pdf.save(path)


def _pdf_page_count(path: Path) -> int:
    with pikepdf.open(path) as pdf:
        return len(pdf.pages)


def _args(source: Path, output: Path, sidecar_json: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        input=source,
        output=output,
        save_preprocessed=None,
        sidecar_json=sidecar_json,
    )


def _logger() -> logging.Logger:
    logger = logging.Logger("test")
    logger.addHandler(logging.NullHandler())
    return logger


def test_cli_ocr_stages_beside_destination_before_publication(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source")
    output = tmp_path / "output.pdf"
    _write_pdf(output)
    staged_paths: list[Path] = []

    def process(_source, staged, progress_callback):
        staged = Path(staged)
        staged_paths.append(staged)
        assert staged.parent.parent == tmp_path
        assert staged.parent.name.startswith(".bigocr_ocr_")
        _write_pdf(staged)
        return ProcessingStats(pages_total=1, pages_processed=1)

    engine = SimpleNamespace(process=process)
    with patch(
        "bigocrpdf.services.rapidocr_service.backend.ProfessionalPDFOCR",
        return_value=engine,
    ):
        result = _run_full_ocr(
            _args(source, output),
            OCRConfig(),
            None,
            _logger(),
        )

    assert result == 0
    assert _pdf_page_count(output) == 1
    # An OCR run publishes one file. Nothing else may appear beside it.
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["output.pdf", "source.pdf"]
    assert staged_paths and not staged_paths[0].parent.exists()


def test_cli_ocr_logs_confidence_as_percentage(tmp_path: Path, caplog) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source")
    output = tmp_path / "output.pdf"

    def process(_source, staged, progress_callback):
        _write_pdf(Path(staged))
        return ProcessingStats(
            pages_total=1,
            pages_processed=1,
            average_confidence=0.9676,
        )

    logger = logging.getLogger("test-confidence")
    caplog.set_level(logging.INFO, logger=logger.name)
    with patch(
        "bigocrpdf.services.rapidocr_service.backend.ProfessionalPDFOCR",
        return_value=SimpleNamespace(process=process),
    ):
        result = _run_full_ocr(_args(source, output), OCRConfig(), None, logger)

    assert result == 0
    assert "96.8% avg confidence" in caplog.text


def test_cli_ocr_failure_preserves_existing_destination(
    tmp_path: Path,
    capsys,
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source")
    output = tmp_path / "output.pdf"
    output.write_bytes(b"existing PDF")

    def fail(_source, staged, progress_callback):
        Path(staged).write_bytes(b"partial PDF")
        raise RuntimeError("simulated OCR failure")

    engine = SimpleNamespace(process=fail)
    with patch(
        "bigocrpdf.services.rapidocr_service.backend.ProfessionalPDFOCR",
        return_value=engine,
    ):
        result = _run_full_ocr(
            _args(source, output),
            OCRConfig(),
            None,
            _logger(),
        )

    assert result == 1
    assert output.read_bytes() == b"existing PDF"
    assert list(tmp_path.glob(".bigocr_ocr_*")) == []
    assert "Traceback" not in capsys.readouterr().err


def test_cli_split_records_each_part_family_inside_the_pdf(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source")
    output = tmp_path / "output.pdf"
    _write_pdf(output)

    def process(_source, staged, progress_callback):
        staged = Path(staged)
        first = staged.with_name("output-01.pdf")
        second = staged.with_name("output-02.pdf")
        _write_pdf(first)
        _write_pdf(second)
        return ProcessingStats(
            pages_total=2,
            pages_processed=2,
            split_output_files=[str(first), str(second)],
            ocr_document=OcrDocument(
                pages=[
                    OcrPage(1, 100, 100, 300, native_text="first"),
                    OcrPage(2, 100, 100, 300, native_text="second"),
                ]
            ),
        )

    engine = SimpleNamespace(process=process)
    with patch(
        "bigocrpdf.services.rapidocr_service.backend.ProfessionalPDFOCR",
        return_value=engine,
    ):
        result = _run_full_ocr(
            _args(source, output),
            OCRConfig(),
            None,
            _logger(),
        )

    assert result == 0
    for index, part_name in enumerate(("output-01.pdf", "output-02.pdf"), start=1):
        part = tmp_path / part_name
        assert part.exists()
        family = _read_split_family(part)
        assert family is not None
        assert (family.family_root, family.part_index, family.part_count) == (
            "output.pdf",
            index,
            2,
        )
    assert not output.exists()
    assert list(tmp_path.glob("*.json")) == []


def test_load_dewarp_image_closes_pillow_image(monkeypatch, tmp_path: Path) -> None:
    import numpy as np

    from bigocrpdf.cli_ocr_commands import _load_dewarp_image

    class FakeImage:
        size = (4, 3)
        mode = "RGB"
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.closed = True

        def __array__(self, dtype=None, copy=None):
            return np.zeros((3, 4, 3), dtype=dtype or np.uint8)

    image = FakeImage()
    monkeypatch.setattr("PIL.Image.open", lambda _path: image)

    result = _load_dewarp_image(tmp_path / "page.png", 1, [], _logger())

    assert result.shape == (3, 4, 3)
    assert image.closed is True


def test_save_dewarp_images_reports_failed_image_write(monkeypatch, tmp_path: Path) -> None:
    import numpy as np
    import pytest

    from bigocrpdf.cli_ocr_commands import _save_dewarp_images

    image = np.zeros((3, 4, 3), dtype=np.uint8)
    preprocessor = SimpleNamespace(process=lambda value: value)
    monkeypatch.setattr("cv2.imwrite", lambda *_args: False)

    with pytest.raises(OSError, match="Failed to save image"):
        _save_dewarp_images(image, 1, tmp_path, preprocessor, _logger())


def _ocr_stats_with_document() -> "ProcessingStats":
    return ProcessingStats(
        pages_total=1,
        pages_processed=1,
        ocr_document=OcrDocument(
            pages=[OcrPage(1, 100, 100, 300, native_text="page one")],
        ),
    )


def _run_ocr_writing_one_page(args: argparse.Namespace) -> int:
    def process(_source, staged, progress_callback):
        _write_pdf(Path(staged))
        return _ocr_stats_with_document()

    with patch(
        "bigocrpdf.services.rapidocr_service.backend.ProfessionalPDFOCR",
        return_value=SimpleNamespace(process=process),
    ):
        return _run_full_ocr(args, OCRConfig(), None, _logger())


def test_sidecar_json_is_written_only_where_asked(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source")
    output = tmp_path / "output.pdf"
    requested = tmp_path / "elsewhere" / "structured.json"
    requested.parent.mkdir()

    assert _run_ocr_writing_one_page(_args(source, output, str(requested))) == 0

    assert requested.exists()
    assert load_ocr_document_json(requested, output) is not None
    assert not ocr_document_json_path(output).exists()


def test_sidecar_json_without_a_path_uses_the_default_name(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source")
    output = tmp_path / "output.pdf"

    assert _run_ocr_writing_one_page(_args(source, output, "")) == 0

    default_path = ocr_document_json_path(output)
    assert default_path.exists()
    assert load_ocr_document_json(default_path, output) is not None


def test_sidecar_json_is_compact(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source")
    output = tmp_path / "output.pdf"

    assert _run_ocr_writing_one_page(_args(source, output, "")) == 0

    # One line: indentation was 62% of the bytes of an 18-page document.
    assert ocr_document_json_path(output).read_text(encoding="utf-8").count("\n") == 1


def test_sidecar_json_reports_that_split_output_has_no_structured_form(
    tmp_path: Path,
    caplog,
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source")
    output = tmp_path / "output.pdf"

    def process(_source, staged, progress_callback):
        staged = Path(staged)
        parts = [staged.with_name("output-01.pdf"), staged.with_name("output-02.pdf")]
        for part in parts:
            _write_pdf(part)
        return ProcessingStats(
            pages_total=2,
            pages_processed=2,
            split_output_files=[str(part) for part in parts],
        )

    logger = logging.getLogger("test-split-json")
    caplog.set_level(logging.WARNING, logger=logger.name)
    with patch(
        "bigocrpdf.services.rapidocr_service.backend.ProfessionalPDFOCR",
        return_value=SimpleNamespace(process=process),
    ):
        result = _run_full_ocr(_args(source, output, ""), OCRConfig(), None, logger)

    assert result == 0
    assert "--sidecar-json needs one output PDF" in caplog.text
    assert list(tmp_path.glob("*.json")) == []


def test_sidecar_json_refuses_a_document_that_misses_pages(
    tmp_path: Path,
    caplog,
) -> None:
    """Partial structured OCR is not written at all, rather than written wrong."""
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source")
    output = tmp_path / "output.pdf"

    def process(_source, staged, progress_callback):
        _write_pdf(Path(staged), pages=2)
        return ProcessingStats(
            pages_total=2,
            pages_processed=2,
            ocr_document=OcrDocument(
                pages=[OcrPage(1, 100, 100, 300, native_text="only the first page")],
            ),
        )

    logger = logging.getLogger("test-partial-json")
    caplog.set_level(logging.WARNING, logger=logger.name)
    with patch(
        "bigocrpdf.services.rapidocr_service.backend.ProfessionalPDFOCR",
        return_value=SimpleNamespace(process=process),
    ):
        result = _run_full_ocr(_args(source, output, ""), OCRConfig(), None, logger)

    assert result == 0
    assert "does not cover every" in caplog.text
    assert not ocr_document_json_path(output).exists()


def test_sidecar_json_failure_does_not_fail_a_finished_ocr(tmp_path: Path, caplog) -> None:
    """The PDF is the deliverable; the JSON is an extra the user asked for.

    The writer refuses a document that does not cover the published PDF, and
    that refusal used to reach the command's own error handler: a finished OCR
    reported a fatal error with its PDF already on disk.
    """
    from bigocrpdf.cli_ocr_commands import _write_requested_sidecar_json

    published = tmp_path / "output.pdf"
    _write_pdf(published, pages=2)
    one_page = OcrDocument(pages=[OcrPage(1, 100, 100, 300)])
    logger = logging.getLogger("test-sidecar-refusal")
    caplog.set_level(logging.WARNING, logger=logger.name)

    _write_requested_sidecar_json("", [published], one_page, logger)

    assert "not written" in caplog.text
    assert not ocr_document_json_path(published).exists()
