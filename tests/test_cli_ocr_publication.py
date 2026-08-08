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
    load_ocr_document_sidecar,
    ocr_document_sidecar_path,
    save_ocr_document_sidecar,
)


def _write_pdf(path: Path, pages: int = 1) -> None:
    with pikepdf.Pdf.new() as pdf:
        for _index in range(pages):
            pdf.add_blank_page()
        pdf.save(path)


def _pdf_page_count(path: Path) -> int:
    with pikepdf.open(path) as pdf:
        return len(pdf.pages)


def _args(source: Path, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        input=source,
        output=output,
        save_preprocessed=None,
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
    save_ocr_document_sidecar(OcrDocument(), output)
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
    assert ocr_document_sidecar_path(output).exists()
    assert load_ocr_document_sidecar(output) is None
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
    save_ocr_document_sidecar(
        OcrDocument(
            pages=[
                OcrPage(
                    page_index=1,
                    width_px=100,
                    height_px=100,
                    dpi=300,
                    native_text="existing text",
                )
            ]
        ),
        output,
    )
    existing_sidecar = ocr_document_sidecar_path(output).read_bytes()

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
    assert ocr_document_sidecar_path(output).read_bytes() == existing_sidecar
    assert list(tmp_path.glob(".bigocr_ocr_*")) == []
    assert "Traceback" not in capsys.readouterr().err


def test_cli_split_publishes_per_part_sidecar_invalidation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source")
    output = tmp_path / "output.pdf"
    _write_pdf(output)
    save_ocr_document_sidecar(OcrDocument(), output)

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
    for part_name in ("output-01.pdf", "output-02.pdf"):
        part = tmp_path / part_name
        assert part.exists()
        assert ocr_document_sidecar_path(part).exists()
        assert load_ocr_document_sidecar(part) is None
    assert not output.exists()
    assert not ocr_document_sidecar_path(output).exists()


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
