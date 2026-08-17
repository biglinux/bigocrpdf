"""Regression tests for OCR output staging and atomic publication."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pikepdf
import pytest

from bigocrpdf.services.processor import OcrProcessor
from bigocrpdf.services.rapidocr_service.config import (
    OcrDocument,
    OcrPage,
    ProcessingStats,
)


def _write_pdf(path: Path, pages: int = 1) -> None:
    with pikepdf.Pdf.new() as pdf:
        for _index in range(pages):
            pdf.add_blank_page()
        pdf.save(path)


def _pdf_page_count(path: Path) -> int:
    with pikepdf.open(path) as pdf:
        return len(pdf.pages)


def _processor(target: Path, *, overwrite: bool = False) -> OcrProcessor:
    settings = SimpleNamespace(
        processed_files=[],
        overwrite_existing=overwrite,
        display_name=lambda path: Path(path).name,
    )
    with patch("bigocrpdf.services.processor.ModelDiscovery"):
        processor = OcrProcessor(settings)  # type: ignore[arg-type]
    processor._get_output_file_path = MagicMock(return_value=str(target))
    processor._create_ocr_config = MagicMock()
    processor._record_processing_history = MagicMock()
    return processor


def _run_with_engine(
    processor: OcrProcessor,
    engine_process,
) -> tuple[bool, str, list, str]:
    engine = MagicMock()
    engine.process.side_effect = engine_process
    with patch("bigocrpdf.services.processor.RapidOCREngine", return_value=engine):
        return processor._process_single_file(__file__, 0)


def test_engine_failure_leaves_no_partial_output(tmp_path: Path) -> None:
    target = tmp_path / "out.pdf"
    processor = _processor(target)

    def fail(_input: Path, staged: Path, _progress) -> ProcessingStats:
        staged.write_bytes(b"partial")
        raise RuntimeError("engine failed")

    with pytest.raises(RuntimeError, match="engine failed"):
        _run_with_engine(processor, fail)

    assert not target.exists()
    assert list(tmp_path.glob(".bigocr_stage_*")) == []


def test_zero_processed_pages_leaves_no_output(tmp_path: Path) -> None:
    target = tmp_path / "out.pdf"
    processor = _processor(target)

    def no_pages(_input: Path, staged: Path, _progress) -> ProcessingStats:
        staged.write_bytes(b"not a valid result")
        return ProcessingStats(pages_processed=0)

    success, output, _boxes, primary_output = _run_with_engine(processor, no_pages)

    assert success is False
    assert output == ""
    assert primary_output == ""
    assert not target.exists()


def test_invalid_engine_pdf_is_not_published(tmp_path: Path) -> None:
    target = tmp_path / "out.pdf"
    processor = _processor(target)

    def invalid_pdf(_input: Path, staged: Path, _progress) -> ProcessingStats:
        staged.write_bytes(b"not a PDF")
        return ProcessingStats(pages_total=1, pages_processed=1)

    with pytest.raises(ValueError, match="Invalid staged OCR PDF"):
        _run_with_engine(processor, invalid_pdf)

    assert not target.exists()
    assert list(tmp_path.glob("*.json")) == []


def test_success_publishes_complete_output(tmp_path: Path) -> None:
    target = tmp_path / "out.pdf"
    processor = _processor(target)

    def succeed(_input: Path, staged: Path, _progress) -> ProcessingStats:
        _write_pdf(staged)
        return ProcessingStats(pages_total=1, pages_processed=1, full_text="text")

    success, text, _boxes, primary_output = _run_with_engine(processor, succeed)

    assert success is True
    assert text == "text"
    assert _pdf_page_count(target) == 1
    assert primary_output == str(target)
    assert processor.settings.processed_files == [str(target)]
    assert processor.get_total_pages() == 1


def test_success_publishes_the_pdf_and_nothing_beside_it(tmp_path: Path) -> None:
    """The interface produces one file per run, structured data or not."""
    target = tmp_path / "out.pdf"
    processor = _processor(target)
    document = OcrDocument(
        pages=[
            OcrPage(
                page_index=1,
                width_px=100,
                height_px=100,
                dpi=300,
                native_text="structured text",
            )
        ]
    )

    def succeed(_input: Path, staged: Path, _progress) -> ProcessingStats:
        _write_pdf(staged)
        return ProcessingStats(
            pages_total=1,
            pages_processed=1,
            ocr_document=document,
        )

    success, _text, _boxes, _primary_output = _run_with_engine(processor, succeed)

    assert success is True
    assert [entry.name for entry in tmp_path.iterdir()] == ["out.pdf"]


def test_a_json_left_beside_the_destination_is_neither_read_nor_touched(
    tmp_path: Path,
) -> None:
    """Whatever else lives in the folder belongs to the user, not to us.

    A file named after our old default used to force the output to a suffixed
    name, because the PDF and its companion had to be free together. Only the
    PDF decides now, and the stranger is left exactly as it was.
    """
    target = tmp_path / "out.pdf"
    stranger = tmp_path / "out.bigocr.json"
    stranger.write_text("external metadata", encoding="utf-8")
    processor = _processor(target)

    def succeed(_input: Path, staged: Path, _progress) -> ProcessingStats:
        _write_pdf(staged)
        return ProcessingStats(pages_processed=1)

    success, _text, _boxes, _primary_output = _run_with_engine(processor, succeed)

    assert success is True
    assert _pdf_page_count(target) == 1
    assert stranger.read_text(encoding="utf-8") == "external metadata"
    assert processor.settings.processed_files == [str(target)]


def test_collision_preserves_existing_output_and_uses_suffix(tmp_path: Path) -> None:
    target = tmp_path / "out.pdf"
    target.write_bytes(b"existing")
    processor = _processor(target)

    def succeed(_input: Path, staged: Path, _progress) -> ProcessingStats:
        _write_pdf(staged)
        return ProcessingStats(pages_processed=1)

    success, _text, _boxes, _primary_output = _run_with_engine(processor, succeed)

    suffixed = tmp_path / "out-1.pdf"
    assert success is True
    assert target.read_bytes() == b"existing"
    assert _pdf_page_count(suffixed) == 1
    assert processor.settings.processed_files == [str(suffixed)]


def test_overwrite_replaces_existing_output_atomically(tmp_path: Path) -> None:
    target = tmp_path / "out.pdf"
    target.write_bytes(b"existing")
    processor = _processor(target, overwrite=True)

    def succeed(_input: Path, staged: Path, _progress) -> ProcessingStats:
        _write_pdf(staged)
        return ProcessingStats(pages_processed=1)

    success, _text, _boxes, _primary_output = _run_with_engine(processor, succeed)

    assert success is True
    assert _pdf_page_count(target) == 1
    assert processor.settings.processed_files == [str(target)]


def test_split_outputs_publish_as_one_complete_set(tmp_path: Path) -> None:
    target = tmp_path / "out.pdf"
    processor = _processor(target)

    def succeed(_input: Path, staged: Path, _progress) -> ProcessingStats:
        first = staged.with_name("out-part-1.pdf")
        second = staged.with_name("out-part-2.pdf")
        _write_pdf(first)
        _write_pdf(second, pages=2)
        return ProcessingStats(
            pages_processed=2,
            split_output_files=[str(first), str(second)],
        )

    success, _text, _boxes, primary_output = _run_with_engine(processor, succeed)

    assert success is True
    assert _pdf_page_count(tmp_path / "out-part-1.pdf") == 1
    assert _pdf_page_count(tmp_path / "out-part-2.pdf") == 2
    assert processor.settings.processed_files == [
        str(tmp_path / "out-part-1.pdf"),
        str(tmp_path / "out-part-2.pdf"),
    ]
    assert primary_output == str(tmp_path / "out-part-1.pdf")


def test_overwrite_split_retires_the_prior_single_output(
    tmp_path: Path,
) -> None:
    target = tmp_path / "out.pdf"
    _write_pdf(target)
    processor = _processor(target, overwrite=True)

    def succeed(_input: Path, staged: Path, _progress) -> ProcessingStats:
        first = staged.with_name("out-01.pdf")
        second = staged.with_name("out-02.pdf")
        _write_pdf(first)
        _write_pdf(second)
        return ProcessingStats(
            pages_total=2,
            pages_processed=2,
            split_output_files=[str(first), str(second)],
        )

    success, _text, _boxes, _primary_output = _run_with_engine(processor, succeed)

    assert success is True
    assert not target.exists()
    assert list(tmp_path.glob("*.json")) == []
    assert (tmp_path / "out-01.pdf").exists()
    assert (tmp_path / "out-02.pdf").exists()


def test_overwrite_single_retires_the_prior_split_parts(
    tmp_path: Path,
) -> None:
    """Each part is recognised by the family recorded inside the PDF."""
    target = tmp_path / "out.pdf"
    processor = _processor(target, overwrite=True)

    def split(_input: Path, staged: Path, _progress) -> ProcessingStats:
        first = staged.with_name("out-01.pdf")
        second = staged.with_name("out-02.pdf")
        _write_pdf(first)
        _write_pdf(second)
        return ProcessingStats(
            pages_total=2,
            pages_processed=2,
            split_output_files=[str(first), str(second)],
        )

    split_success, _text, _boxes, _primary_output = _run_with_engine(
        processor,
        split,
    )
    assert split_success is True
    prior_parts = [tmp_path / "out-01.pdf", tmp_path / "out-02.pdf"]

    def succeed(_input: Path, staged: Path, _progress) -> ProcessingStats:
        _write_pdf(staged)
        return ProcessingStats(pages_total=1, pages_processed=1)

    success, _text, _boxes, _primary_output = _run_with_engine(processor, succeed)

    assert success is True
    assert target.exists()
    assert all(not part.exists() for part in prior_parts)
    assert list(tmp_path.glob("*.json")) == []


def test_overwrite_does_not_retire_an_unrelated_numbered_collision(
    tmp_path: Path,
) -> None:
    """A numbered name is not proof of membership; only the metadata is."""
    target = tmp_path / "out.pdf"
    target.write_bytes(b"existing root output")
    for counter in range(1, 10):
        (tmp_path / f"out-{counter}.pdf").write_bytes(b"existing collision")

    def succeed(_input: Path, staged: Path, _progress) -> ProcessingStats:
        _write_pdf(staged)
        return ProcessingStats(pages_total=1, pages_processed=1)

    collision_processor = _processor(target)
    collision_success, _text, _boxes, _primary_output = _run_with_engine(
        collision_processor,
        succeed,
    )
    assert collision_success is True
    unrelated_pdf = tmp_path / "out-10.pdf"
    old_pdf = unrelated_pdf.read_bytes()

    overwrite_processor = _processor(target, overwrite=True)
    success, _text, _boxes, _primary_output = _run_with_engine(
        overwrite_processor,
        succeed,
    )

    assert success is True
    assert unrelated_pdf.read_bytes() == old_pdf


def test_success_marks_checkpoint_without_ui_callback(tmp_path: Path) -> None:
    processor = _processor(tmp_path / "out.pdf")
    processor.on_file_complete = None
    checkpoint = MagicMock()
    processor.settings.processed_files = [str(tmp_path / "out.pdf")]

    processor._record_file_processing_result(
        "input.pdf",
        str(tmp_path / "out.pdf"),
        True,
        "text",
        [],
        checkpoint,
    )

    checkpoint.mark_file_completed.assert_called_once_with(
        "input.pdf",
        str(tmp_path / "out.pdf"),
    )


def test_split_uses_first_part_as_primary_and_counts_one_input(
    tmp_path: Path,
) -> None:
    processor = _processor(tmp_path / "out.pdf")
    checkpoint = MagicMock()
    callback = MagicMock()
    processor.on_file_complete = callback

    def succeed(_input: Path, staged: Path, _progress) -> ProcessingStats:
        first = staged.with_name("out-part-1.pdf")
        second = staged.with_name("out-part-2.pdf")
        _write_pdf(first)
        _write_pdf(second)
        return ProcessingStats(
            pages_total=2,
            pages_processed=2,
            split_output_files=[str(first), str(second)],
        )

    engine = MagicMock()
    engine.process.side_effect = succeed
    with patch("bigocrpdf.services.processor.RapidOCREngine", return_value=engine):
        should_continue = processor._process_file_with_checkpoint(
            __file__,
            0,
            checkpoint,
        )

    first_output = str(tmp_path / "out-part-1.pdf")
    second_output = str(tmp_path / "out-part-2.pdf")
    assert should_continue is True
    assert processor.settings.processed_files == [first_output, second_output]
    checkpoint.mark_file_completed.assert_called_once_with(__file__, first_output)
    callback.assert_called_once_with(__file__, first_output, "", [])
    assert processor._record_processing_history.call_args.args[1] == first_output
    assert processor.get_completed_input_count() == 1
    assert processor.get_successful_input_count() == 1
    assert processor.get_processed_count() == 1


def test_terminal_failure_counts_completed_but_not_successful_input(
    tmp_path: Path,
) -> None:
    processor = _processor(tmp_path / "out.pdf")
    checkpoint = MagicMock()
    processor.on_file_complete = MagicMock()

    with patch.object(
        processor,
        "_process_single_file",
        side_effect=RuntimeError("simulated failure"),
    ):
        should_continue = processor._process_file_with_checkpoint(
            __file__,
            0,
            checkpoint,
        )

    assert should_continue is True
    assert processor.get_completed_input_count() == 1
    assert processor.get_successful_input_count() == 0
    assert processor.get_processed_count() == 0
    checkpoint.mark_file_failed.assert_called_once()
