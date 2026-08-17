"""Tests for crash-safe standalone editor saves."""

import argparse
import logging
import stat
import tempfile
from pathlib import Path
from unittest.mock import patch

import pikepdf
import pytest

from bigocrpdf.cli_editor_commands import _cmd_edit, _standalone_save
from bigocrpdf.ui.pdf_editor.page_model import PDFDocument


def _write_pdf(path: str | Path) -> None:
    with pikepdf.Pdf.new() as pdf:
        pdf.add_blank_page()
        pdf.save(path)


def test_edit_command_checks_runtime_before_importing_editor(tmp_path: Path) -> None:
    real_import = __import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"gi.repository", "bigocrpdf.ui.pdf_editor.editor_window"}:
            raise AssertionError("editor imported before runtime validation")
        return real_import(name, globals, locals, fromlist, level)

    with (
        patch("bigocrpdf._check_gtk_dependencies", return_value=False),
        patch("builtins.__import__", side_effect=guarded_import),
    ):
        result = _cmd_edit(
            argparse.Namespace(input=tmp_path / "document.pdf"),
            logging.getLogger("test"),
        )

    assert result == 1


def test_standalone_save_stages_beside_original_before_publication(
    tmp_path: Path,
) -> None:
    original = tmp_path / "document.pdf"
    original.write_bytes(b"original")
    document = PDFDocument(path=str(original), total_pages=1)
    real_mkstemp = tempfile.mkstemp
    allocation_directories: list[Path] = []

    def allocate_beside_original(*args, **kwargs):
        allocation_directories.append(Path(kwargs["dir"]))
        return real_mkstemp(*args, **kwargs)

    def write_edited_pdf(_document, output_path) -> bool:
        _write_pdf(output_path)
        return True

    with (
        patch("tempfile.mkstemp", side_effect=allocate_beside_original),
        patch(
            "bigocrpdf.ui.pdf_editor.page_operations.apply_changes_to_pdf",
            side_effect=write_edited_pdf,
        ),
    ):
        saved = _standalone_save(document, original, logging.getLogger("test"))

    assert saved is True
    assert allocation_directories == [tmp_path]
    with pikepdf.open(original) as published:
        assert len(published.pages) == 1
    # Editing publishes the PDF and nothing else: no companion file is born,
    # and none has to be invalidated.
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["document.pdf"]


def test_standalone_save_preserves_original_access_mode(tmp_path: Path) -> None:
    original = tmp_path / "document.pdf"
    _write_pdf(original)
    original.chmod(0o640)
    document = PDFDocument(path=str(original), total_pages=1)

    def write_edited_pdf(_document, output_path) -> bool:
        _write_pdf(output_path)
        return True

    with patch(
        "bigocrpdf.ui.pdf_editor.page_operations.apply_changes_to_pdf",
        side_effect=write_edited_pdf,
    ):
        saved = _standalone_save(document, original, logging.getLogger("test"))

    assert saved is True
    assert stat.S_IMODE(original.stat().st_mode) == 0o640


def test_standalone_materialization_failure_reports_unsaved(
    tmp_path: Path,
) -> None:
    original = tmp_path / "document.pdf"
    original.write_bytes(b"original")
    document = PDFDocument(path=str(original), total_pages=1)

    with patch(
        "bigocrpdf.ui.pdf_editor.page_operations.apply_changes_to_pdf",
        return_value=False,
    ):
        saved = _standalone_save(document, original, logging.getLogger("test"))

    assert saved is False
    assert original.read_bytes() == b"original"
    assert list(tmp_path.glob("bigocr_edit_*")) == []


def test_standalone_publication_failure_preserves_original_and_removes_stage(
    tmp_path: Path,
) -> None:
    original = tmp_path / "document.pdf"
    original.write_bytes(b"original")
    document = PDFDocument(path=str(original), total_pages=1)

    def write_edited_pdf(_document, output_path) -> bool:
        Path(output_path).write_bytes(b"partial replacement")
        return True

    with (
        patch(
            "bigocrpdf.ui.pdf_editor.page_operations.apply_changes_to_pdf",
            side_effect=write_edited_pdf,
        ),
        patch(
            "bigocrpdf.utils.durable_writes.publish_file_atomically",
            side_effect=OSError("simulated publication failure"),
        ),
        pytest.raises(OSError, match="simulated publication failure"),
    ):
        _standalone_save(document, original, logging.getLogger("test"))

    assert original.read_bytes() == b"original"
    assert list(tmp_path.glob("bigocr_edit_*")) == []
