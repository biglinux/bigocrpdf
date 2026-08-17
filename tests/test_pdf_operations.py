"""Tests for pdf_operations module."""

import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pikepdf
import pytest

from bigocrpdf.services import pdf_operations
from bigocrpdf.services.pdf_operations import (
    PDFInfo,
    extract_pages,
    get_pdf_info,
    insert_pages,
    merge_pdfs,
    reorder_pages,
    reverse_pages,
    rotate_pages,
    split_by_pages,
    split_by_ranges,
    split_by_size,
)
from bigocrpdf.utils.durable_writes import recover_pending_publications


def _create_test_pdf(path: str, num_pages: int = 3) -> str:
    """Create a simple test PDF with the given number of pages."""
    pdf = pikepdf.Pdf.new()
    for i in range(num_pages):
        page = pikepdf.Page(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Page,
                MediaBox=[0, 0, 612, 792],
                Resources=pikepdf.Dictionary(),
                Contents=pdf.make_stream(f"BT /F1 12 Tf 100 700 Td (Page {i + 1}) Tj ET".encode()),
            )
        )
        pdf.pages.append(page)
    pdf.save(path)
    return path


class TestGetPdfInfo:
    def test_basic_info(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name
        try:
            _create_test_pdf(path, 5)
            info = get_pdf_info(path)
            assert isinstance(info, PDFInfo)
            assert info.page_count == 5
            assert info.file_size_bytes > 0
            assert info.path == path
        finally:
            os.unlink(path)

    def test_nonexistent_file_raises(self):
        with pytest.raises((FileNotFoundError, OSError)):
            get_pdf_info("/nonexistent/file.pdf")


def test_atomic_pdf_save_preserves_existing_destination_on_failure(tmp_path):
    destination = tmp_path / "existing.pdf"
    destination.write_bytes(b"existing")
    pdf = MagicMock()

    def fail_after_partial_write(path, **_options):
        Path(path).write_bytes(b"partial")
        raise OSError("simulated save failure")

    pdf.save.side_effect = fail_after_partial_write

    with pytest.raises(OSError, match="simulated save failure"):
        pdf_operations._save_pdf_atomically(pdf, destination)

    assert destination.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [destination]


def test_atomic_pdf_save_publishes_the_pdf_and_nothing_else(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "output.pdf"
    _create_test_pdf(str(destination), 1)
    replacement = pikepdf.Pdf.new()
    replacement.add_blank_page()

    try:
        pdf_operations._save_pdf_atomically(replacement, destination)
    finally:
        replacement.close()

    assert [entry.name for entry in tmp_path.iterdir()] == ["output.pdf"]


def test_atomic_new_pdf_save_closes_document_on_failure(tmp_path):
    pdf = MagicMock()

    with (
        patch.object(pdf_operations, "_save_pdf_atomically", side_effect=OSError("failure")),
        pytest.raises(OSError, match="failure"),
    ):
        pdf_operations._save_and_close_pdf_atomically(pdf, tmp_path / "output.pdf")

    pdf.close.assert_called_once_with()


def test_interrupted_pdf_batch_recovers_qpdf_valid_originals(tmp_path: Path) -> None:
    if shutil.which("qpdf") is None:
        pytest.skip("qpdf is required for PDF integrity validation")

    first_target = tmp_path / "first.pdf"
    second_target = tmp_path / "second.pdf"
    first_staged = tmp_path / ".first.staged"
    second_staged = tmp_path / ".second.staged"
    _create_test_pdf(str(first_target), 1)
    _create_test_pdf(str(second_target), 2)
    _create_test_pdf(str(first_staged), 3)
    _create_test_pdf(str(second_staged), 4)
    child_code = """
import os
import sys
from pathlib import Path
from bigocrpdf.utils import durable_writes

root = Path(sys.argv[1])
first_target = root / "first.pdf"
real_rename = durable_writes._rename_without_replacement

def rename_and_die(source, destination):
    real_rename(source, destination)
    if Path(source).name.endswith(".0.new") and Path(destination) == first_target:
        os._exit(84)

durable_writes._rename_without_replacement = rename_and_die
durable_writes.publish_files_transactionally(
    [
        (root / ".first.staged", root / "first.pdf"),
        (root / ".second.staged", root / "second.pdf"),
    ],
    overwrite=True,
)
"""
    repo_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root / "src")
    child = subprocess.run(
        [sys.executable, "-c", child_code, str(tmp_path)],
        capture_output=True,
        cwd=repo_root,
        env=environment,
        text=True,
        timeout=10,
        check=False,
    )
    assert child.returncode == 84, child.stderr

    recover_pending_publications(tmp_path)

    for target, expected_pages in ((first_target, 1), (second_target, 2)):
        with pikepdf.open(target) as pdf:
            assert len(pdf.pages) == expected_pages
        qpdf_check = subprocess.run(
            ["qpdf", "--check", str(target)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert qpdf_check.returncode == 0, qpdf_check.stderr
    assert list(tmp_path.glob(".bigocr-publish-*")) == []


class TestSplitByPages:
    def test_split_preserves_staged_output_access_mode(self, tmp_path: Path):
        source = tmp_path / "source.pdf"
        output_dir = tmp_path / "parts"
        _create_test_pdf(str(source), 1)
        real_save = pdf_operations._save_split_part

        def save_with_explicit_mode(pdf, output_path):
            real_save(pdf, output_path)
            Path(output_path).chmod(0o644)

        with patch.object(
            pdf_operations,
            "_save_split_part",
            side_effect=save_with_explicit_mode,
        ):
            result = split_by_pages(source, output_dir, pages_per_file=1)

        assert len(result.output_files) == 1
        assert stat.S_IMODE(Path(result.output_files[0]).stat().st_mode) == 0o644

    def test_split_3_pages_into_1_per_file(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name
        with tempfile.TemporaryDirectory() as out_dir:
            try:
                _create_test_pdf(path, 3)
                result = split_by_pages(path, out_dir, pages_per_file=1)
                assert len(result.output_files) == 3
                assert result.total_pages == 3
                for out_file in result.output_files:
                    assert os.path.exists(out_file)
                # Splitting produces PDFs only.
                assert list(Path(out_dir).glob("*.json")) == []
            finally:
                os.unlink(path)

    def test_split_all_in_one(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name
        with tempfile.TemporaryDirectory() as out_dir:
            try:
                _create_test_pdf(path, 3)
                result = split_by_pages(path, out_dir, pages_per_file=10)
                assert len(result.output_files) == 1
            finally:
                os.unlink(path)

    def test_collision_on_one_part_suffixes_the_entire_batch(
        self,
        tmp_path: Path,
    ) -> None:
        source = tmp_path / "source.pdf"
        output_dir = tmp_path / "parts"
        output_dir.mkdir()
        _create_test_pdf(str(source), 3)
        existing = output_dir / "document_part002.pdf"
        _create_test_pdf(str(existing), 5)

        result = split_by_pages(
            source,
            output_dir,
            pages_per_file=1,
            prefix="document",
        )

        assert result.parts == 3
        assert [Path(path).name for path in result.output_files] == [
            "document_part001-1.pdf",
            "document_part002-1.pdf",
            "document_part003-1.pdf",
        ]
        assert get_pdf_info(existing).page_count == 5
        assert all(Path(path).exists() for path in result.output_files)

    def test_size_split_reports_only_actual_published_paths(
        self,
        tmp_path: Path,
    ) -> None:
        source = tmp_path / "source.pdf"
        output_dir = tmp_path / "parts"
        output_dir.mkdir()
        _create_test_pdf(str(source), 3)
        existing = output_dir / "document_part002.pdf"
        _create_test_pdf(str(existing), 5)

        result = split_by_size(
            source,
            output_dir,
            max_size_mb=0.0001,
            prefix="document",
        )

        assert result.parts == 3
        assert [Path(path).name for path in result.output_files] == [
            "document_part001-1.pdf",
            "document_part002-1.pdf",
            "document_part003-1.pdf",
        ]
        assert all(Path(path).exists() for path in result.output_files)

    def test_range_split_reports_only_actual_published_paths(
        self,
        tmp_path: Path,
    ) -> None:
        source = tmp_path / "source.pdf"
        output_dir = tmp_path / "parts"
        output_dir.mkdir()
        _create_test_pdf(str(source), 3)
        existing = output_dir / "document_pages2-3.pdf"
        _create_test_pdf(str(existing), 5)

        result = split_by_ranges(
            source,
            output_dir,
            [(1, 1), (2, 3)],
            prefix="document",
        )

        assert result.parts == 2
        assert [Path(path).name for path in result.output_files] == [
            "document_pages1-1-1.pdf",
            "document_pages2-3-1.pdf",
        ]
        assert all(Path(path).exists() for path in result.output_files)

    def test_generation_failure_publishes_no_partial_parts(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name
        with tempfile.TemporaryDirectory() as out_dir:
            try:
                _create_test_pdf(path, 3)
                keep = os.path.join(out_dir, "keep.txt")
                with open(keep, "w", encoding="utf-8") as stream:
                    stream.write("keep")
                original_save = pdf_operations._save_split_part
                saves = 0

                def fail_second_save(pdf, output_path):
                    nonlocal saves
                    saves += 1
                    if saves == 2:
                        pdf.close()
                        raise OSError("simulated write failure")
                    original_save(pdf, output_path)

                with (
                    patch.object(pdf_operations, "_save_split_part", side_effect=fail_second_save),
                    pytest.raises(OSError, match="simulated write failure"),
                ):
                    split_by_pages(path, out_dir, pages_per_file=1)

                assert os.listdir(out_dir) == ["keep.txt"]
            finally:
                os.unlink(path)


class TestExtractPages:
    def test_extract_single_page(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            src = f.name
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            dst = f.name
        try:
            _create_test_pdf(src, 5)
            result = extract_pages(src, dst, [2])
            assert result.success is True
            assert result.pages_affected == 1
            info = get_pdf_info(dst)
            assert info.page_count == 1
        finally:
            os.unlink(src)
            if os.path.exists(dst):
                os.unlink(dst)

    def test_extract_multiple_pages(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            src = f.name
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            dst = f.name
        try:
            _create_test_pdf(src, 5)
            result = extract_pages(src, dst, [1, 3, 5])
            assert result.success is True
            assert result.pages_affected == 3
        finally:
            os.unlink(src)
            if os.path.exists(dst):
                os.unlink(dst)

    def test_invalid_pages_do_not_allocate_output_pdf(self, tmp_path):
        source = tmp_path / "source.pdf"
        _create_test_pdf(str(source), 1)

        with patch.object(pdf_operations.pikepdf.Pdf, "new") as new_pdf:
            result = extract_pages(source, tmp_path / "output.pdf", [99])

        assert result.success is False
        new_pdf.assert_not_called()


class TestMergePdfs:
    def test_merge_two_pdfs(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f1:
            p1 = f1.name
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f2:
            p2 = f2.name
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fo:
            out = fo.name
        try:
            _create_test_pdf(p1, 2)
            _create_test_pdf(p2, 3)
            result = merge_pdfs([p1, p2], out)
            assert result.success is True
            info = get_pdf_info(out)
            assert info.page_count == 5
        finally:
            for p in (p1, p2, out):
                if os.path.exists(p):
                    os.unlink(p)

    def test_all_missing_inputs_do_not_replace_existing_output(self, tmp_path):
        output = tmp_path / "existing.pdf"
        output.write_bytes(b"existing")

        result = merge_pdfs([tmp_path / "missing.pdf"], output)

        assert result.success is False
        assert output.read_bytes() == b"existing"


class TestRotatePages:
    def test_rotate_90(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            src = f.name
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            dst = f.name
        try:
            _create_test_pdf(src, 2)
            result = rotate_pages(src, dst, [1], 90)
            assert result.success is True
        finally:
            os.unlink(src)
            if os.path.exists(dst):
                os.unlink(dst)


class TestReorderPages:
    def test_reverse_order(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            src = f.name
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            dst = f.name
        try:
            _create_test_pdf(src, 3)
            result = reorder_pages(src, dst, [3, 2, 1])
            assert result.success is True
            info = get_pdf_info(dst)
            assert info.page_count == 3
        finally:
            os.unlink(src)
            if os.path.exists(dst):
                os.unlink(dst)


class TestReversePages:
    def test_reverse(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            src = f.name
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            dst = f.name
        try:
            _create_test_pdf(src, 4)
            result = reverse_pages(src, dst)
            assert result.success is True
            info = get_pdf_info(dst)
            assert info.page_count == 4
        finally:
            os.unlink(src)
            if os.path.exists(dst):
                os.unlink(dst)


def test_atomic_pdf_save_preserves_existing_destination_mode(tmp_path: Path) -> None:
    destination = tmp_path / "existing.pdf"
    _create_test_pdf(str(destination), 1)
    destination.chmod(0o640)
    replacement = pikepdf.Pdf.new()
    replacement.add_blank_page()

    try:
        pdf_operations._save_pdf_atomically(replacement, destination)
    finally:
        replacement.close()

    assert stat.S_IMODE(destination.stat().st_mode) == 0o640


def test_rotate_duplicate_page_only_once(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "rotated.pdf"
    _create_test_pdf(str(source), 1)

    result = rotate_pages(source, output, [1, 1], 90)

    assert result.success is True
    assert result.pages_affected == 1
    with pikepdf.open(output) as pdf:
        assert int(pdf.pages[0].get("/Rotate", 0)) == 90


def test_insert_empty_source_page_list_inserts_nothing(tmp_path: Path) -> None:
    target = tmp_path / "target.pdf"
    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    _create_test_pdf(str(target), 1)
    _create_test_pdf(str(source), 2)

    result = insert_pages(target, source, output, source_pages=[])

    assert result.success is False
    assert result.pages_affected == 0
    assert not output.exists()


def test_atomic_pdf_save_cleans_staged_file_when_mode_copy_fails(tmp_path: Path) -> None:
    destination = tmp_path / "existing.pdf"
    destination.write_bytes(b"existing")
    destination.chmod(0o640)
    pdf = MagicMock()

    with (
        patch.object(pdf_operations.os, "fchmod", side_effect=OSError("chmod failed")),
        pytest.raises(OSError, match="chmod failed"),
    ):
        pdf_operations._save_pdf_atomically(pdf, destination)

    pdf.save.assert_not_called()
    assert list(tmp_path.iterdir()) == [destination]
