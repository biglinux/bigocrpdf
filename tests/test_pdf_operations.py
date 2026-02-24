"""Tests for pdf_operations module."""

import os
import tempfile

import pikepdf

from bigocrpdf.services.pdf_operations import (
    OperationResult,
    PDFInfo,
    extract_pages,
    get_pdf_info,
    merge_pdfs,
    reorder_pages,
    reverse_pages,
    rotate_pages,
    split_by_pages,
)


def _create_test_pdf(path: str, num_pages: int = 3) -> str:
    """Create a simple test PDF with the given number of pages."""
    pdf = pikepdf.Pdf.new()
    for i in range(num_pages):
        page = pikepdf.Page(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Page,
                MediaBox=[0, 0, 612, 792],
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
        try:
            get_pdf_info("/nonexistent/file.pdf")
            assert False, "Should have raised"
        except Exception:
            pass


class TestSplitByPages:
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
