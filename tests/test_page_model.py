"""Tests for page_model module (PDFDocument and PageState)."""

from bigocrpdf.ui.pdf_editor.page_model import PageState, PDFDocument


class TestPageState:
    def test_default_values(self):
        ps = PageState(page_number=1)
        assert ps.rotation == 0
        assert ps.included_for_ocr is True
        assert ps.deleted is False
        assert ps.position == 0

    def test_rotation_normalization(self):
        ps = PageState(page_number=1, rotation=450)
        assert ps.rotation == 90

    def test_invalid_rotation_rounds(self):
        ps = PageState(page_number=1, rotation=45)
        # Should round to nearest valid (0, 90, 180, 270)
        assert ps.rotation in (0, 90)

    def test_rotate_right(self):
        ps = PageState(page_number=1)
        ps.rotate_right()
        assert ps.rotation == 90
        ps.rotate_right()
        assert ps.rotation == 180

    def test_rotate_left(self):
        ps = PageState(page_number=1)
        ps.rotate_left()
        assert ps.rotation == 270

    def test_rotate_degrees(self):
        ps = PageState(page_number=1)
        ps.rotate(180)
        assert ps.rotation == 180

    def test_to_dict_roundtrip(self):
        ps = PageState(page_number=3, rotation=90, included_for_ocr=False, position=2)
        d = ps.to_dict()
        restored = PageState.from_dict(d)
        assert restored.page_number == 3
        assert restored.rotation == 90
        assert restored.included_for_ocr is False
        assert restored.position == 2


class TestPDFDocument:
    def test_auto_creates_pages(self):
        doc = PDFDocument(path="/test.pdf", total_pages=5)
        assert len(doc.pages) == 5
        assert doc.pages[0].page_number == 1
        assert doc.pages[4].page_number == 5

    def test_get_active_pages_excludes_deleted(self):
        doc = PDFDocument(path="/test.pdf", total_pages=3)
        doc.pages[1].deleted = True
        active = doc.get_active_pages()
        assert len(active) == 2
        assert all(not p.deleted for p in active)

    def test_get_page_by_position(self):
        doc = PDFDocument(path="/test.pdf", total_pages=3)
        page = doc.get_page_by_position(0)
        assert page is not None
        assert page.page_number == 1

    def test_get_page_by_invalid_position(self):
        doc = PDFDocument(path="/test.pdf", total_pages=3)
        assert doc.get_page_by_position(99) is None

    def test_mark_modified(self):
        doc = PDFDocument(path="/test.pdf", total_pages=1)
        assert doc.modified is False
        doc.mark_modified()
        assert doc.modified is True

    def test_clear_modifications(self):
        doc = PDFDocument(path="/test.pdf", total_pages=1)
        doc.mark_modified()
        doc.clear_modifications()
        assert doc.modified is False

    def test_update_positions(self):
        doc = PDFDocument(path="/test.pdf", total_pages=3)
        doc.pages[0].deleted = True
        doc.update_positions()
        active = doc.get_active_pages()
        assert active[0].position == 0
        assert active[1].position == 1

    def test_to_dict_roundtrip(self):
        doc = PDFDocument(path="/test.pdf", total_pages=2)
        doc.pages[0].rotation = 90
        d = doc.to_dict()
        restored = PDFDocument.from_dict(d)
        assert restored.path == "/test.pdf"
        assert restored.total_pages == 2
        assert restored.pages[0].rotation == 90
