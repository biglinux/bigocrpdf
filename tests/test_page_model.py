"""Tests for page_model module (PDFDocument and PageState)."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

from bigocrpdf.ui.pdf_editor.page_grid import PageGrid
from bigocrpdf.ui.pdf_editor.page_model import PageState, PDFDocument
from bigocrpdf.ui.pdf_editor.page_operations import (
    delete_pages,
    deselect_all_for_ocr,
    rotate_pages,
    select_all_for_ocr,
    set_ocr_selection,
)
from bigocrpdf.ui.pdf_editor.page_thumbnail import PageThumbnail


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

    def test_quarter_turn_swaps_display_axis_flips(self):
        ps = PageState(
            page_number=1,
            flip_horizontal=True,
            flip_vertical=False,
        )

        ps.rotate_right()

        assert ps.rotation == 90
        assert ps.flip_horizontal is False
        assert ps.flip_vertical is True

    def test_half_turn_preserves_display_axis_flips(self):
        ps = PageState(
            page_number=1,
            flip_horizontal=True,
            flip_vertical=False,
        )

        ps.rotate(180)

        assert ps.rotation == 180
        assert ps.flip_horizontal is True
        assert ps.flip_vertical is False

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

    def test_restore_derives_total_pages_from_serialized_pages(self):
        restored = PDFDocument.from_dict(
            {
                "path": "/test.pdf",
                "total_pages": 99,
                "pages": [PageState(page_number=1).to_dict()],
            }
        )

        assert restored.total_pages == 1

    def test_restore_preserves_explicitly_empty_page_list(self):
        restored = PDFDocument.from_dict(
            {
                "path": "/test.pdf",
                "total_pages": 99,
                "pages": [],
            }
        )

        assert restored.pages == []
        assert restored.total_pages == 0

    def test_hard_delete_updates_total_pages(self):
        doc = PDFDocument(path="/test.pdf", total_pages=3)

        assert delete_pages(doc, [1], hard_delete=True) is True

        assert len(doc.pages) == 2
        assert doc.total_pages == 2


def test_rotate_pages_delegates_normalized_delta_to_page_state() -> None:
    doc = PDFDocument(path="/test.pdf", total_pages=2)

    assert rotate_pages(doc, [0], -90) is True
    assert rotate_pages(doc, [1], 180) is True

    assert [page.rotation for page in doc.pages] == [270, 180]
    assert doc.modified is True


def test_page_operations_ignore_duplicates_and_report_real_mutations() -> None:
    doc = PDFDocument(path="/test.pdf", total_pages=2)

    assert rotate_pages(doc, [0, 0, 99], 90) is True
    assert [page.rotation for page in doc.pages] == [90, 0]
    doc.clear_modifications()

    assert delete_pages(doc, [1, 1], hard_delete=True) is True
    assert doc.total_pages == 1
    assert len(doc.pages) == 1
    doc.clear_modifications()

    assert rotate_pages(doc, [99], 90) is False
    assert rotate_pages(doc, [0], 0) is False
    assert delete_pages(doc, [99]) is False
    assert set_ocr_selection(doc, [99], False) is False
    assert doc.modified is False


def test_ocr_selection_helpers_do_not_mark_no_ops_as_modifications() -> None:
    doc = PDFDocument(path="/test.pdf", total_pages=1)

    assert select_all_for_ocr(doc) is False
    assert set_ocr_selection(doc, [0], True) is False
    assert doc.modified is False

    assert deselect_all_for_ocr(doc) is True
    doc.clear_modifications()
    assert deselect_all_for_ocr(doc) is False
    assert doc.modified is False


def test_page_grid_skips_undo_and_events_for_ocr_no_ops() -> None:
    page = PageState(page_number=1)
    update_from_state = MagicMock()
    thumbnail = cast(
        PageThumbnail,
        SimpleNamespace(page_state=page, update_from_state=update_from_state),
    )
    grid = PageGrid.__new__(PageGrid)
    grid._thumbnails = [thumbnail]
    grid._selected_indices = set()
    grid._document = PDFDocument(path="/test.pdf", total_pages=1)
    grid.on_before_mutate = MagicMock()
    grid.emit = MagicMock()

    grid.set_ocr_for_all(True)
    grid.toggle_ocr_for_selected()

    grid.on_before_mutate.assert_not_called()
    grid.emit.assert_not_called()
    update_from_state.assert_not_called()
    assert grid._document.modified is False


def test_page_grid_records_one_undo_for_real_bulk_ocr_mutation() -> None:
    page = PageState(page_number=1)
    update_from_state = MagicMock()
    thumbnail = cast(
        PageThumbnail,
        SimpleNamespace(page_state=page, update_from_state=update_from_state),
    )
    grid = PageGrid.__new__(PageGrid)
    grid._thumbnails = [thumbnail]
    grid._selected_indices = {0, 99}
    grid._document = PDFDocument(path="/test.pdf", total_pages=1)
    grid.on_before_mutate = MagicMock()
    grid.emit = MagicMock()

    grid.toggle_ocr_for_selected()

    grid.on_before_mutate.assert_called_once_with()
    update_from_state.assert_called_once_with()
    assert page.deleted is True
    assert page.included_for_ocr is False
    assert grid._document.modified is True
