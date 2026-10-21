"""Standalone vs overlay: which image a page is built from.

This decides whether the processed image replaces the original. Get it wrong
and OCR coordinates measured on a corrected image are drawn over an
uncorrected one, displacing every word on the page.

Rule for this file: never reimplement a production formula. Call it.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bigocrpdf.services.ocr.backend_text_layer import BackendTextLayerMixin
from bigocrpdf.services.ocr.config import ProcessingStats
from bigocrpdf.services.ocr.pipeline_chunked_ocr import ChunkedOCRMixin


class _ModeBackend(BackendTextLayerMixin):
    """Minimal host for the mixin; the pattern is used by other tests too."""

    def __init__(self, export_format: str = "original") -> None:
        self.config = SimpleNamespace(image_export_format=export_format)


def determine_mode(result: dict, proc_w: int, proc_h: int, export_format: str = "original"):
    return _ModeBackend(export_format)._determine_page_mode(result, proc_w, proc_h)


def _page(**overrides) -> dict:
    base = {
        "page_num": 1,
        "orig_w": 2814,
        "orig_h": 3864,
        "orientation_angle": 0,
        "image_prerotated": False,
        "original_pdf_rotation": 0,
        "geometry_applied": False,
    }
    base.update(overrides)
    return base


class TestDimensionThreshold:
    def test_deskew_adding_small_borders_is_not_significant(self):
        """Real numbers from contrato.pdf: 2814x3864 -> 2939x3954, ~3%."""
        use_processed, geometry_changed = determine_mode(_page(), 2939, 3954)

        assert (use_processed, geometry_changed) == (False, False)

    def test_perspective_correction_is_significant(self):
        """Real numbers from the prescription photo: 1920x2560 -> 1495x2114, ~19%."""
        result = _page(orig_w=1920, orig_h=2560)

        use_processed, geometry_changed = determine_mode(result, 1495, 2114)

        assert (use_processed, geometry_changed) == (True, True)

    def test_identical_dimensions_are_not_significant(self):
        assert determine_mode(_page(), 2814, 3864) == (False, False)

    def test_threshold_is_strictly_greater_than_five_percent(self):
        """Exactly 5% must not trip it; just above must."""
        orig_w, orig_h = 1000, 1000
        at_threshold = int(orig_w + 0.05 * (orig_w + orig_h))
        result = _page(orig_w=orig_w, orig_h=orig_h)

        assert determine_mode(result, at_threshold, orig_h)[1] is False
        assert determine_mode(result, at_threshold + 1, orig_h)[1] is True

    def test_zero_sized_page_does_not_divide_by_zero(self):
        assert determine_mode(_page(orig_w=0, orig_h=0), 0, 0) == (False, False)


class TestCoordinateSpaceChanges:
    def test_geometry_applied_forces_standalone_without_a_size_change(self):
        """A perspective warp can keep the dimensions while moving every pixel,
        so the flag alone must force standalone mode.
        """
        result = _page(geometry_applied=True)

        assert determine_mode(result, 2814, 3864) == (True, True)

    def test_orientation_angle_forces_standalone(self):
        assert determine_mode(_page(orientation_angle=90), 2814, 3864) == (True, True)

    def test_prerotated_image_on_a_rotated_page_forces_standalone(self):
        result = _page(image_prerotated=True, original_pdf_rotation=90)

        assert determine_mode(result, 2814, 3864) == (True, True)

    def test_prerotated_image_on_an_unrotated_page_does_not(self):
        result = _page(image_prerotated=True, original_pdf_rotation=0)

        assert determine_mode(result, 2814, 3864) == (False, False)


class TestExportFormat:
    def test_a_requested_format_replaces_the_image_without_geometry_change(self):
        """Replacing the image is not a geometry change."""
        use_processed, geometry_changed = determine_mode(_page(), 2814, 3864, export_format="jpeg")

        assert use_processed is True
        assert geometry_changed is False

    @pytest.mark.parametrize("export_format", ["original", ""])
    def test_the_default_formats_do_not_replace_the_image(self, export_format):
        assert determine_mode(_page(), 2814, 3864, export_format=export_format) == (False, False)


class TestDiagnosticLogging:
    def test_the_geometry_change_line_is_logged_only_when_it_changed(self, caplog):
        """That INFO line is the field-diagnosis tool for displaced text.

        The level is set on ``BigOcrPdf`` by name: another test in the suite
        raises that logger's own level, which a root-only ``at_level`` would
        not override.
        """
        import logging

        caplog.set_level(logging.INFO, logger="BigOcrPdf")

        determine_mode(_page(geometry_applied=True), 2814, 3864)
        assert "geometry/coordinate change" in caplog.text

        caplog.clear()
        determine_mode(_page(), 2814, 3864)
        assert "geometry/coordinate change" not in caplog.text


@pytest.mark.parametrize(("export_format", "keeps_original"), [("original", True), ("jpeg", False)])
def test_oversized_page_keeps_its_image(export_format: str, keeps_original: bool):
    """The page was read from a render: lay the text over the original scan,
    unless the user asked for the images to be re-encoded anyway."""
    mixin = ChunkedOCRMixin()
    mixin.config = SimpleNamespace(
        image_export_format=export_format,
        enable_deskew=True,
        enable_perspective_correction=False,
        enable_baseline_dewarp=False,
    )
    mixin.extractor = SimpleNamespace(masked_pages=set(), rendered_pages={1}, oversized_pages={1})
    rotation = SimpleNamespace(
        deleted=False, included_for_ocr=True, original_pdf_rotation=0, image_turn=(False, 0)
    )

    [item] = mixin._build_chunk_work_items(
        ["page_1.png"], 0, [rotation], set(), {}, SimpleNamespace(downscale_probmap=0)
    )

    assert (item["skip_geometric"], item["force_overlay"]) == (keeps_original, keeps_original)


@pytest.mark.parametrize(
    ("page_rendered", "expected_rect"), [(True, None), (False, (100, 50, 400, 600))]
)
def test_page_render_text_spans_the_page(tmp_path, page_rendered: bool, expected_rect):
    """OCR of a page render is measured on the whole page, not on its image."""
    backend = _ModeBackend()
    backend._render_ocr_to_page = Mock(return_value=0.0)
    out = tmp_path / "page.png"
    out.write_bytes(b"")
    result = {"success": True, "temp_out_path": str(out), "proc_w": 1240, "proc_h": 1754}
    result.update(_page(orig_w=1240, orig_h=1754), ocr_raw={"boxes": [], "txts": [], "scores": []})
    page_info = {"rotation": 0, "mediabox": [0, 0, 595, 842], "image_rect": (100, 50, 400, 600)}
    work_item = {"img_path": str(out), "page_rendered": page_rendered}

    backend._process_page_result(None, result, work_item, [page_info], 1, ProcessingStats())

    assert backend._render_ocr_to_page.call_args.kwargs["image_rect"] == expected_rect


def test_blank_page_reads_as_empty():
    stats = ProcessingStats()

    lines = BackendTextLayerMixin._page_ocr_results(
        {"ocr_raw": {"boxes": [], "txts": [], "scores": []}}, 1, stats
    )

    assert lines == []
    assert stats.failed_pages == {}


def test_unread_page_is_recorded_as_failed():
    stats = ProcessingStats()

    lines = BackendTextLayerMixin._page_ocr_results(
        {"ocr_raw": None, "ocr_error": "worker crashed"}, 4, stats
    )

    assert lines == []
    assert stats.failed_pages == {4: "worker crashed"}
